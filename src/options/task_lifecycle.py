from __future__ import annotations

import random
from collections import deque
from typing import Optional

import numpy as np
import torch

from src.continual_learning.column import Column
from src.options.meta_controller import MetaController
from src.options.option import Option
from src.task_detection.autoencoder import AutoencoderModule
from src.task_detection.gru_identifier import GRUTaskIdentifier
from src.types import (
    Action, LayerIdx, ColumnIdx, Observation, SystemState, TaskID,
)


def _to_tensor(obs: Observation | np.ndarray | object) -> torch.Tensor:
    if isinstance(obs, torch.Tensor):
        return obs.float()
    return torch.tensor(np.asarray(obs), dtype=torch.float32)


class TaskLifecycleManager:
    """
    Orchestrates the full AE → GRU → MetaController → Option → Action pipeline.

    Layer convention (fixed):
        - Primitive columns live at n=0 with 1-indexed m.
        - Root MetaController lives at n=1, managing all n=0 options.
        - on_column_stabilized() adds options to meta_controllers[column.n + 1].

    Phase 1 limitations (known):
        - Single-level execution only; no recursive sub_layer traversal.
        - _active_option is a single shared slot.
          TODO (Phase 2): replace with _active_options: dict[LayerIdx, Option]
          so each MetaController tracks its own active option independently,
          which is required for correct recursive multi-level hierarchy.
    """

    NOVELTY_BUFFER_SIZE = 100

    def __init__(self) -> None:
        self.ae = AutoencoderModule()
        self.gru = GRUTaskIdentifier()
        self.columns: dict[tuple[LayerIdx, ColumnIdx], Column] = {}
        self.meta_controllers: dict[LayerIdx, MetaController] = {}

        self._current_task_id: TaskID = -1
        self._active_option: Optional[Option] = None
        self._training_column: Optional[Column] = None
        self._n_tasks_seen: int = 0
        self._recent_ae_errors: deque[float] = deque(maxlen=self.NOVELTY_BUFFER_SIZE)
        self._last_ae_error: float = 0.0
        self._last_gru_confidence: float = 0.0

    # ------------------------------------------------------------------
    # Main step — called every environment step
    # ------------------------------------------------------------------

    def step(self, obs: Observation) -> Action:
        obs_t = _to_tensor(obs)

        # 1. AE novelty check
        ae_out = self.ae.forward(obs_t)
        self._last_ae_error = ae_out.reconstruction_error
        self._recent_ae_errors.append(ae_out.reconstruction_error)

        if ae_out.is_novel:
            # Guardrail: do not open another column while one is training
            if self._training_column is not None:
                return random.randint(0, 6)
            self.on_novel_task_detected()
            return random.randint(0, 6)

        # 2. GRU task identification
        self.gru.observe(obs_t)
        gru_out = self.gru.forward(self.gru.get_sequence())
        self._current_task_id = gru_out.task_id
        self._last_gru_confidence = gru_out.confidence

        # 3. Route through root MetaController (n=1 manages n=0 primitives)
        root_mc = self.meta_controllers.get(1)
        if root_mc is None or not root_mc.available_options:
            return random.randint(0, 6)

        # 4. Single-level leaf execution (Phase 1)
        #    Phase 2 NOTE: recursive non-leaf execution requires sub_layer support
        #    and per-MetaController active option tracking — not implemented here.
        if self._active_option is None:
            mc_out = root_mc.select_option(obs_t)
            self._active_option = mc_out.option
            self._active_option.reset()

        opt_out = self._active_option.step(obs_t)
        if opt_out.terminated:
            self._active_option = None

        return opt_out.action

    def on_reward(self, reward: float, done: bool) -> None:
        """Feed env reward to the root MetaController on its separate channel."""
        root_mc = self.meta_controllers.get(1)
        if root_mc is not None and self._active_option is not None:
            obs_placeholder = torch.zeros(147)  # MC reward channel; obs unused in Progress I
            root_mc.step(obs_placeholder, reward, done)

    # ------------------------------------------------------------------
    # Lifecycle events
    # ------------------------------------------------------------------

    def on_novel_task_detected(self) -> None:
        """Open a new primitive column. Caller is responsible for training it."""
        m = self._n_tasks_seen + 1  # 1-indexed
        n = 0
        lateral = self.columns.get((n, m - 1)) if m > 1 else None
        col = Column(n=n, m=m, lateral_source=lateral)
        self.columns[(n, m)] = col
        self._n_tasks_seen += 1
        # Expand GRU head for the new task (0-indexed task id)
        self.gru.register_new_task(self._n_tasks_seen - 1)
        self._training_column = col

    def on_column_stabilized(self, column: Column) -> None:
        """
        Freeze column, wrap as option, add to MetaController at layer n+1.
        Only called after StabilizationMonitor.is_stable() returns True.
        """
        column.freeze()
        option = column.as_option()
        mc_layer: LayerIdx = column.n + 1
        if mc_layer not in self.meta_controllers:
            self.meta_controllers[mc_layer] = MetaController(mc_layer)
        self.meta_controllers[mc_layer].add_option(option)
        self._training_column = None

    # ------------------------------------------------------------------
    # State query
    # ------------------------------------------------------------------

    def get_state(self) -> SystemState:
        total_options = sum(mc.n_options for mc in self.meta_controllers.values())
        return SystemState(
            current_task_id=self._current_task_id,
            active_column=self._training_column,
            active_option=self._active_option,
            is_training=self._training_column is not None,
            ae_last_error=self._last_ae_error,
            gru_confidence=self._last_gru_confidence,
            n_columns=len(self.columns),
            n_options=total_options,
        )
