from __future__ import annotations

import random
from typing import Optional, TYPE_CHECKING

from src.types import LayerIdx, MetaControllerOutput, Observation

if TYPE_CHECKING:
    from src.options.option import Option


class MetaController:
    """
    Selects options for a given hierarchy layer.

    Progress I: Epsilon-greedy selection with option history logging.
    PPO-based optimisation is deferred to after the integration milestone —
    the dynamic action space makes SB3 integration risky for Progress I.

    Layer convention: n=1 is the root MetaController managing primitive
    columns at n=0. Higher n = higher abstraction.
    """

    EPSILON_START = 1.0
    EPSILON_END = 0.1
    EPSILON_DECAY_STEPS = 500

    def __init__(self, n: LayerIdx) -> None:
        self.n = n
        self.available_options: list[Option] = []
        self._step_count: int = 0
        self.selection_history: list[dict] = []  # for GUI and debugging

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def select_option(self, obs: Observation) -> MetaControllerOutput:
        if not self.available_options:
            raise RuntimeError(
                f"MetaController n={self.n} has no available options. "
                "Call add_option() before select_option()."
            )
        self._step_count += 1
        if random.random() < self._epsilon():
            idx = random.randrange(len(self.available_options))
        else:
            # Placeholder for learned selection: currently picks the first option.
            idx = 0
        option = self.available_options[idx]
        self.selection_history.append({
            "mc_step": self._step_count,
            "option_id": option.option_id,
            "reward": None,
        })
        return MetaControllerOutput(selected_option_id=idx, option=option)

    def add_option(self, option: Option) -> None:
        """Add a stabilised option. Does not trigger retraining."""
        self.available_options.append(option)

    def step(self, obs: Observation, reward: float, done: bool) -> None:
        """
        Called after each env step with the env reward on this layer's channel.
        Progress I: logs reward only.
        Phase 2: triggers PPO update.
        """
        if self.selection_history and self.selection_history[-1]["reward"] is None:
            self.selection_history[-1]["reward"] = reward

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _epsilon(self) -> float:
        return max(
            self.EPSILON_END,
            self.EPSILON_START
            - self._step_count * (self.EPSILON_START - self.EPSILON_END) / self.EPSILON_DECAY_STEPS,
        )

    @property
    def n_options(self) -> int:
        return len(self.available_options)

    def __repr__(self) -> str:
        return f"MetaController(n={self.n}, options={self.n_options}, steps={self._step_count})"
