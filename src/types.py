from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

# MiniGrid with FlatObsWrapper: 7×7×3 = 147-dim float vector
Observation = torch.Tensor   # shape (147,)
ObsSequence = torch.Tensor   # shape (N, 147)
Action = int                 # discrete 0-6 (MiniGrid)
TaskID = int                 # 0-indexed; -1 = unknown/uncertain
LayerIdx = int               # n: 0 = primitive, 1 = first meta layer, …
ColumnIdx = int              # m: 1-indexed within a layer


@dataclass
class AEOutput:
    reconstruction_error: float
    is_novel: bool
    latent_z: torch.Tensor


@dataclass
class GRUOutput:
    task_id: TaskID
    confidence: float
    all_probs: torch.Tensor


@dataclass
class ColumnOutput:
    action: Action
    value: float
    activations: dict  # int → torch.Tensor (layer index → hidden activation)


@dataclass
class OptionStepOutput:
    action: Action
    terminated: bool
    info: dict


@dataclass
class MetaControllerOutput:
    selected_option_id: int
    option: object  # Option — forward reference avoids circular import


@dataclass
class SystemState:
    current_task_id: TaskID
    active_column: Optional[object]   # Column | None
    active_option: Optional[object]   # Option | None
    is_training: bool
    ae_last_error: float = 0.0
    gru_confidence: float = 0.0
    n_columns: int = 0
    n_options: int = 0
