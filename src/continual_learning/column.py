from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np
import torch

from src.types import Action, ColumnOutput, LayerIdx, ColumnIdx, Observation

if TYPE_CHECKING:
    from src.continual_learning.column_policy import ColumnPolicy


def _to_tensor(obs: Observation | np.ndarray) -> torch.Tensor:
    if isinstance(obs, torch.Tensor):
        return obs.float()
    return torch.tensor(obs, dtype=torch.float32)


class Column:
    """
    A single Progressive Network column: policy + lateral receiver + hierarchy slot.

    Coordinates use 1-indexed m (e.g. {0,1} is the first primitive column).

    sub_layer: always None in Phase 1. Set to a MetaController in Phase 2 to
    make this column a non-leaf node in the recursive hierarchy.
    """

    def __init__(
        self,
        n: LayerIdx,
        m: ColumnIdx,
        lateral_source: Optional[Column] = None,
        sub_layer=None,
    ) -> None:
        self.n = n
        self.m = m  # 1-indexed
        self.frozen = False
        self.lateral_source = lateral_source
        self.sub_layer = sub_layer  # None = leaf (Phase 1); MetaController in Phase 2
        self.policy: Optional[ColumnPolicy] = None  # set by ColumnTrainer after training

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def forward(self, obs: Observation | np.ndarray) -> ColumnOutput:
        if self.policy is None:
            raise RuntimeError(
                f"Column {{{self.n},{self.m}}}: policy is None. "
                "Call ColumnTrainer.train() or attach a policy before calling forward()."
            )
        obs_t = _to_tensor(obs)
        obs_np = obs_t.cpu().numpy()

        action, _ = self.policy.predict(obs_np, deterministic=not self.frozen)
        action = int(action)

        params = list(self.policy.parameters())
        device = params[0].device if params else torch.device("cpu")
        obs_tensor = obs_t.unsqueeze(0).to(device)  # match policy device
        with torch.no_grad():
            value = self.policy.predict_values(obs_tensor).item()

        activations = self.policy.mlp_extractor.get_activations()
        return ColumnOutput(action=action, value=value, activations=dict(activations))

    def freeze(self) -> None:
        """Freeze all policy weights. Forward pass still runs for lateral connections."""
        if self.policy is None:
            raise RuntimeError(f"Column {{{self.n},{self.m}}}: cannot freeze — policy is None.")
        self.frozen = True
        for p in self.policy.parameters():
            p.requires_grad_(False)

    def get_activations(self) -> dict[int, torch.Tensor]:
        """Return intermediate activations from the last forward pass."""
        if self.policy is None:
            return {}
        return self.policy.mlp_extractor.get_activations()

    def as_option(self) -> object:
        """Wrap this column as an Option (import deferred to avoid circular import)."""
        from src.options.option import Option
        return Option(column=self, option_id=(self.n, self.m))

    def __repr__(self) -> str:
        status = "frozen" if self.frozen else "active"
        lateral = f"←{{{self.lateral_source.n},{self.lateral_source.m}}}" if self.lateral_source else ""
        return f"Column({{{self.n},{self.m}}} {status}{lateral})"
