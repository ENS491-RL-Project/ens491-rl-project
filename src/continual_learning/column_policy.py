from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy

if TYPE_CHECKING:
    from src.continual_learning.column import Column


class LateralMlpExtractor(nn.Module):
    """
    Two-hidden-layer MLP feature extractor with optional lateral connections.

    Architecture: obs(147) → fc1(64) → fc2(64)
    Lateral: a linear projection from each of the source's hidden layers is
    added element-wise to the receiving hidden layer *after* its ReLU, i.e.
    h = relu(W x) + lat(h_source), matching v2 §7 and CLAUDE.md.

    SB3 2.x contract: must expose forward(), forward_actor(), forward_critic().
    - forward(features) → (pi_features, vf_features)  [called during PPO update]
    - forward_actor(features) → pi_features            [called during predict / get_distribution]
    - forward_critic(features) → vf_features           [called during predict_values]

    Lateral re-computation: during SB3's batched PPO update, forward_actor() re-runs
    the frozen source on the same features batch under torch.no_grad(). This ensures
    activations are correct for the training batch, not stale inference activations.
    """

    def __init__(self, feature_dim: int, lateral_source_column: Optional[Column] = None) -> None:
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 64)
        self.fc2 = nn.Linear(64, 64)

        self.lateral_source_column = lateral_source_column
        if lateral_source_column is not None:
            self.lat1 = nn.Linear(64, 64, bias=False)
            self.lat2 = nn.Linear(64, 64, bias=False)

        # SB3 contract
        self.latent_dim_pi = 64
        self.latent_dim_vf = 64

        self._acts: dict[int, torch.Tensor] = {}

    def extract_with_acts(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """
        Core MLP + lateral computation. Returns (h2, activations).

        [A: F15] Activations are *returned*, not stashed on the module. A frozen
        node is a single shared `nn.Module` referenced by every `OptionEnv` copy
        in a vectorized run, so a `self._acts` dict is last-writer-wins: one env
        could read activations another env had just overwritten. `_acts` is still
        updated, but only as a convenience cache for the GUI -- nothing in the
        forward path reads it.
        """
        features = features.to(self.fc1.weight.device)
        lat_acts: dict[int, torch.Tensor] = {}
        if self.lateral_source_column is not None:
            # Re-run the frozen source on this batch so activations match it.
            # The source has requires_grad_(False), so no gradient flows into it.
            with torch.no_grad():
                _, lat_acts = self.lateral_source_column.policy.mlp_extractor.extract_with_acts(
                    features
                )

        h1 = F.relu(self.fc1(features))
        if 0 in lat_acts:
            h1 = h1 + self.lat1(lat_acts[0])

        h2 = F.relu(self.fc2(h1))
        if 1 in lat_acts:
            h2 = h2 + self.lat2(lat_acts[1])

        acts = {0: h1.detach(), 1: h2.detach()}
        self._acts = acts  # GUI cache only; never read by the forward path
        return h2, acts

    def _extract(self, features: torch.Tensor) -> torch.Tensor:
        """Backwards-compatible shim: the h2 half of `extract_with_acts`."""
        return self.extract_with_acts(features)[0]

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """SB3 2.x: called during PPO evaluate_actions(). Returns (pi, vf)."""
        h, _ = self.extract_with_acts(features)
        return h, h

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        """SB3 2.x: called during predict() / get_distribution()."""
        return self.extract_with_acts(features)[0]

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """SB3 2.x: called during predict_values()."""
        return self.extract_with_acts(features)[0]

    def get_activations(self) -> dict[int, torch.Tensor]:
        """The last computed activations. For the GUI; see `extract_with_acts`."""
        return self._acts


class ColumnPolicy(ActorCriticPolicy):
    """
    SB3 ActorCriticPolicy backed by LateralMlpExtractor.

    Pass lateral_source_column via policy_kwargs:
        PPO(ColumnPolicy, env, policy_kwargs={"lateral_source_column": col})
    """

    def __init__(self, *args, lateral_source_column: Optional[Column] = None, **kwargs) -> None:
        self._lateral_source_column = lateral_source_column
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = LateralMlpExtractor(
            feature_dim=self.features_dim,
            lateral_source_column=self._lateral_source_column,
        )
