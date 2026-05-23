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
    added element-wise before the ReLU activation.

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

    def _extract(self, features: torch.Tensor) -> torch.Tensor:
        """Core MLP + lateral computation. Stores activations; returns h2."""
        features = features.to(self.fc1.weight.device)
        lat_acts: dict[int, torch.Tensor] = {}
        if self.lateral_source_column is not None:
            # Re-run frozen source on this batch to get matching activations.
            # Source has requires_grad_(False) so no gradient flows into it.
            with torch.no_grad():
                self.lateral_source_column.policy.mlp_extractor._extract(features)
            lat_acts = self.lateral_source_column.policy.mlp_extractor._acts

        h1 = F.relu(self.fc1(features))
        if lat_acts and 0 in lat_acts:
            h1 = h1 + self.lat1(lat_acts[0])
        self._acts[0] = h1.detach()

        h2 = F.relu(self.fc2(h1))
        if lat_acts and 1 in lat_acts:
            h2 = h2 + self.lat2(lat_acts[1])
        self._acts[1] = h2.detach()

        return h2

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """SB3 2.x: called during PPO evaluate_actions(). Returns (pi, vf)."""
        h = self._extract(features)
        return h, h

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        """SB3 2.x: called during predict() / get_distribution()."""
        return self._extract(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """SB3 2.x: called during predict_values()."""
        return self._extract(features)

    def get_activations(self) -> dict[int, torch.Tensor]:
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
