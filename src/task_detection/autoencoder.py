from __future__ import annotations

import statistics
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.types import AEOutput, Observation


def _to_tensor(obs: Observation | object) -> torch.Tensor:
    if isinstance(obs, torch.Tensor):
        return obs.float()
    import numpy as np
    return torch.tensor(obs, dtype=torch.float32)


class AutoencoderModule(nn.Module):
    """
    MLP autoencoder for observation-level novelty detection.

    Architecture: 147 → 64 → 32 (latent) → 64 → 147
    Novelty: reconstruction_error (MSE) > threshold → is_novel = True

    Calibration protocol for demo:
        1. Collect familiar-task observations into a buffer.
        2. Call train_on_batch() repeatedly on that buffer.
        3. Call update_threshold() on the buffer's reconstruction errors.
        4. Freeze threshold (do not call update_threshold() again during inference).
           Novel observations inflate the threshold and weaken detection.
    """

    def __init__(
        self,
        obs_dim: int = 147,
        latent_dim: int = 32,
        threshold: float = 0.05,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.threshold = threshold

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, obs_dim),
        )
        self._optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def encode(self, obs: Observation) -> torch.Tensor:
        x = _to_tensor(obs)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.encoder(x).squeeze(0)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.decoder(z).squeeze(0)

    def forward(self, obs: Observation) -> AEOutput:
        x = _to_tensor(obs)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        z = self.encoder(x)
        recon = self.decoder(z)
        error = F.mse_loss(recon, x).item()
        return AEOutput(
            reconstruction_error=error,
            is_novel=error > self.threshold,
            latent_z=z.squeeze(0).detach(),
        )

    def train_on_batch(self, obs_batch: torch.Tensor) -> float:
        """One gradient step on a batch. Returns mean MSE loss."""
        self.train()
        self._optimizer.zero_grad()
        z = self.encoder(obs_batch)
        recon = self.decoder(z)
        loss = F.mse_loss(recon, obs_batch)
        loss.backward()
        self._optimizer.step()
        self.eval()
        return loss.item()

    def update_threshold(self, recent_errors: list[float]) -> None:
        """
        Adaptive threshold: mean + 2*std of familiar-task errors.
        Only call this on familiar-task error distributions.
        """
        if len(recent_errors) < 2:
            return
        mean = statistics.mean(recent_errors)
        std = statistics.stdev(recent_errors)
        self.threshold = mean + 2.0 * std

    def compute_errors(self, obs_batch: torch.Tensor) -> list[float]:
        """Compute per-sample reconstruction errors for calibration."""
        self.eval()
        with torch.no_grad():
            z = self.encoder(obs_batch)
            recon = self.decoder(z)
            errors = F.mse_loss(recon, obs_batch, reduction="none").mean(dim=1)
        return errors.tolist()
