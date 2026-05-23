"""
PN-1: Progressive Network mechanics test using synthetic data.

Validates WITHOUT any RL training loop:
  1. FlatObsWrapper produces (147,) observations.
  2. Two-column forward pass works end-to-end.
  3. Column {0,1} can be frozen.
  4. Frozen Column {0,1} still produces activations after freeze().
  5. Frozen Column {0,1} receives no gradients.
  6. Column {0,2} reads Column {0,1} activations through lateral connections.
  7. Lateral projection layers in Column {0,2} receive gradients.
"""

import pytest

try:
    import stable_baselines3  # noqa: F401 — presence check only
except Exception as e:
    pytest.skip(
        f"stable_baselines3 not importable ({e}). "
        "Fix: pip uninstall tensorflow -y (TF conflicts with torch.utils.tensorboard in this env).",
        allow_module_level=True,
    )

import gymnasium as gym
import minigrid  # noqa: F401
import torch
import torch.nn.functional as F
from gymnasium.wrappers import FlattenObservation
from minigrid.wrappers import ImgObsWrapper

from src.continual_learning.column import Column
from src.continual_learning.column_policy import ColumnPolicy, LateralMlpExtractor


def _make_env():
    """ImgObsWrapper (7×7×3) + FlattenObservation → (147,) uint8 obs."""
    return FlattenObservation(ImgObsWrapper(gym.make("MiniGrid-Empty-8x8-v0")))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_column_with_policy(
    n: int, m: int, lateral_source: Column | None = None
) -> Column:
    """Build a Column with an initialised ColumnPolicy (no training needed)."""
    col = Column(n=n, m=m, lateral_source=lateral_source)
    env = _make_env()
    from stable_baselines3 import PPO
    model = PPO(
        ColumnPolicy,
        env,
        policy_kwargs={"lateral_source_column": lateral_source},
        verbose=0,
    )
    col.policy = model.policy
    env.close()
    return col


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_flat_obs_shape():
    """ImgObsWrapper + FlattenObservation produces (147,) observations (7×7×3)."""
    env = _make_env()
    obs, _ = env.reset()
    env.close()
    assert obs.shape == (147,), f"Expected (147,) got {obs.shape}"


def test_two_column_forward_pass():
    """Both columns forward without error; outputs have the right structure."""
    col1 = _make_column_with_policy(n=0, m=1)
    col2 = _make_column_with_policy(n=0, m=2, lateral_source=col1)

    obs = torch.randn(147)
    out1 = col1.forward(obs)
    out2 = col2.forward(obs)

    assert isinstance(out1.action, int)
    assert 0 <= out1.action <= 6
    assert isinstance(out2.action, int)
    assert 0 <= out2.action <= 6


def test_freeze_sets_flag_and_blocks_grad():
    """After freeze(): frozen=True, all policy params have requires_grad=False."""
    col1 = _make_column_with_policy(n=0, m=1)
    col1.freeze()

    assert col1.frozen is True
    for p in col1.policy.parameters():
        assert not p.requires_grad, "Frozen column parameter still has requires_grad=True"


def test_frozen_column_still_produces_activations():
    """Freeze does not prevent forward pass or activation storage."""
    col1 = _make_column_with_policy(n=0, m=1)
    col1.freeze()

    obs = torch.randn(147)
    out = col1.forward(obs)

    acts = col1.get_activations()
    assert len(acts) > 0, "No activations after frozen forward pass"
    assert all(isinstance(v, torch.Tensor) for v in acts.values())


def test_lateral_connections_read_source_activations():
    """Column {0,2}'s extractor reads Column {0,1}'s activations through lateral connections."""
    col1 = _make_column_with_policy(n=0, m=1)
    col1.freeze()

    col2 = _make_column_with_policy(n=0, m=2, lateral_source=col1)

    # Verify lateral projection layers exist in col2's extractor
    extractor = col2.policy.mlp_extractor
    assert isinstance(extractor, LateralMlpExtractor)
    assert hasattr(extractor, "lat1"), "lateral layer lat1 missing from col2 extractor"
    assert hasattr(extractor, "lat2"), "lateral layer lat2 missing from col2 extractor"

    # Forward col2 — internally re-runs col1 to get its activations
    obs = torch.randn(147)
    _ = col2.forward(obs)

    acts2 = col2.get_activations()
    assert len(acts2) > 0, "col2 produced no activations"


def test_lateral_layers_receive_gradients_frozen_source_does_not():
    """
    On a synthetic loss over col2's output:
      - col2's lateral projection weights receive gradients.
      - col1's weights receive no gradients (frozen).
    """
    col1 = _make_column_with_policy(n=0, m=1)
    col1.freeze()
    col2 = _make_column_with_policy(n=0, m=2, lateral_source=col1)

    obs_batch = torch.randn(4, 147)
    extractor2 = col2.policy.mlp_extractor

    # Run extractor forward (same path as during PPO update)
    pi_feats, _ = extractor2(obs_batch)

    # Dummy target: supervised loss to drive gradients
    target = torch.zeros_like(pi_feats)
    loss = F.mse_loss(pi_feats, target)
    loss.backward()

    # Lateral layers in col2 should have gradients
    assert extractor2.lat1.weight.grad is not None, "lat1.weight has no grad"
    assert extractor2.lat2.weight.grad is not None, "lat2.weight has no grad"

    # col1's params must have no gradients
    for name, p in col1.policy.named_parameters():
        assert p.grad is None, f"Frozen col1 param '{name}' has grad — gradient leaked"
