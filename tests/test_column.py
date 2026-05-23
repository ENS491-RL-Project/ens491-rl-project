"""
Column unit tests (PN-3 / PN-4 scope).
Uses a MockPolicy so tests run without SB3 or gym.
"""

import pytest
import torch

from src.continual_learning.column import Column
from src.types import ColumnOutput


class MockMlpExtractor:
    def __init__(self):
        self._acts = {0: torch.randn(64), 1: torch.randn(64)}

    def get_activations(self):
        return self._acts

    def __call__(self, x):
        return self._acts[1], self._acts[1]


class MockPolicy:
    """Minimal policy stub implementing the Column interface contract."""

    def __init__(self):
        self.mlp_extractor = MockMlpExtractor()
        self._params = [torch.nn.Parameter(torch.randn(3, 3))]

    def predict(self, obs, deterministic=True):
        return 3, None  # always action 3

    def predict_values(self, obs):
        return torch.tensor([[0.5]])

    def parameters(self):
        return iter(self._params)

    def named_parameters(self):
        return iter([("w", self._params[0])])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_column_forward_raises_without_policy():
    col = Column(n=0, m=1)
    with pytest.raises(RuntimeError, match="policy is None"):
        col.forward(torch.randn(147))


def test_column_forward_returns_column_output():
    col = Column(n=0, m=1)
    col.policy = MockPolicy()
    out = col.forward(torch.randn(147))
    assert isinstance(out, ColumnOutput)
    assert isinstance(out.action, int)
    assert 0 <= out.action <= 6
    assert isinstance(out.value, float)
    assert isinstance(out.activations, dict)


def test_column_freeze_sets_flag():
    col = Column(n=0, m=1)
    col.policy = MockPolicy()
    assert col.frozen is False
    col.freeze()
    assert col.frozen is True


def test_column_freeze_disables_grad():
    col = Column(n=0, m=1)
    col.policy = MockPolicy()
    # Ensure param starts requiring grad
    for p in col.policy.parameters():
        p.requires_grad_(True)
    col.freeze()
    for p in col.policy.parameters():
        assert not p.requires_grad


def test_column_get_activations_nonempty():
    col = Column(n=0, m=1)
    col.policy = MockPolicy()
    col.forward(torch.randn(147))
    acts = col.get_activations()
    assert len(acts) > 0


def test_column_freeze_raises_without_policy():
    col = Column(n=0, m=1)
    with pytest.raises(RuntimeError, match="policy is None"):
        col.freeze()


def test_column_sub_layer_field_exists():
    """sub_layer must exist on every Column — Phase 2 recursive hook."""
    col = Column(n=0, m=1)
    assert hasattr(col, "sub_layer")
    assert col.sub_layer is None


def test_column_1indexed_m():
    """First column is {0,1}, second is {0,2}."""
    col1 = Column(n=0, m=1)
    col2 = Column(n=0, m=2, lateral_source=col1)
    assert col1.m == 1
    assert col2.m == 2
    assert col2.lateral_source is col1


def test_column_as_option():
    from src.options.option import Option
    col = Column(n=0, m=1)
    col.policy = MockPolicy()
    opt = col.as_option()
    assert isinstance(opt, Option)
    assert opt.option_id == (0, 1)
