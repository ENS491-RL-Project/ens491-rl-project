"""Option wrapper unit tests (OPT-2)."""

import torch

from src.continual_learning.column import Column
from src.options.option import Option
from src.types import OptionStepOutput


class MockPolicy:
    def __init__(self):
        class _Ext:
            def get_activations(self):
                return {0: torch.randn(64), 1: torch.randn(64)}

        self.mlp_extractor = _Ext()

    def predict(self, obs, deterministic=True):
        return 2, None

    def predict_values(self, obs):
        return torch.tensor([[0.7]])

    def parameters(self):
        return iter([])

    def named_parameters(self):
        return iter([])


def _make_option() -> Option:
    col = Column(n=0, m=1)
    col.policy = MockPolicy()
    col.freeze()
    return Option(column=col, option_id=(0, 1))


def test_option_step_returns_option_step_output():
    opt = _make_option()
    out = opt.step(torch.randn(147))
    assert isinstance(out, OptionStepOutput)
    assert isinstance(out.action, int)
    assert isinstance(out.terminated, bool)
    assert "step_count" in out.info


def test_option_action_in_valid_range():
    opt = _make_option()
    for _ in range(10):
        out = opt.step(torch.randn(147))
        assert 0 <= out.action <= 6


def test_option_terminates_at_max_steps():
    opt = _make_option()
    opt.reset()
    out = None
    for _ in range(Option.MAX_STEPS):
        out = opt.step(torch.randn(147))
    assert out.terminated is True
    assert opt.step_count == Option.MAX_STEPS


def test_option_not_terminated_before_max_steps():
    opt = _make_option()
    for _ in range(Option.MAX_STEPS - 1):
        out = opt.step(torch.randn(147))
    assert out.terminated is False


def test_option_reset_clears_step_count():
    opt = _make_option()
    for _ in range(50):
        opt.step(torch.randn(147))
    opt.reset()
    assert opt.step_count == 0


def test_option_can_initiate_always_true():
    opt = _make_option()
    assert opt.can_initiate(torch.randn(147)) is True


def test_option_id_matches_column_coordinates():
    opt = _make_option()
    assert opt.option_id == (0, 1)
