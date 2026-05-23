"""MetaController unit tests (OPT-1)."""

import pytest
import torch

from src.continual_learning.column import Column
from src.options.meta_controller import MetaController
from src.options.option import Option
from src.types import MetaControllerOutput


class MockPolicy:
    def __init__(self):
        class _Ext:
            def get_activations(self):
                return {0: torch.randn(64), 1: torch.randn(64)}
        self.mlp_extractor = _Ext()

    def predict(self, obs, deterministic=True):
        return 0, None

    def predict_values(self, obs):
        return torch.tensor([[0.5]])

    def parameters(self):
        return iter([])

    def named_parameters(self):
        return iter([])


def _make_option(n: int, m: int) -> Option:
    col = Column(n=n, m=m)
    col.policy = MockPolicy()
    col.freeze()
    return Option(column=col, option_id=(n, m))


def test_select_option_raises_without_options():
    mc = MetaController(n=1)
    with pytest.raises(RuntimeError, match="no available options"):
        mc.select_option(torch.randn(147))


def test_add_option_increases_count():
    mc = MetaController(n=1)
    assert mc.n_options == 0
    mc.add_option(_make_option(0, 1))
    assert mc.n_options == 1
    mc.add_option(_make_option(0, 2))
    assert mc.n_options == 2


def test_select_option_returns_valid_output():
    mc = MetaController(n=1)
    opt = _make_option(0, 1)
    mc.add_option(opt)
    out = mc.select_option(torch.randn(147))
    assert isinstance(out, MetaControllerOutput)
    assert out.option is opt
    assert out.selected_option_id == 0


def test_select_option_id_in_valid_range():
    mc = MetaController(n=1)
    for i in range(1, 4):
        mc.add_option(_make_option(0, i))
    obs = torch.randn(147)
    for _ in range(50):
        out = mc.select_option(obs)
        assert 0 <= out.selected_option_id < 3


def test_selection_history_logged():
    mc = MetaController(n=1)
    mc.add_option(_make_option(0, 1))
    mc.select_option(torch.randn(147))
    assert len(mc.selection_history) == 1
    assert "option_id" in mc.selection_history[0]


def test_step_logs_reward():
    mc = MetaController(n=1)
    mc.add_option(_make_option(0, 1))
    mc.select_option(torch.randn(147))
    mc.step(torch.randn(147), reward=0.9, done=False)
    assert mc.selection_history[-1]["reward"] == 0.9


def test_layer_index_stored():
    mc = MetaController(n=1)
    assert mc.n == 1
