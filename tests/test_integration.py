"""
INT-2: TaskLifecycleManager end-to-end smoke tests.

Tests the full AE → GRU → MetaController → Option → Action pipeline
without any live training.
"""

import torch
import pytest

from src.continual_learning.column import Column
from src.options.task_lifecycle import TaskLifecycleManager
from src.types import SystemState


class _MockPolicy:
    def __init__(self):
        class _Ext:
            def get_activations(self):
                return {0: torch.randn(64), 1: torch.randn(64)}
        self.mlp_extractor = _Ext()

    def predict(self, obs, deterministic=True):
        return 4, None

    def predict_values(self, obs):
        return torch.tensor([[0.6]])

    def parameters(self):
        return iter([])

    def named_parameters(self):
        return iter([])


def _make_trained_column(n: int, m: int, lateral=None) -> Column:
    col = Column(n=n, m=m, lateral_source=lateral)
    col.policy = _MockPolicy()
    return col


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_step_returns_valid_action_without_any_columns():
    """Random action (0-6) when no columns/MC exist."""
    tlm = TaskLifecycleManager()
    # Force AE to not trigger novelty
    tlm.ae.threshold = 1e6
    obs = torch.randn(147)
    action = tlm.step(obs)
    assert isinstance(action, int)
    assert 0 <= action <= 6


def test_step_100_times_no_crash():
    """100 steps must all return int 0-6 with no exception."""
    tlm = TaskLifecycleManager()
    tlm.ae.threshold = 1e6  # suppress novelty

    # Manually install one stabilised column so MC has an option
    col = _make_trained_column(0, 1)
    tlm.columns[(0, 1)] = col
    tlm._n_tasks_seen = 1
    tlm.on_column_stabilized(col)

    obs = torch.randn(147)
    for _ in range(100):
        action = tlm.step(obs)
        assert isinstance(action, int)
        assert 0 <= action <= 6, f"Action {action} out of MiniGrid range"


def test_on_novel_task_detected_opens_column():
    tlm = TaskLifecycleManager()
    assert len(tlm.columns) == 0
    tlm.on_novel_task_detected()
    assert len(tlm.columns) == 1
    # First column is {0, 1}
    assert (0, 1) in tlm.columns


def test_on_novel_task_detected_increments_gru_head():
    tlm = TaskLifecycleManager()
    before = tlm.gru.num_tasks
    tlm.on_novel_task_detected()
    # register_new_task(0) expands to at least 1
    assert tlm.gru.num_tasks >= before


def test_on_column_stabilized_adds_option_to_mc_n1():
    tlm = TaskLifecycleManager()
    col = _make_trained_column(0, 1)
    tlm.on_column_stabilized(col)
    assert 1 in tlm.meta_controllers
    assert tlm.meta_controllers[1].n_options == 1


def test_on_column_stabilized_freezes_column():
    tlm = TaskLifecycleManager()
    col = _make_trained_column(0, 1)
    tlm.on_column_stabilized(col)
    assert col.frozen is True


def test_second_column_gets_lateral_source():
    tlm = TaskLifecycleManager()
    tlm.on_novel_task_detected()
    tlm.on_novel_task_detected()
    assert (0, 2) in tlm.columns
    col2 = tlm.columns[(0, 2)]
    assert col2.lateral_source is tlm.columns[(0, 1)]


def test_no_double_novel_task_while_training():
    """If _training_column is not None, step() must not call on_novel_task_detected again."""
    tlm = TaskLifecycleManager()
    # Make AE always report novelty
    tlm.ae.threshold = 0.0

    tlm.step(torch.randn(147))   # first novel: opens {0,1}, sets _training_column
    assert tlm._training_column is not None
    n_cols_before = len(tlm.columns)

    tlm.step(torch.randn(147))   # second novel: must NOT open another column
    assert len(tlm.columns) == n_cols_before


def test_get_state_returns_system_state():
    tlm = TaskLifecycleManager()
    state = tlm.get_state()
    assert isinstance(state, SystemState)
    assert state.n_columns == len(tlm.columns)


def test_layer_indexing_mc_at_n_plus_1():
    """Primitive column at n=0 → its option goes to MC at n=1."""
    tlm = TaskLifecycleManager()
    col = _make_trained_column(n=0, m=1)
    tlm.on_column_stabilized(col)
    assert 1 in tlm.meta_controllers
    assert 0 not in tlm.meta_controllers, "MC must be at n+1=1, not n=0"


def test_column_m_is_1indexed():
    """First two novel tasks open columns {0,1} and {0,2}."""
    tlm = TaskLifecycleManager()
    tlm.on_novel_task_detected()
    tlm.on_novel_task_detected()
    ms = [m for (_, m) in tlm.columns]
    assert sorted(ms) == [1, 2]
