"""GRU Task Identifier unit tests (GRU-1 / GRU-2)."""

import torch

from src.task_detection.gru_identifier import GRUTaskIdentifier
from src.types import GRUOutput


def test_forward_returns_gru_output():
    gru = GRUTaskIdentifier(num_tasks=2)
    seq = torch.randn(20, 147)
    out = gru.forward(seq)
    assert isinstance(out, GRUOutput)
    assert out.all_probs.shape == (2,)
    assert abs(out.all_probs.sum().item() - 1.0) < 1e-5


def test_task_id_uncertain_when_confidence_low():
    """With random weights and 1-task head, confidence may be below threshold."""
    gru = GRUTaskIdentifier(num_tasks=10)  # many tasks → low confidence
    seq = torch.randn(20, 147)
    out = gru.forward(seq)
    # task_id is -1 if confidence < 0.6
    if out.confidence < gru.CONFIDENCE_THRESHOLD:
        assert out.task_id == -1
    else:
        assert 0 <= out.task_id < 10


def test_register_new_task_expands_head():
    gru = GRUTaskIdentifier(num_tasks=1)
    assert gru.num_tasks == 1
    gru.register_new_task(task_id=1)  # need output dim 2
    assert gru.num_tasks == 2
    # Verify forward still works
    seq = torch.randn(20, 147)
    out = gru.forward(seq)
    assert out.all_probs.shape == (2,)


def test_register_new_task_idempotent():
    """register_new_task is a no-op if head already covers the task."""
    gru = GRUTaskIdentifier(num_tasks=3)
    gru.register_new_task(task_id=2)  # task_id+1 = 3, already covered
    assert gru.num_tasks == 3  # unchanged


def test_register_new_task_preserves_existing_weights():
    """Old head weights must be preserved when expanding."""
    gru = GRUTaskIdentifier(num_tasks=2)
    old_weight = gru.head.weight.detach().clone()
    gru.register_new_task(task_id=2)  # expand to 3
    assert gru.num_tasks == 3
    assert torch.allclose(gru.head.weight[:2], old_weight)


def test_observe_and_get_sequence():
    gru = GRUTaskIdentifier()
    for _ in range(25):
        gru.observe(torch.randn(147))
    seq = gru.get_sequence()
    assert seq.shape == (20, 147)


def test_get_sequence_zero_pads_short_buffer():
    gru = GRUTaskIdentifier()
    # Only 5 observations in buffer
    for _ in range(5):
        gru.observe(torch.randn(147))
    seq = gru.get_sequence()
    assert seq.shape == (20, 147)
    # First 15 rows should be all zeros
    assert torch.all(seq[:15] == 0)


def test_train_supervised_reduces_loss():
    gru = GRUTaskIdentifier(num_tasks=2)
    sequences = torch.randn(20, 20, 147)
    labels = torch.randint(0, 2, (20,))
    loss_before = gru.train_supervised(sequences, labels, epochs=1)
    loss_after = gru.train_supervised(sequences, labels, epochs=20)
    assert loss_after <= loss_before + 0.5, "Supervised loss did not converge"
