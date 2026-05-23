"""Autoencoder unit tests (AE-1 / AE-2)."""

import torch

from src.task_detection.autoencoder import AutoencoderModule
from src.types import AEOutput


def test_forward_returns_ae_output():
    ae = AutoencoderModule()
    obs = torch.randn(147)
    out = ae.forward(obs)
    assert isinstance(out, AEOutput)
    assert isinstance(out.reconstruction_error, float)
    assert isinstance(out.is_novel, bool)
    assert out.latent_z.shape == (32,)


def test_novelty_flag_respects_threshold():
    ae = AutoencoderModule(threshold=1e6)   # enormous threshold → never novel
    obs = torch.randn(147)
    out = ae.forward(obs)
    assert out.is_novel is False

    ae2 = AutoencoderModule(threshold=0.0)  # zero threshold → always novel
    out2 = ae2.forward(obs)
    assert out2.is_novel is True


def test_train_on_batch_reduces_loss():
    ae = AutoencoderModule()
    obs_batch = torch.randn(16, 147)
    loss_before = ae.train_on_batch(obs_batch)
    # Train for many steps
    for _ in range(50):
        ae.train_on_batch(obs_batch)
    loss_after = ae.train_on_batch(obs_batch)
    assert loss_after < loss_before, "Loss did not decrease after training"


def test_update_threshold_changes_value():
    ae = AutoencoderModule(threshold=0.05)
    original = ae.threshold
    # Use a list of low errors — threshold should be set to mean + 2*std
    errors = [0.01, 0.012, 0.011, 0.013, 0.010]
    ae.update_threshold(errors)
    assert ae.threshold != original
    assert ae.threshold > 0


def test_update_threshold_noop_on_single_error():
    """update_threshold requires at least 2 values (std is undefined for 1)."""
    ae = AutoencoderModule(threshold=0.05)
    ae.update_threshold([0.01])
    assert ae.threshold == 0.05  # unchanged


def test_compute_errors_returns_per_sample_list():
    ae = AutoencoderModule()
    obs_batch = torch.randn(8, 147)
    errors = ae.compute_errors(obs_batch)
    assert len(errors) == 8
    assert all(isinstance(e, float) for e in errors)


def test_encode_decode_shape():
    ae = AutoencoderModule()
    obs = torch.randn(147)
    z = ae.encode(obs)
    assert z.shape == (32,)
    recon = ae.decode(z)
    assert recon.shape == (147,)
