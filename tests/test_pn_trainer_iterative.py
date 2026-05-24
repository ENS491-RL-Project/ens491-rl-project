"""
PN-4 trainer API tests: iterative training (train_until_stable) and checkpoint save/load.

Runs one tiny chunk (512 steps) to validate the API surface — not a training quality test.

Verification commands (use PYTHONNOUSERSITE to avoid user-site package conflicts):
    $env:PYTHONNOUSERSITE="1"
    python -m pytest tests/test_pn_trainer_iterative.py -v
    python scripts/train_pn4.py --smoke
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

import shutil
from pathlib import Path

from src.continual_learning.column import Column
from src.continual_learning.pn_trainer import ColumnTrainer


def test_train_until_stable_exits_on_max_total():
    """train_until_stable stops at max_total and sets column.policy."""
    col = Column(n=0, m=1)
    trainer = ColumnTrainer(col, "MiniGrid-Empty-8x8-v0", verbose=0)
    trainer.train_until_stable(chunk_size=512, max_total=512)
    assert col.policy is not None, "column.policy must be set after train_until_stable"


def test_save_creates_expected_files():
    """save() creates model.zip, policy_state_dict.pt, and meta.json in the given directory."""
    col = Column(n=0, m=1)
    trainer = ColumnTrainer(col, "MiniGrid-Empty-8x8-v0", verbose=0)
    trainer.train_until_stable(chunk_size=512, max_total=512)

    run_dir = Path("runs/test_pn4_tmp")
    try:
        trainer.save(run_dir)
        assert (run_dir / "model.zip").exists(), "model.zip missing"
        assert (run_dir / "policy_state_dict.pt").exists(), "policy_state_dict.pt missing"
        assert (run_dir / "meta.json").exists(), "meta.json missing"
    finally:
        if run_dir.exists():
            shutil.rmtree(run_dir)
