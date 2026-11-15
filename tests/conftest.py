import pytest


@pytest.fixture(autouse=True)
def isolated_training_runs(tmp_path, monkeypatch):
    """Tests exercise persistence without filling the experiment registry."""
    monkeypatch.setenv('ENS491_RUNS_ROOT', str(tmp_path / 'training_runs'))
