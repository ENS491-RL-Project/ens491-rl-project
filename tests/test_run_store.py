import json
from pathlib import Path

import pytest
import torch

from src.skills.node import SkillNode, NEVER
from src.skills.registry import NodeRegistry
from src.training.run_store import record_training, track_model
from src.types import NodeStatus


class FakeModel:
    num_timesteps = 17
    def save(self, path):
        Path(path).write_bytes(b'model with optimizer')
    def get_env(self):
        return None


@record_training
def attempt(node, *, stable=False, fail=False, run_root=None):
    node.policy = torch.nn.Linear(2, 2)
    node.status = NodeStatus.TRAINING
    model = FakeModel()
    track_model(model)
    if fail:
        raise RuntimeError('injected training failure')
    if stable:
        node.status = NodeStatus.STABLE
    return {'stable': stable}, model


def test_runs_are_unique_and_budget_exhaustion_keeps_weights(tmp_path):
    reports = [attempt(SkillNode('same', goal=NEVER), run_root=tmp_path)[0] for _ in range(2)]
    assert reports[0]['run_dir'] != reports[1]['run_dir']
    for report in reports:
        p = Path(report['run_dir'])
        assert (p / 'model.zip').is_file()
        assert (p / 'policy_state_dict.pt').is_file()
        assert json.loads((p / 'run.json').read_text())['state'] == 'budget_exhausted'
        assert json.loads((p / 'complete.json').read_text())['sha256']['policy_state_dict.pt']


def test_failure_is_archived_and_original_error_is_raised(tmp_path):
    with pytest.raises(RuntimeError, match='injected training failure'):
        attempt(SkillNode('failed', goal=NEVER), fail=True, run_root=tmp_path)
    p, = tmp_path.iterdir()
    data = json.loads((p / 'run.json').read_text())
    assert data['state'] == 'failed' and data['collected_timesteps'] == 17
    assert (p / 'policy_state_dict.pt').is_file() and (p / 'complete.json').is_file()


def test_freeze_requires_stability_and_training_rejects_frozen(tmp_path):
    node = SkillNode('x', goal=NEVER)
    node.policy = torch.nn.Linear(2, 2)
    for status in [NodeStatus.DECLARED, NodeStatus.TRAINING, NodeStatus.ABANDONED]:
        node.status = status
        with pytest.raises(RuntimeError, match='STABLE'):
            node.freeze()
    node.status = NodeStatus.STABLE
    node.freeze()
    node.freeze()  # idempotent
    with pytest.raises(RuntimeError, match='FROZEN'):
        attempt(node, run_root=tmp_path)


def test_registry_does_not_replace_existing_checkpoint(tmp_path):
    node = SkillNode('x', goal=NEVER)
    node.policy = torch.nn.Linear(2, 2)
    node.status = NodeStatus.STABLE
    node.freeze()
    registry = NodeRegistry(tmp_path)
    path = registry.save(node)
    original = (path / 'policy_state_dict.pt').read_bytes()
    with pytest.raises(FileExistsError):
        registry.save(node)
    assert (path / 'policy_state_dict.pt').read_bytes() == original
