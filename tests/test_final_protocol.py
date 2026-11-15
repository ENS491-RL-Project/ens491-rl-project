import json
from pathlib import Path

import pytest
import torch

from src.envs.subgoal_env import NO_HOOK
from src.skills.node import SkillNode, NEVER
from src.training.final_protocol import reserve_seeds, lock_candidates, evaluate_locked
from src.training.run_store import TrainingRun


def fixture_runs(tmp_path):
    dirs, nodes = [], {}
    for seed in [42, 43, 44]:
        node = SkillNode('candidate', goal=NEVER, env_ids=['Toy-v0'])
        node.policy = torch.nn.Linear(2, 2)
        run = TrainingRun(node, {'seed': seed}, tmp_path / 'runs')
        run.finish('budget_exhausted', report={'seed': seed, 'env_ids': ['Toy-v0'], 'reset_hook': NO_HOOK.spec})
        dirs.append(run.path)
        nodes[run.run_id] = node
    return dirs, nodes


def test_reservation_is_disjoint_and_immutable(tmp_path):
    history = tmp_path / 'history'
    history.mkdir()
    (history / 'old.json').write_text(json.dumps({'seeds': [42, 10000, 20000, 2**28]}))
    path = tmp_path / 'final.json'
    manifest = reserve_seeds(path, [history], count=20)
    assert not set(manifest['seeds']) & {42, 10000, 20000, 2**28}
    assert len(set(manifest['seeds'])) == 20
    with pytest.raises(FileExistsError):
        reserve_seeds(path, [history])


def test_lock_rejects_incomplete_seed_group_and_changed_weights(tmp_path):
    manifest = tmp_path / 'final.json'
    reserve_seeds(manifest, [], count=3)
    dirs, _ = fixture_runs(tmp_path)
    with pytest.raises(ValueError, match='exactly training seeds'):
        lock_candidates(manifest, dirs[:2], tmp_path / 'lock.json')
    (dirs[0] / 'policy_state_dict.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Changed run artifact'):
        lock_candidates(manifest, dirs, tmp_path / 'lock.json')


def test_evaluation_requires_lock_and_runs_once(tmp_path, monkeypatch):
    manifest, lock = tmp_path / 'final.json', tmp_path / 'lock.json'
    reserve_seeds(manifest, [], count=3)
    dirs, nodes = fixture_runs(tmp_path)
    key = next(iter(nodes))
    calls = []
    def evaluate(node, factory, **kwargs):
        from src.training.evaluation import EvalResult
        calls.append(kwargs['seeds'])
        return EvalResult(3, 2, 2.0, kwargs['seeds'][0], [True, True, False], [1, 2, 3])
    monkeypatch.setattr('src.training.evaluation.evaluate_node', evaluate)
    kwargs = dict(env_id='Toy-v0', reset_hook=NO_HOOK, output=tmp_path / 'result.json')
    with pytest.raises(FileNotFoundError):
        evaluate_locked(manifest, lock, key, nodes[key], **kwargs)
    assert not calls
    lock_candidates(manifest, dirs, lock)
    result = evaluate_locked(manifest, lock, key, nodes[key], **kwargs)
    assert len(calls) == 1 and result['successes'] == 2
    kwargs['output'] = tmp_path / 'another-result.json'
    copied_lock = tmp_path / 'copied-lock.json'
    copied_lock.write_bytes(lock.read_bytes())
    with pytest.raises(FileExistsError):
        evaluate_locked(manifest, copied_lock, key, nodes[key], **kwargs)
    assert len(calls) == 1


def test_changed_manifest_cannot_open_final_test(tmp_path):
    manifest, lock = tmp_path / 'final.json', tmp_path / 'lock.json'
    reserve_seeds(manifest, [], count=3)
    dirs, nodes = fixture_runs(tmp_path)
    lock_candidates(manifest, dirs, lock)
    manifest.write_text(manifest.read_text() + '\n')
    key = next(iter(nodes))
    with pytest.raises(ValueError, match='manifest changed'):
        evaluate_locked(manifest, lock, key, nodes[key], env_id='Toy-v0', reset_hook=NO_HOOK,
                        output=tmp_path / 'result.json')


def test_lock_requires_same_training_configuration_across_seeds(tmp_path):
    manifest = tmp_path / 'final.json'
    reserve_seeds(manifest, [], count=3)
    dirs = []
    for seed in [42, 43, 44]:
        node = SkillNode('candidate', goal=NEVER, env_ids=['Toy-v0'])
        node.policy = torch.nn.Linear(2, 2)
        run = TrainingRun(node, {'seed': seed, 'total_timesteps': 512 if seed == 42 else 1024}, tmp_path / 'runs')
        run.finish('budget_exhausted', report={'seed': seed, 'env_ids': ['Toy-v0'], 'reset_hook': NO_HOOK.spec})
        dirs.append(run.path)
    with pytest.raises(ValueError, match='exactly training seeds'):
        lock_candidates(manifest, dirs, tmp_path / 'lock.json')


def test_changed_evaluator_cannot_open_final_test(tmp_path, monkeypatch):
    manifest, lock = tmp_path / 'final.json', tmp_path / 'lock.json'
    reserve_seeds(manifest, [], count=3)
    dirs, nodes = fixture_runs(tmp_path)
    lock_candidates(manifest, dirs, lock)
    monkeypatch.setattr('src.training.final_protocol.evaluation_code', lambda: {'changed': 'hash'})
    key = next(iter(nodes))
    with pytest.raises(ValueError, match='Evaluation code changed'):
        evaluate_locked(manifest, lock, key, nodes[key], env_id='Toy-v0', reset_hook=NO_HOOK,
                        output=tmp_path / 'result.json')
    assert not Path(str(manifest) + '.claims').exists()


def test_reservation_binds_to_one_unchanged_selection(tmp_path):
    manifest, lock = tmp_path / 'final.json', tmp_path / 'lock.json'
    reserve_seeds(manifest, [], count=3)
    dirs, nodes = fixture_runs(tmp_path)
    lock_candidates(manifest, dirs, lock)
    with pytest.raises(FileExistsError, match='already locked'):
        lock_candidates(manifest, dirs, tmp_path / 'replacement-lock.json')
    lock.write_text(lock.read_text() + '\n')
    key = next(iter(nodes))
    with pytest.raises(ValueError, match='selection lock changed'):
        evaluate_locked(manifest, lock, key, nodes[key], env_id='Toy-v0', reset_hook=NO_HOOK,
                        output=tmp_path / 'result.json')
