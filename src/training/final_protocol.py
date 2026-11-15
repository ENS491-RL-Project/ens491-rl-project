"""Reserve final seeds, lock candidates, then permit a single recorded evaluation.

Seed disjointness is checked against available records, not a guarantee of distinct
MiniGrid layouts. Development code must never use the reserved seeds.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets

from src.training.run_store import policy_hash, write_json

REPO = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _exclusive_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write('\n')


def _integers(value):
    if type(value) is int:
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _integers(v)
    elif isinstance(value, list):
        for v in value:
            yield from _integers(v)


def reserve_seeds(output, history_roots, count=200):
    """Conservatively exclude every integer found in historical JSON records."""
    if count < 1:
        raise ValueError('count must be positive')
    if Path(output).exists():
        raise FileExistsError(output)
    used = set(range(0, 100_000))  # training/stopping/legacy development ranges
    inventory = {}
    for root in history_roots:
        for path in sorted(Path(root).rglob('*.json')):
            try:
                record = json.loads(path.read_text())
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f'Cannot audit historical record {path}') from error
            used.update(_integers(record))
            label = str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)
            inventory[label] = sha256(path)
    seeds = set()
    while len(seeds) < count:
        seed = secrets.randbelow(2**30 - 2**28) + 2**28
        if seed not in used:
            seeds.add(seed)
    manifest = {
        'schema_version': 1, 'purpose': 'final_only', 'created_at': datetime.now(timezone.utc).isoformat(),
        'seeds': sorted(seeds), 'training_seeds': [42, 43, 44],
        'historical_records': inventory, 'excluded_integer_count': len(used),
        'selection_rule': 'One prespecified configuration per task/arm; last completed checkpoint per training seed. No best-seed selection.',
        'execution_mode': 'deterministic',
        'metrics': ['success_rate', 'wilson_lower', 'wilson_upper', 'mean_length', 'per_episode', 'lengths'],
        'limitations': 'No recorded seed overlap; different seeds may generate the same MiniGrid layout. Unrecorded historical use cannot be excluded.',
    }
    _exclusive_json(output, manifest)
    return manifest


def _read_manifest(path):
    data = json.loads(Path(path).read_text())
    seeds = data['seeds']
    if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError('Final seeds must be nonempty, unique uint32 integers')
    if data.get('purpose') != 'final_only':
        raise ValueError('Not a final-only manifest')
    return data


def evaluation_code():
    files = [*(REPO / 'src').rglob('*.py'), *(REPO / 'scripts').rglob('*.py'), REPO / 'requirements.txt']
    return {str(p.relative_to(REPO)): sha256(p) for p in sorted(files)}


def lock_candidates(manifest_path, run_dirs, output):
    """Lock all training seeds per configuration, without running any final episodes."""
    manifest = _read_manifest(manifest_path)
    candidates, groups, configurations = {}, {}, {}
    for run_dir in run_dirs:
        p = Path(run_dir).resolve()
        complete = json.loads((p / 'complete.json').read_text())
        if complete['state'] not in ('stable', 'budget_exhausted'):
            raise ValueError(f'Cannot select failed/incomplete run {p}')
        required_artifacts = {'run.json', 'report.json', 'policy_state_dict.pt', 'source.zip'}
        if not required_artifacts <= complete['sha256'].keys():
            raise ValueError(f'Incomplete artifact inventory: {p}')
        for name, expected in complete['sha256'].items():
            if Path(name).name != name or sha256(p / name) != expected:
                raise ValueError(f'Changed run artifact: {p / name}')
        run = json.loads((p / 'run.json').read_text())
        report = json.loads((p / 'report.json').read_text())
        seed = report['seed']
        strict = report.get('initiation_masking', False)
        config = {k: v for k, v in run['config'].items() if k != 'seed'}
        group = json.dumps([run['node_spec'], config, run['children'], strict,
                            report['env_ids'], report['reset_hook'], report.get('normalize_obs', True)], sort_keys=True)
        arm = (run['node_spec']['id'], strict)
        if arm in configurations and configurations[arm] != group:
            raise ValueError('Each task/arm must contain exactly training seeds with one common configuration')
        configurations[arm] = group
        groups.setdefault(group, []).append(seed)
        candidate_id = run['run_id']
        if candidate_id in candidates:
            raise ValueError('Duplicate candidate')
        candidates[candidate_id] = {
            'run_dir': str(p), 'seed': seed, 'node_spec': run['node_spec'],
            'training_config': run['config'],
            'model_sha256': sha256(p / 'model.zip') if (p / 'model.zip').exists() else None,
            'policy_sha256': run['policy_sha256'], 'children': run['children'],
            'checkpoint_sha256': sha256(p / 'policy_state_dict.pt'),
            'env_ids': report['env_ids'], 'reset_hook': report['reset_hook'],
            'initiation_masking': strict, 'normalize_obs': report.get('normalize_obs', True),
        }
    required = sorted(manifest['training_seeds'])
    if not candidates or any(sorted(seeds) != required for seeds in groups.values()):
        raise ValueError(f'Each task/arm must contain exactly training seeds {required}')
    lock = {'schema_version': 1, 'manifest_sha256': sha256(manifest_path),
            'evaluation_code_sha256': evaluation_code(), 'candidates': candidates,
            'selection_rule': manifest['selection_rule'], 'execution_mode': manifest['execution_mode']}
    selection_path = Path(str(manifest_path) + '.selection.json')
    if selection_path.exists():
        raise FileExistsError('Candidate selection is already locked for this manifest')
    _exclusive_json(output, lock)
    _exclusive_json(selection_path, {'lock_sha256': sha256(output)})
    return lock


def evaluate_locked(manifest_path, lock_path, candidate_id, node, *, env_id, reset_hook, output):
    """Evaluate only a hash-matched, locked candidate. Never called by training."""
    from src.training.evaluation import evaluate_node
    from src.training.node_trainer import execution_env_factory

    manifest = _read_manifest(manifest_path)
    lock = json.loads(Path(lock_path).read_text())
    selection = json.loads(Path(str(manifest_path) + '.selection.json').read_text())
    if sha256(lock_path) != selection['lock_sha256']:
        raise ValueError('Candidate selection lock changed')
    if sha256(manifest_path) != lock['manifest_sha256']:
        raise ValueError('Final seed manifest changed after selection')
    if evaluation_code() != lock['evaluation_code_sha256']:
        raise ValueError('Evaluation code changed after selection')
    candidate = lock['candidates'][candidate_id]
    if node.spec() != candidate['node_spec'] or policy_hash(node.policy) != candidate['policy_sha256']:
        raise ValueError('Node specification or weights differ from locked candidate')
    checkpoint = Path(candidate['run_dir']) / 'policy_state_dict.pt'
    if sha256(checkpoint) != candidate['checkpoint_sha256']:
        raise ValueError('Checkpoint changed after selection')
    descendants, pending = {}, list(node.children) + list(node.lateral_sources)
    while pending:
        child = pending.pop()
        if child.id in descendants:
            continue
        descendants[child.id] = child
        pending.extend((*child.children, *child.lateral_sources))
    for expected in candidate['children']:
        child = descendants.get(expected['id'])
        if child is None or child.spec_hash() != expected['spec_hash'] or policy_hash(child.policy) != expected['policy_sha256']:
            raise ValueError('Child checkpoint differs from locked training dependency')
    if candidate['env_ids'] != [env_id] or reset_hook.spec != candidate['reset_hook']:
        raise ValueError('Environment or reset hook differs from locked configuration')
    # Claims belong to the seed reservation: copying a lock or changing output cannot rerun it.
    claims = Path(str(manifest_path) + '.claims')
    key = hashlib.sha256(candidate_id.encode()).hexdigest()
    if Path(output).exists():
        raise FileExistsError(output)
    _exclusive_json(claims / f'{key}.json', {'candidate_id': candidate_id, 'state': 'started'})
    result = evaluate_node(node, execution_env_factory(env_id, reset_hook, normalize=candidate['normalize_obs']),
                           seeds=manifest['seeds'], execution_mode=lock['execution_mode'],
                           initiation_masking=candidate['initiation_masking'])
    data = {**result.as_dict(), 'candidate_id': candidate_id,
            'manifest_sha256': lock['manifest_sha256'], 'lock_sha256': sha256(lock_path),
            'seeds': manifest['seeds'], 'per_episode': result.per_episode, 'lengths': result.lengths}
    _exclusive_json(output, data)
    write_json(claims / f'{key}.json', {'candidate_id': candidate_id, 'state': 'completed',
                                      'result_sha256': sha256(output)})
    return data
