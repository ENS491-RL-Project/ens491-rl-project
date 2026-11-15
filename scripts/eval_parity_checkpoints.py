"""Evaluate all paired archives under one fixed runtime on development seeds only."""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'scripts')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=Path('output/parity_2026_11_12'))
    args = parser.parse_args()
    output = args.out_dir / 'common_evaluator.json'
    if output.exists():
        raise FileExistsError(output)
    from sb3_contrib import MaskablePPO
    from src.envs.subgoal_env import NO_HOOK
    from src.skills.registry import NodeRegistry
    from src.training.evaluation import evaluate_node, heldout_seeds
    from src.training.node_trainer import execution_env_factory, CompositePolicyAdapter, set_torch_threads
    from src.training.run_store import policy_hash, write_json
    from src.training.final_protocol import sha256, evaluation_code
    from eval_final import build_candidate
    from parity_tasks import depth_task, UNLOCK
    set_torch_threads(2)
    expected = json.loads((args.out_dir / 'controls_new.json').read_text())['runtime_source_sha256']
    assert all(sha256(REPO / path) == digest for path, digest in expected.items()), 'Fixed runtime changed'
    rows = []
    for path in sorted(args.out_dir.glob('*.json')):
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or 'report' not in data:
            continue
        run_path = Path(data['report']['run_dir'])
        run = json.loads((run_path / 'run.json').read_text())
        complete = json.loads((run_path / 'complete.json').read_text())
        assert sha256(run_path / 'model.zip') == complete['sha256']['model.zip']
        if data['task'] == 'depth':
            node, env_id, hook = depth_task(), UNLOCK, NO_HOOK
        else:
            node, env_id, hook = build_candidate(run['node_spec']['id'], NodeRegistry(REPO / 'runs/v2'))
        assert node.spec() == run['node_spec']
        model = MaskablePPO.load(run_path / 'model.zip', device='cpu')
        node.policy = CompositePolicyAdapter(model.policy)
        assert policy_hash(node.policy) == data['weights']
        result = evaluate_node(node, execution_env_factory(env_id, hook), seeds=heldout_seeds(50),
                               initiation_masking=data['strict'])
        rows.append({'result_file': path.name, 'result_sha256': sha256(path),
                     **result.as_dict(), 'per_episode': result.per_episode, 'lengths': result.lengths})
        print(path.stem, result.success_rate, flush=True)
    assert len(rows) == 24
    write_json(output, {'final_test_used': False, 'runtime_source_sha256': expected,
                        'evaluation_source_sha256': evaluation_code(), 'rows': rows})


if __name__ == '__main__':
    main()
