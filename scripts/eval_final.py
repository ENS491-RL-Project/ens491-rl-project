"""Open a locked composite candidate once; never select candidates using final scores.

Reserve and lock first with scripts/final_test_protocol.py. Supported task builders:
unlock_door, solve_doorkey, unlock_matching and flat_doorkey_options.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'scripts')]


def build_candidate(node_id, registry):
    from src.envs.subgoal_env import NO_HOOK, two_keys
    node_id = node_id.rsplit(' ', 1)[-1]
    if node_id == 'unlock_matching':
        import train_choice_task as task
        return task.build_t11(task.load_choice_children(registry, third='open_door_tk')), task.DOORKEY, two_keys(task.COLORS)
    if node_id == 'flat_doorkey_options':
        import train_flat_baseline as task
        return task.build_flat_options(task.load_option_children('doorkey', registry)), task.DOORKEY, NO_HOOK
    if node_id == 'unlock_door':
        import train_unlock_door as task
        return task.build_t3(task.load_children(registry)), task.UNLOCK, NO_HOOK
    if node_id == 'solve_doorkey':
        import train_solve_doorkey as task
        return task.build_t5(*task.load_children(registry)), task.DOORKEY, NO_HOOK
    raise ValueError(f'No final-evaluation builder for {node_id}; add and validate it before locking candidates')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--open', action='store_true', required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--candidate', required=True, help='run_id from the candidate lock')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--registry', type=Path, default=REPO / 'runs/v2')
    args = parser.parse_args()
    from sb3_contrib import MaskablePPO
    from src.skills.registry import NodeRegistry
    from src.training.final_protocol import evaluate_locked, sha256
    from src.training.node_trainer import CompositePolicyAdapter, set_torch_threads
    set_torch_threads(2)
    candidate = json.loads(args.lock.read_text())['candidates'][args.candidate]
    model_path = Path(candidate['run_dir']) / 'model.zip'
    if sha256(model_path) != candidate['model_sha256']:
        raise ValueError('Archived model changed after selection')
    node, env_id, hook = build_candidate(candidate['node_spec']['id'], NodeRegistry(args.registry))
    model = MaskablePPO.load(model_path, device='cpu')
    node.policy = CompositePolicyAdapter(model.policy)
    result = evaluate_locked(args.manifest, args.lock, args.candidate, node,
                             env_id=env_id, reset_hook=hook, output=args.output)
    print(f"{args.candidate}: success_rate={result['success_rate']:.3f}; saved {args.output}")


if __name__ == '__main__':
    main()
