"""One isolated old/fixed runtime control or paired development training run."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'scripts')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['old', 'new'], required=True)
    parser.add_argument('--task', choices=['choice', 'baseline', 'depth'], default='choice')
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--budget', type=int, default=5120)
    parser.add_argument('--controls-only', action='store_true')
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = f'{args.task}_{"strict" if args.strict else "goal"}_{args.seed}_{args.variant}'
    if args.controls_only:
        name = f'controls_{args.variant}'
    result_path = args.out_dir / f'{name}.json'
    if result_path.exists():
        raise FileExistsError(result_path)

    # Import the two runtime modules before downstream modules bind their classes.
    import src.skills.executor as ex_module
    import src.envs.option_env as option_module
    ref = '917665e' if args.variant == 'old' else '584b4be'
    sources = {}
    for module, path in [(ex_module, 'src/skills/executor.py'), (option_module, 'src/envs/option_env.py')]:
        source = subprocess.check_output(['git', 'show', f'{ref}:{path}'], cwd=REPO, text=True)
        sources[path] = source
        exec(compile(source, f'{ref}:{path}', 'exec'), module.__dict__)

    import numpy as np
    from sb3_contrib import MaskablePPO
    from src.envs.subgoal_env import NO_HOOK, two_keys
    from src.skills.registry import NodeRegistry
    from src.training.evaluation import run_episode, evaluate_node, heldout_seeds
    from src.training.node_trainer import train_composite, execution_env_factory, set_torch_threads
    from src.training.run_store import policy_hash, write_json
    import parity_tasks as tasks
    set_torch_threads(2)

    source_hashes = {p: hashlib.sha256(s.encode()).hexdigest() for p, s in sources.items()}
    provenance = {'variant': args.variant, 'runtime_ref': subprocess.check_output(
        ['git', 'rev-parse', ref], cwd=REPO, text=True).strip(), 'runtime_source_sha256': source_hashes}

    if args.controls_only:
        results = {}
        for label, build, strict in [('timeout', tasks.timeout_task, False),
                                     ('strict', tasks.strict_task, True),
                                     ('depth', tasks.depth_task, False)]:
            train_log, eval_log = [], []
            node = build(train_log)
            env = option_module.make_option_env(node, tasks.UNLOCK, seed=7, initiation_masking=strict)
            env.reset(seed=7)
            try:
                for _ in range(100):
                    _, reward, term, trunc, info = env.step(env.action_masks().index(True))
                    if term or trunc:
                        break
                else:
                    raise AssertionError('Control exceeded decision bound')
            finally:
                env.close()
            evaluation_env = execution_env_factory(tasks.UNLOCK)()
            try:
                ok, steps, _ = run_episode(build(eval_log), evaluation_env, 7, initiation_masking=strict)
            finally:
                evaluation_env.close()
            results[label] = {'train': [bool(term), info['option_steps']], 'eval': [ok, steps],
                              'train_actions': train_log, 'eval_actions': eval_log, 'reward': reward}
        expected = {'timeout': [False, 12], 'strict': [True, 2], 'depth': [True, 5]}
        if args.variant == 'new':
            for label, result in results.items():
                assert result['train'] == result['eval'] == expected[label], (label, result)
                assert result['train_actions'] == result['eval_actions'], label
        else:
            assert results['timeout']['train'] == [True, 13]
            assert results['timeout']['eval'] == [False, 12]
            assert results['strict']['train_actions'] != results['strict']['eval_actions']
            assert results['depth']['train'] == [False, 2]
        write_json(result_path, {**provenance, 'controls': results})
        print(result_path, 'PASS', flush=True)
        return

    registry = NodeRegistry(REPO / 'runs/v2')
    hook = NO_HOOK
    if args.task == 'choice':
        import train_choice_task as choice
        node = choice.build_t11(choice.load_choice_children(registry, device='cpu', third='open_door_tk'))
        hook = two_keys(choice.COLORS)
        env_id = choice.DOORKEY
    elif args.task == 'baseline':
        import train_flat_baseline as baseline
        node = baseline.build_flat_options(baseline.load_option_children('doorkey', registry))
        env_id = baseline.DOORKEY
    else:
        node = tasks.depth_task()
        env_id = tasks.UNLOCK

    def array_hash(x):
        x = np.asarray(x)
        return hashlib.sha256(str((x.shape, str(x.dtype))).encode() + x.tobytes()).hexdigest()

    transitions, updates = [], []
    events = Counter()
    active = None
    original_mask = ex_module.Executor._child_mask
    original_close = ex_module.Executor._close_from
    def mask(self, child, sv):
        value = original_mask(self, child, sv)
        if active is not None:
            active['nested_masks'].append([child.id, bool(value), bool(child.initiation(sv))])
        return value
    def close(self, k, outcome, sv):
        if active is not None:
            frame = self.stack[k]
            count = getattr(frame, 'decisions', self.decisions)
            cap = self.max_decisions if k == 0 else frame.node.max_decisions
            if outcome.value == 'timeout' and not frame.node.is_primitive and count >= cap:
                events['decision_cap_closures'] += 1
        return original_close(self, k, outcome, sv)
    ex_module.Executor._child_mask = mask
    ex_module.Executor._close_from = close
    original_step = option_module.OptionEnv.step
    def step(self, idx):
        nonlocal active
        active = {'before': array_hash(self.obs), 'action': int(idx),
                  'mask': list(map(bool, self.action_masks())), 'nested_masks': []}
        try:
            obs, reward, term, trunc, info = original_step(self, idx)
            active.update(after=array_hash(obs), reward=float(reward), term=bool(term), trunc=bool(trunc),
                          k=info['k'], option_steps=info['option_steps'])
            transitions.append(active)
            if info['option_steps'] > self.node.timeout:
                events['parent_timeout_overruns'] += 1
            if info['option_steps'] >= self.node.timeout and info['child_outcome'] is None:
                events['parent_timeout_cuts'] += 1
            if info['k'] == 0:
                events['zero_step_decisions'] += 1
            return obs, reward, term, trunc, info
        finally:
            active = None
    option_module.OptionEnv.step = step
    original_train = MaskablePPO.train
    def train(self):
        fields = ['observations', 'actions', 'rewards', 'episode_starts', 'values',
                  'log_probs', 'advantages', 'returns', 'action_masks']
        row = {'at_decisions': self.num_timesteps, 'before_weights': policy_hash(self.policy),
               'buffer': {f: array_hash(getattr(self.rollout_buffer, f)) for f in fields}}
        original_train(self)
        row['after_weights'] = policy_hash(self.policy)
        updates.append(row)
    MaskablePPO.train = train

    report, model = train_composite(node, [env_id], hook, total_timesteps=args.budget,
        eval_every=512, n_eval=50, n_envs=8, seed=args.seed, device='cpu',
        initiation_masking=args.strict, run_root=args.out_dir / 'runs', verbose=0)
    # Keep explicit runtime overrides alongside each archive: source.zip contains the
    # current on-disk code, while these two modules were loaded from the pinned refs.
    run_path = Path(report['run_dir'])
    runtime_dir = args.out_dir / f'{name}_runtime'
    runtime_dir.mkdir()
    for path, source in sources.items():
        (runtime_dir / Path(path).name).write_text(source)
    development = evaluate_node(node, execution_env_factory(env_id, hook),
                                seeds=heldout_seeds(50), initiation_masking=args.strict)
    result = {**provenance, 'task': args.task, 'strict': args.strict, 'seed': args.seed,
              'budget_decisions': args.budget, 'report': report, 'events': dict(events),
              'weights': policy_hash(model.policy), 'actual_train_calls': len(updates),
              'updates': updates, 'transitions': transitions,
              'development': {**development.as_dict(), 'per_episode': development.per_episode,
                              'lengths': development.lengths}, 'final_test_used': False}
    if args.task == 'choice':
        result['first_key'] = choice.first_decision_accuracy(node, hook, heldout_seeds(50),
                                                           initiation_masking=args.strict)
    write_json(result_path, result)
    print(name, 'steps', report['env_steps_own'], 'train_calls', len(updates),
          'dev_sr', development.success_rate, flush=True)


if __name__ == '__main__':
    main()
