"""Reproduce old/new training, auditing actual PPO inputs without saving checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--variant', choices=['old', 'new'], required=True)
p.add_argument('--task', choices=['unlock', 'solve'], default='unlock')
p.add_argument('--strict', action='store_true')
p.add_argument('--seed', type=int, default=42)
p.add_argument('--out-dir', type=Path, default=Path('output/training_parity_audit'))
a = p.parse_args()
a.out_dir.mkdir(parents=True, exist_ok=True)
repo = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(repo), str(repo / 'scripts')]
import numpy as np
import src.skills.executor as executor_module
import src.envs.option_env as option_module
runtime_ref = '917665e' if a.variant == 'old' else '584b4be'
if runtime_ref:
    for module, path in [(executor_module, 'src/skills/executor.py'),
                         (option_module, 'src/envs/option_env.py')]:
        source = subprocess.check_output(['git', 'show', f'{runtime_ref}:{path}'], cwd=repo, text=True)
        exec(compile(source, f'{runtime_ref}:{path}', 'exec'), module.__dict__)
from sb3_contrib import MaskablePPO
from src.envs.subgoal_env import NO_HOOK
from src.skills.registry import NodeRegistry
from src.training.node_trainer import set_torch_threads, train_composite
import train_unlock_door as unlock
import train_solve_doorkey as solve


def digest_array(x):
    x = np.asarray(x)
    return hashlib.sha256(str((x.shape, str(x.dtype))).encode() + x.tobytes()).hexdigest()


def weights(model):
    h = hashlib.sha256()
    for k, v in sorted(model.policy.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

records, updates = [], []
active = None
orig_mask = executor_module.Executor._child_mask

def child_mask(self, child, sv):
    result = orig_mask(self, child, sv)
    if active is not None:
        active['masks'].append([child.id, bool(result), bool(child.initiation(sv))])
    return result

executor_module.Executor._child_mask = child_mask
orig_step = option_module.OptionEnv.step

def step(self, idx):
    global active
    before = digest_array(self.obs)
    mask = list(map(bool, self.action_masks()))
    active = {'masks': []}
    tail = []
    env_step = self.env.step
    steps = self.steps
    def wrapped_env_step(action):
        nonlocal steps
        result = env_step(action)
        steps += 1
        if steps >= self.node.timeout:
            tail.append([steps, int(action), digest_array(result[0])])
        return result
    self.env.step = wrapped_env_step
    try:
        obs, reward, term, trunc, info = orig_step(self, idx)
        records.append({'before': before, 'mask': mask, 'action': int(idx),
                        'after': digest_array(obs), 'reward': float(reward),
                        'term': bool(term), 'trunc': bool(trunc),
                        'k': info['k'], 'option_steps': info['option_steps'],
                        'nested_masks': active['masks'], 'timeout_tail': tail})
        return obs, reward, term, trunc, info
    finally:
        self.env.step = env_step
        active = None

option_module.OptionEnv.step = step
orig_train = MaskablePPO.train

def train(self):
    b = self.rollout_buffer
    row = {'at_decisions': self.num_timesteps, 'before_weights': weights(self),
           'buffer': {name: digest_array(getattr(b, name)) for name in
                      ['observations', 'actions', 'rewards', 'episode_starts',
                       'values', 'log_probs', 'advantages', 'returns', 'action_masks']}}
    orig_train(self)
    row['after_weights'] = weights(self)
    updates.append(row)

MaskablePPO.train = train
set_torch_threads(2)
registry = NodeRegistry(repo / 'runs' / 'v2')
if a.task == 'unlock':
    node = unlock.build_t3(unlock.load_children(registry, device='cpu'))
    env_id = unlock.UNLOCK
else:
    node = solve.build_t5(*solve.load_children(registry, device='cpu'))
    env_id = solve.DOORKEY
report, model = train_composite(node, [env_id], NO_HOOK, total_timesteps=51200,
    eval_every=512, n_eval=50, n_envs=8, seed=a.seed, device='cpu',
    initiation_masking=a.strict, verbose=0)
result = {'args': {**vars(a), 'out_dir': str(a.out_dir)}, 'current_commit': subprocess.check_output(
    ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
    'runtime_ref': runtime_ref, 'old_source': '917665e', 'weights': weights(model), 'report': report,
    'actual_train_calls': len(updates), 'optimizer_epochs': model._n_updates,
    'updates': updates, 'transitions': records}
name = f'{a.task}_{"strict" if a.strict else "goal"}_{a.seed}_{a.variant}.json'
(a.out_dir / name).write_text(json.dumps(result))
print(name, 'env_steps', report['env_steps_own'], 'train_calls', len(updates), flush=True)
