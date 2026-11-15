# Training parity audit — 2026-11-05

## Result

18 training runs were re-run: unlock_door goal-only and solve_doorkey goal-only/strict, each on seeds 42–44, paired old/new. In every pair the SHA-256 values of the final weights, of the input buffers of every actual PPO update, and of the weights before and after each update are equal. This result holds only for these checkpoints, settings and seeds.

| Task/arm | Seed | Old env steps | New env steps | Decisions with changed duration | Actual PPO train calls | Reported updates |
|---|---:|---:|---:|---:|---:|---:|
| unlock_door goal | 42 | 34,877 | 34,733 | 3 | 2 | 3 |
| unlock_door goal | 43 | 32,869 | 32,629 | 5 | 2 | 3 |
| unlock_door goal | 44 | 33,369 | 33,129 | 5 | 2 | 3 |
| solve_doorkey goal | 42 | 55,659 | 55,659 | 0 | 2 | 3 |
| solve_doorkey goal | 43 | 48,887 | 48,887 | 0 | 2 | 3 |
| solve_doorkey goal | 44 | 53,930 | 53,930 | 0 | 2 | 3 |
| solve_doorkey strict | 42 | 7,280 | 7,280 | 0 | 1 | 2 |
| solve_doorkey strict | 43 | 7,253 | 7,253 | 0 | 1 | 2 |
| solve_doorkey strict | 44 | 7,240 | 7,240 | 0 | 1 | 2 |

### unlock_door: why identical?

Under the old code, 3/5/5 decisions run past the parent's 240-step timeout and reach 288. The new code cuts each of them 48 steps earlier. In all of these tails action=2 (forward) and the observation hash is the same at every step. The start/end observations, action, mask, reward and terminated/truncated flags handed to PPO are the same in every decision. Only k and option_steps change.

The current MaskablePPO uses a fixed per-decision gamma=0.99; the k in info does not enter the learning buffer. The terminal observation used for the timeout bootstrap is also the same. So identical actual training inputs explain why the weights stay equal. We verified this not only on the final checkpoint but by comparing the hashes of the observations/actions/rewards/episode_starts/values/log_probs/advantages/returns/action_masks buffers before every train call.

### solve_doorkey: why identical?

In the goal-only arm the recorded transitions, durations and nested mask calls are fully identical. In the strict arm the nested mask records of 512 top-level decisions change on every seed; the top-level policy's transitions and all PPO buffers do not. This does not mean passing the mask down has no effect or does not work: the mask changes, but with these child checkpoints and the states reached it does not show up in the top-level policy's training data. The probe does not record every inner selection and primitive action; the measurement is at the level of top-level transitions and mask calls.

### Additional finding: the update count is over-reported

src/training/node_trainer.py computes policy_updates as decisions // (n_steps * n_envs). But EvaluationCallback._on_step() returns False once stability is reached. On that signal the installed sb3_contrib collect_rollouts finishes the rollout and returns without calling train(). The last collected rollout is never applied to the weights.

The measured actual train() calls are 2 for unlock/solve goal (report: 3) and 1 for solve strict (report: 2). Each train call here contains 4 optimizer epochs; _n_updates and train calls are different units. The counter should keep completed_rollouts, train_calls, optimizer_epochs and collected_decisions separately. This reporting fix does not change any past weights.

## Method and re-running

- New code: 584b4be (full SHA in provenance.json).
- Old code: the src/skills/executor.py and src/envs/option_env.py modules at commit 917665e, loaded in a separate process; the trainer, child checkpoints and all other settings are kept the same.
- CPU, torch threads=2, 8 envs, n_steps=64, batch_size=64, eval_every=512, n_eval=50, budget 51,200 decisions; early stopping with the current stability criterion.
- The probe saves and freezes no checkpoint. The registry is only read. The stopping seeds are used; no new final test set is opened.
- Provenance: package versions and checkpoint/metadata SHA-256 values are in provenance.json. The per-run JSON records contain the transitions, buffer hashes, actual updates and the report.

From the repo root, with the Python of the ens491 environment (output goes to `runs/training_parity_audit/`, which is gitignored; the summary is committed as `2026-11-05_training_parity_summary.json`):

```sh
python scripts/audit_training_parity.py --variant old --task unlock --seed 42
python scripts/audit_training_parity.py --variant new --task unlock --seed 42
python scripts/audit_training_parity.py --variant old --task solve --strict --seed 42
python scripts/audit_training_parity.py --variant new --task solve --strict --seed 42
```

Also run for seeds 43/44 and for solve goal-only. Once all files are present:

```sh
python scripts/compare_training_parity.py
```

Test suite at the time: 267 passed, including the new MAX_ITERS regression tests.


## Integration

The runtime parity changes are in PR #15. Checkpoint loading and the update counter are a separate change based on main. Old reports without `policy_updates_semantics` retain their original numbers: do not subtract one indiscriminately. The overcount occurs when early stopping discards a rollout on its boundary. New reports declare `completed_train_calls`.
