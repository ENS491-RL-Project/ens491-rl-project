# W4(a) — Does a flat PPO with memory solve what the memoryless one could not?

> Written 2026-11-15 by Mehmet P. with Claude Code, against `fix/audit-followups-main` at `d031676`.
> Budgets, seeds, architecture and readings below are fixed before the first real run (decision 0010).
> Smoke runs and the equivalence test are not runs of this plan.
>
> Informs: `open-questions.md` "Does the flat baseline need memory?" (owner Emir); decision 0006,
> Evidence "missing (4)"; roadmap W4(a), claim part B3. **Changes no shared training file**
> (`executor.py`, `option_env.py`, `evaluation.py`, `node_trainer.py`). Nothing is frozen as a
> curriculum node or registered in `runs/v2/` outside `baselines/`.

## Question

Flat PPO scored 0/50 on UnlockPickup after 3,000,320 env steps on each of three seeds (censored),
and is STABLE on DoorKey-6x6 in 300–450k. The policy is memoryless and the view is 7×7. Is the
UnlockPickup result a memory limit?

## Arm

`{0,B} flat_<task>_rppo`: `sb3_contrib.RecurrentPPO("MlpLstmPolicy")`, `net_arch=[64, 64]`,
`lstm_hidden_size=64`, every other hyperparameter `PPO_DEFAULTS` unchanged, 8 envs.
Same goal predicate, timeout, env, observation pipeline and reward as the memoryless
`{0,B} flat_<task>_ppo` (`train_flat_baseline.TASKS`). Stopping rule: decision 0004 unchanged, on the
same 50 stopping seeds; the quoted score is on the same 50 held-out seeds.

Evaluation is not driven by the `Executor`, which carries no hidden state: a loop on `SubgoalEnv`
keeps the LSTM state for one episode. `tests/test_flat_recurrent.py` shows that loop and
`evaluate_node` agree episode by episode for a memoryless policy.

## Runs

| Task | Budget (env steps) | `eval_every` | Seeds |
|---|---|---|---|
| UnlockPickup | 3,000,000 | 100,000 | 42, 43, 44 |
| DoorKey-6x6 | 1,500,000 | 50,000 | 42, 43, 44 |

Budgets equal the memoryless rows and are not extended. A run stops at STABLE. The final weights
are saved whether or not the run is STABLE (`runs/v2/baselines/<node>_s<seed>/`).

## Readings, fixed in advance (UnlockPickup)

| Outcome | Reading |
|---|---|
| STABLE on 2 or 3 seeds | The 0/50 was at least partly a memory limit. B3 is restated; the cost is set beside the hierarchy's 2.38M recipe |
| No seed STABLE and best stopping success rate < 0.10 on all three | Memory alone does not explain it, for this architecture and budget |
| Anything else | Reported as measured, no verdict |

DoorKey: steps to STABLE beside the memoryless 300–450k. Not a gate.

## Limits

One architecture; PPO hyperparameters not tuned for the recurrent policy; three seeds; argmax
evaluation. A negative result does not show that no recurrent learner can solve the task.

## Not in this plan

Frame stacking; Unlock and BlockedUnlockPickup; W4(b) (one flat PPO fine-tuned through the
curriculum); memory inside hierarchy nodes.
