# Runtime parity: controlled checks and 24 development training runs

Completed 2026-11-12. This extends the earlier unlock/solve audit; it does not
replace it. [Preregistered plan and final protocol](../protocols/2026-11-12_final_and_parity_experiments.md).
[Machine-readable paired results](2026-11-12_parity_experiments.json).

## What was run

- Three controlled old/fixed checks: parent deadline, nested strict mask and a
  depth-3 per-frame decision budget. Fixed training and evaluation agree on both
  outcomes and primitive action traces. Old controls reproduce each defect.
- 24 training runs: 4 conditions × training seeds 42–44 × old/fixed runtimes.
  Runtime modules pinned to `917665e` and `584b4be`; CPU, 8 envs, 64 decisions/env,
  budget 5120 decisions, evaluation every 512, standard stability early stopping.
- Each matched pair has identical initial policy weights and frozen dependency
  hashes. All 24 archives match the cohort plan's training source fingerprints.
  Runtime overrides are saved separately because the source ZIP records on-disk
  code rather than modules loaded dynamically from Git.
- Development evaluation uses 50 seeds, 20000–20049. A post-plan diagnostic then
  evaluated all 24 saved models under the *same fixed runtime*, without retraining,
  to separate training changes from execution semantics. Final seeds were unused.

## Results

Success rates below use the common fixed evaluator. Numbers in each cell follow
training seeds 42, 43, 44. Three seeds support a sensitivity check, not a statistical
claim of improved learning.

| Condition | Old-trained SR | Fixed-trained SR | Final weights | First different PPO buffer update |
|---|---|---|---|---|
| Choice, goal-only | .18 / .44 / .78 | .50 / .24 / .68 | Different in all 3 | 1 / 1 / 1 |
| Choice, strict | .46 / .46 / .50 | .46 / .46 / .50 | Different in all 3 | 5 / 1 / 2 |
| Flat DoorKey options | 1 / 1 / 1 | 1 / 1 / 1 | Identical in all 3 | None |
| Synthetic depth 3 | .02 / 1 / 1 | 1 / 1 / 1 | Different in all 3 | 1 / 1 / 1 |

Choice uses all 10 updates in both runtimes. Baseline stops after 3 / 2 / 2
updates in both runtimes. Synthetic depth uses all 10 updates under the old
runtime and stops after 2 under the fixed one.

**Choice:** rewards, advantages and returns are the first PPO buffer fields to
change. The first transition discrepancy is the terminal observation when the
parent reaches its deadline; old execution continues beyond 240 env steps.
Time-limit bootstrap can therefore change the stored PPO reward even when the
wrapper's immediate reward is unchanged. Goal-only mean SR is .467 → .473, with
mixed per-seed directions: no demonstrated average benefit at this budget.
Strict SR is unchanged despite different learning inputs and final weights.
Total env-step savings across the six choice pairs: 30,449. All fixed choice
runs have zero observed parent-deadline overruns.

**Baseline:** buffers, weights, env-step counts and scores all remain equal.
The selected flat option set does not expose an effective learning difference in
these runs. This is consistent with the previous unlock/solve finding: fixes need
not change PPO data when the affected execution paths do not change its inputs.

**Depth:** own-runtime scores were 0 / 0 / 0 → 1 / 1 / 1. Under the common fixed
evaluator the old models instead score .02 / 1 / 1. Thus much of the apparent
0-to-1 gain is execution semantics, not learned policy quality. The changed
rollouts and weights still demonstrate that per-frame budgeting affects training.
This is a deliberately constructed step-count goal with scripted descendants,
not evidence of useful real-task deep composition.

The `parent_timeout_cuts` telemetry is a boundary heuristic (deadline reached and
no child outcome), not an exact causal count: baseline logs it in both runtimes.
Overrun counts and full transition traces are the reliable deadline evidence.

## Final test status

The two concurrent implementations were consolidated into
`src/training/final_protocol.py`. The unused 50,000-range proposal was removed.
`docs/protocols/final_test_v1.json` reserves 200 seed IDs after auditing 107
historical JSON files. No real candidate selection, opening claim or final result
has been created. The set remains untouched.

Reserve/lock and open are separate commands. Locking requires all training seeds
42–44, one common configuration per task/arm, complete artifact hashes and fixed
child checkpoints. It pins evaluator sources and the model archive. Evaluation
checks those identities and claims each candidate once beside the manifest;
changing output or copying the lock cannot rerun it. A reservation also binds to
one selection lock. Local guards cannot prevent deliberate deletion of audit files.
Different seeds can generate the same MiniGrid layout, so this is seed-level
separation rather than proof of unseen layouts.

## Validation and remaining scope

- Consolidated suite: 281 tests passed (duplicate protocol tests replaced).
- `eval_regression.py`: all 9 frozen nodes exactly reproduce recorded scores.
- Common-evaluator replay successfully loads all 24 archived models and verifies
  their policy hashes and node specifications; it does not alter checkpoints.
- `graphify update .` refreshed the local code graph.

Still outside this pilot: original 100-update choice convergence experiment,
other baseline option sets, natural depth-3+ task learning, and a layout-disjoint
benchmark. The group's decision on per-frame versus shared caps remains a design
decision; this study measures the implemented per-frame semantics. Final model
selection and opening must wait until the development experiment choices are
closed. No final score should be used to choose the next training configuration.

## Artifacts and reproduction

Raw local artifacts: `output/parity_2026_11_12/` (24 isolated training archives,
logs, per-update hashes, transitions, controls, cohort plan/status and common
replay). Compact results above are retained in the adjacent JSON with raw-result
SHA-256 digests. Raw archives are local and have not been uploaded.

```sh
# Use a NEW output directory after code changes; existing cohort plans reject drift.
python scripts/run_parity_experiment.py --variant old --controls-only --out-dir NEW_OUTPUT
python scripts/run_parity_experiment.py --variant new --controls-only --out-dir NEW_OUTPUT
python scripts/run_parity_suite.py --out-dir NEW_OUTPUT
python scripts/eval_parity_checkpoints.py --out-dir NEW_OUTPUT
python scripts/summarize_parity_suite.py --out-dir NEW_OUTPUT
```
