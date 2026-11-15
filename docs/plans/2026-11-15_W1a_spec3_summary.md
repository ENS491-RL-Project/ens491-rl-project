# W1a, specification 3 summary: `pickup_box` trained half at the real handoff, W1a's test unchanged

> Run 2026-11-15 on `main` at `73bffca`, CPU (Windows), following
> [`2026-11-15_W1a_spec3_plan.md`](2026-11-15_W1a_spec3_plan.md) (revision 2). Decision
> [0011](../kb/decisions/0011-pickup-box-spec-3.md) is `Proposed`; this experiment ran while it is
> `Proposed`, with Emir's go-ahead, under `scripts/` only. No file under `src/` changed, v2 was not
> edited, and no spec-3 model was frozen or registered.

## Verdict: spec 3 passes (7/10)

Seeds 43–47 gave exactly 3 of 5, so the extension declared in the plan ran (seeds 48–52, trained
and scored the same way). Over the ten seeds 43–52, **7 pass and 3 fail; the rule needs at least 7.
Spec 3 passes**, at the threshold.

- Passing seeds: 43, 45, 46, 48, 49, 51, 52. Failing seeds: 44, 47, 50.
- The **handoff** row passes on all ten seeds (sampled 0.984–1.000).
- Every failing seed fails on the **own-hook** row (0.898, 0.778, 0.896).
- The handoff chain left the handoff state in 488 of 500 episodes on the new range (≥ 475).
- No training run met decision 0004's (argmax) rule; all ran the full budget.

`pickup_box` is not called solved or working here. Per the plan, Emir may now set 0011 to
`Accepted`; that is his decision and is not taken here. W1b is not started.

## What was run

| Step | What | Result |
|---|---|---|
| 0 | Decision file 0011, `Proposed`, naming the v2 §11 clause it sets aside | Written; Emir read it and said to continue |
| 1 | `scripts/pickup_box_spec3.py`: spec-3 node and the handoff hook; gate on env seeds 51000–51199 | Gate passed: the hook's start states equal `eval_w1a.py`'s chain in all 197 valid seeds (grid, pose, carrying, and the 147-dim observation through `SubgoalEnv`); the 3 invalid seeds are rejected by the hook and invalid in evaluation; 0 mismatches under the mixture |
| 2 | Training, seeds 43–47, `--no-freeze` | All `budget_exhausted` at 1,001,472 steps |
| 3 | `scripts/eval_w1a_spec3.py`: gate, then W1a's test on fresh ranges (own hook 52000–52499, handoff 53000–53499), then descriptive rows on W1a's ranges | Gate passed (10 spec-2 models reproduce W1a episode for episode); exactly 3 of 5 passed |
| 3, extension | Seeds 48–52 trained the same way; scored with `--extension`, the gate re-run first because the script changed | Gate passed again; 4 of 5 pass; 7 of 10 overall: **passes** |

## Results (sampled, the measurement the criterion uses)

Rate over valid episodes, Wilson 95 % bounds. Own hook: spec 2's hook (as in W1a), env seeds
52000–52499, 500 valid of 500. Handoff: env seeds 53000–53499, 488 valid of 500. Sampling seed 0.

| Seed | run_id | policy_sha256 | Stopping evals best / last (of 50) | Held-out, argmax, mixed hook | Own hook, sampled | Handoff, sampled | Seed |
|---|---|---|---|---|---|---|---|
| 43 | `20261115T125405-eaea36cd…` | `7d6c6432c17eee2c` | 44 / 43 | 43/50 | 456/500 = 0.912 [0.884, 0.934] | 482/488 = 0.988 [0.973, 0.994] | pass |
| 44 | `20261115T125405-265dd652…` | `eb4a6ffac20085b0` | 45 / 45 | 42/50 | 449/500 = 0.898 [0.868, 0.922] | 488/488 = 1.000 [0.992, 1.000] | **fail** |
| 45 | `20261115T125406-03312e12…` | `b05ae151ac43c398` | 43 / 40 | 41/50 | 459/500 = 0.918 [0.891, 0.939] | 488/488 = 1.000 [0.992, 1.000] | pass |
| 46 | `20261115T125406-3acc5eb0…` | `d6e41337ba3c1f91` | 45 / 44 | 43/50 | 479/500 = 0.958 [0.937, 0.972] | 486/488 = 0.996 [0.985, 0.999] | pass |
| 47 | `20261115T125406-1d8954e4…` | `e288776d05ed95dc` | 41 / 37 | 36/50 | 389/500 = 0.778 [0.740, 0.812] | 480/488 = 0.984 [0.968, 0.992] | **fail** |
| 48 | `20261115T135439-1f929506…` | `5539c11647145764` | 45 / 44 | 43/50 | 482/500 = 0.964 [0.944, 0.977] | 488/488 = 1.000 [0.992, 1.000] | pass |
| 49 | `20261115T135439-eeb13520…` | `b591e0e32b428a81` | 45 / 45 | 44/50 | 475/500 = 0.950 [0.927, 0.966] | 488/488 = 1.000 [0.992, 1.000] | pass |
| 50 | `20261115T135439-a1b83ba5…` | `e70a45bb25b7d109` | 42 / 41 | 40/50 | 448/500 = 0.896 [0.866, 0.920] | 487/488 = 0.998 [0.989, 1.000] | **fail** |
| 51 | `20261115T135439-edb7a0ed…` | `2dcf91e5bab78666` | 46 / 42 | 44/50 | 484/500 = 0.968 [0.949, 0.980] | 488/488 = 1.000 [0.992, 1.000] | pass |
| 52 | `20261115T135439-6a444cca…` | `bd7639819c5eeb64` | 44 / 44 | 43/50 | 477/500 = 0.954 [0.932, 0.969] | 488/488 = 1.000 [0.992, 1.000] | pass |

The stopping evaluations and the held-out score ran on the mixed hook (about 33 of the 50 fixed
stopping seeds fall on the handoff branch), so they are not comparable with spec 2's. Full run ids
and hashes are in the result JSONs; the run-store archives are under `runs/training/<run_id>/`
(not committed).

## Argmax, recorded beside it (a different measurement)

| Seed | Own hook, argmax | Failures with a state repeat | Handoff, argmax | Failures with a state repeat | Sampled timeouts, own / handoff | Mean steps of sampled successes, own / handoff |
|---|---|---|---|---|---|---|
| 43 | 328/500 = 0.656 | 172/172 | 464/488 = 0.951 | 24/24 | 44 / 6 | 18.0 / 10.6 |
| 44 | 320/500 = 0.640 | 180/180 | 486/488 = 0.996 | 2/2 | 51 / 0 | 20.8 / 7.7 |
| 45 | 305/500 = 0.610 | 195/195 | 485/488 = 0.994 | 3/3 | 41 / 0 | 18.4 / 7.3 |
| 46 | 307/500 = 0.614 | 193/193 | 488/488 = 1.000 | 0/0 | 21 / 2 | 18.8 / 7.3 |
| 47 | 293/500 = 0.586 | 207/207 | 440/488 = 0.902 | 48/48 | 111 / 8 | 17.4 / 11.4 |
| 48 | 359/500 = 0.718 | 141/141 | 478/488 = 0.980 | 10/10 | 18 / 0 | 18.4 / 7.7 |
| 49 | 339/500 = 0.678 | 161/161 | 488/488 = 1.000 | 0/0 | 25 / 0 | 16.5 / 7.1 |
| 50 | 303/500 = 0.606 | 197/197 | 469/488 = 0.961 | 19/19 | 52 / 1 | 18.7 / 8.2 |
| 51 | 329/500 = 0.658 | 171/171 | 485/488 = 0.994 | 3/3 | 16 / 0 | 19.1 / 7.1 |
| 52 | 352/500 = 0.704 | 148/148 | 487/488 = 0.998 | 1/1 | 23 / 0 | 17.8 / 7.3 |

Every argmax failure is a state repeat (H-7a.5's definition), on both rows and every model.

## Descriptive only: the same models on W1a's ranges, beside spec 2

Sampled success over valid episodes on W1a's ranges (own hook 50000–50499, handoff 51000–51499),
next to W1a's recorded spec-2 numbers for the same training seed. Not part of the verdict.

| Seed | Spec 3, own hook | Spec 3, handoff | Spec 2 (W1a record), own hook | Spec 2 (W1a record), handoff |
|---|---|---|---|---|
| 43 | 0.916 | 0.980 | 0.994 | 0.982 |
| 44 | 0.912 | 0.998 | 0.980 | 0.918 |
| 45 | 0.932 | 1.000 | 0.972 | 0.972 |
| 46 | 0.952 | 0.996 | 0.918 | 0.874 |
| 47 | 0.824 | 0.974 | 0.964 | 0.876 |
| 48 | 0.976 | 0.998 | 0.970 | 0.957 |
| 49 | 0.946 | 0.998 | 0.968 | 0.965 |
| 50 | 0.906 | 1.000 | 0.956 | 0.878 |
| 51 | 0.950 | 1.000 | 0.946 | 0.970 |
| 52 | 0.946 | 0.998 | 0.948 | 0.892 |

Readings, descriptive only: for the same training seed, spec 3's handoff rate is higher than
spec 2's on 9 of 10 seeds, and its own-hook rate is lower on 7 of 10.

## Key-position measurement (run during the plan review)

[`scripts/measure_handoff_key_position.py`](../../scripts/measure_handoff_key_position.py) →
[`runs/v2/w1a_handoff_key_position.json`](../../runs/v2/w1a_handoff_key_position.json). On W1a's
491 valid handoff states (env seeds 51000–51499): the key is never orthogonally adjacent to the door
(0/491); in all 491 the agent stands on the door's approach cell facing along the wall (264 south,
227 north) with the key on the cell in front of it; the chain uses 16.8 env steps on average (at
most 32). This replaced revision 1's unmeasured "cause on record" (plan revision 2, decision 0011).

## Checks

- **Step 1 gate** (200 env seeds): 197 equal, 3 rejected by the hook and invalid in evaluation,
  0 mismatches, directly, through `SubgoalEnv`, and under the mixture.
- **Step 3 gate**, run twice (before the first scoring and again before the extension, the script
  having changed): the ten spec-2 models reproduce W1a's recorded episodes on both rows, argmax and
  sampled, episode for episode (20,000 episodes, 0 differ).
- **Load checks:** each spec-3 model's weights hash to the `policy_sha256` its run recorded, and its
  argmax held-out score under the mixed hook equals the training run's.
- **Harness cross-check** (W1a's `score_row`): in every row the argmax episodes equal
  `evaluate_node` and the sampled ones `run_episode`.
- **Seed ranges:** 52000–53499 appear nowhere in `scripts/`, `src/`, `tests/` or `runs/` (JSON and
  logs, `runs/training/` included).

## Deviations and choices the plan did not spell out

Deviations from the plan: none in the mixture, budget, timeout, thresholds, seed lists or criterion.

Choices:
1. **Step 1 gate seeds:** 51000–51199 (W1a's handoff seeds; the plan says "200 env seeds"), chosen
   because three known invalid seeds lie in it. Beyond the plan's direct comparison, the gate also
   compares through `SubgoalEnv` (state and observation) and under the mixture.
2. **The coin flip** is one `_rand_float` draw from the env's own RNG. A rejected layout is redrawn
   with a new flip, so the handoff share among accepted episodes is close to, not exactly, 0.5
   (training envs, ten seeds: 173,774 handoff-branch and 175,536 spec-2-branch hook calls, 3,554
   of the handoff ones rejected and redrawn).
3. **Chain steps split:** training vs evaluation envs are told apart by env-instance order (the
   first `n_envs` instances are the training envs, which `make_training_vec_env` builds first).
4. **Step 3 gate** re-runs the spec-2 models with `eval_w1a.run_mode` and compares episodes; W1a's
   harness cross-checks are not repeated in the gate. Spec 3 is scored with W1a's `score_row`,
   cross-checks included.
5. **Extension:** the same script with `--extension`, writing a separate file; the first five seeds'
   verdict is read from the first file. The gate was re-run first, as Emir asked.
6. **Run records:** each run's `code_commit` reads `73bffca`, while the spec-3 scripts were not yet
   committed; the run store's `source.zip` holds the scripts that ran, with their sha256.
7. **Work location:** a separate worktree from `origin/main`; outputs were written to a scratch folder
   and copied unchanged into `runs/v2/` here.

## Cost

| | Env steps |
|---|---|
| Spec 3: 10 training runs × 1,001,472 (none stopped early) | 10,014,720 |
| Spec 3: chain steps inside training resets (own line, not added to the node's cost) | 3,715,701 |
| Spec 3: chain steps inside the stopping evaluations' resets | 515,000 |
| W1a on spec 2: 10 × 1,001,472 | 10,014,720 |
| The six withdrawn runs (spec 1) | 4,804,608 |
| Flat PPO on UnlockPickup (3 × 3,000,320, censored at 0/50) | 9,000,960 |

Evaluation episodes are counted on neither side. Wall time: about 22–23 min per training batch of
five (two batches); scoring 1,671 s (gate and seeds 43–47) and 1,420 s (gate and seeds 48–52).

## Files

- Results: [`runs/v2/w1a_spec3_sampled.json`](../../runs/v2/w1a_spec3_sampled.json) (gate, seeds
  43–47, first verdict), [`runs/v2/w1a_spec3_sampled_ext.json`](../../runs/v2/w1a_spec3_sampled_ext.json)
  (gate, seeds 48–52, ten-seed verdict); logs `runs/v2/w1a_spec3_eval.log`, `w1a_spec3_eval_ext.log`.
- Training reports: `runs/v2/pickup_box_s3_s{43..52}.json`; logs `runs/v2/pickup_box_s3_s{43..52}.log`.
- Key position: `runs/v2/w1a_handoff_key_position.json`.
- Scripts: [`pickup_box_spec3.py`](../../scripts/pickup_box_spec3.py),
  [`eval_w1a_spec3.py`](../../scripts/eval_w1a_spec3.py),
  [`measure_handoff_key_position.py`](../../scripts/measure_handoff_key_position.py).
- Decision: [`0011`](../kb/decisions/0011-pickup-box-spec-3.md) (`Proposed`).
