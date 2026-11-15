# W1a handoff diagnostic summary: why do some `pickup_box` models time out at the handoff?

> Run 2026-11-15 on `main` at `c75cdb1` (W1a merged), CPU (Windows), following
> [`2026-11-15_W1a_handoff_diagnostic_plan.md`](2026-11-15_W1a_handoff_diagnostic_plan.md).
> Evaluation only: nothing was trained, frozen or registered, no rule changed, no file under `src/`
> changed, and no existing file under `runs/v2/` was written. K and T are counterfactual
> diagnostics, not results for `pickup_box`.

## Reading: Key

Over the four failing seeds pooled (46, 47, 50, 52; 1,964 valid episodes):

| Condition | Success over valid episodes |
|---|---|
| Unchanged (W1a's sampled handoff episodes) | 0.880 [0.865, 0.893] |
| K: the dropped key taken off the grid | **0.985** [0.979, 0.990] |
| T: 240 steps instead of 120 | 0.921 [0.908, 0.932] |

- **Key holds** (K ≥ 0.95). **Time does not** (T < 0.95). **Search does not**: K lifts the rate, and
  no unchanged failure had both no key pick-up and the box never in view (0 of 236).
- On the six passing seeds pooled, Key holds as well (K 0.997) and so does Time (T 0.984). Key holds
  for failing and passing seeds alike, so per the plan it is reported as a property of the skill,
  not of the failing seeds.

## What was run

| Step | What | Result |
|---|---|---|
| 1 | `scripts/diagnose_w1a_handoff.py` and its gate: the unchanged condition must reproduce W1a's sampled handoff episodes | Passed for all 11 models (42–52), episode for episode (validity, success, steps), 5,500 episodes; the per-seed success counts equal W1a's |
| 2 | The same episodes traced (env seeds 51000–51499, sampling seed 0) | Gate passed again; per-episode trace for all models |
| 3 | Interventions K and T on the same models, env seeds and sampling seed | Gate passed again; T's first 120 steps equal the unchanged episodes (checked) |

Step 3's run wrote every part (gate, trace, K, T) at once; that file is
[`runs/v2/w1a_handoff_diagnostic.json`](../../runs/v2/w1a_handoff_diagnostic.json) (538 s).

## Per seed

Success over valid episodes (491 of 500 for every model), Wilson 95 % bounds.

| Seed | Unchanged | K: key removed | T: 240 steps |
|---|---|---|---|
| 42 (exploratory) | 480/491 = 0.978 [0.960, 0.987] | 491/491 = 1.000 [0.992, 1.000] | 490/491 = 0.998 [0.989, 1.000] |
| 43 | 482/491 = 0.982 [0.966, 0.990] | 491/491 = 1.000 [0.992, 1.000] | 490/491 = 0.998 [0.989, 1.000] |
| 44 | 451/491 = 0.918 [0.891, 0.940] | 489/491 = 0.996 [0.985, 0.999] | 470/491 = 0.957 [0.935, 0.972] |
| 45 | 477/491 = 0.972 [0.953, 0.983] | 491/491 = 1.000 [0.992, 1.000] | 487/491 = 0.992 [0.979, 0.997] |
| 46 (failing) | 429/491 = 0.874 [0.841, 0.900] | 475/491 = 0.967 [0.948, 0.980] | 445/491 = 0.906 [0.877, 0.929] |
| 47 (failing) | 430/491 = 0.876 [0.844, 0.902] | 490/491 = 0.998 [0.989, 1.000] | 450/491 = 0.916 [0.889, 0.938] |
| 48 | 470/491 = 0.957 [0.935, 0.972] | 488/491 = 0.994 [0.982, 0.998] | 484/491 = 0.986 [0.971, 0.993] |
| 49 | 474/491 = 0.965 [0.945, 0.978] | 488/491 = 0.994 [0.982, 0.998] | 485/491 = 0.988 [0.974, 0.994] |
| 50 (failing) | 431/491 = 0.878 [0.846, 0.904] | 490/491 = 0.998 [0.989, 1.000] | 454/491 = 0.925 [0.898, 0.945] |
| 51 | 476/491 = 0.970 [0.950, 0.981] | 490/491 = 0.998 [0.989, 1.000] | 482/491 = 0.982 [0.966, 0.990] |
| 52 (failing) | 438/491 = 0.892 [0.862, 0.916] | 480/491 = 0.978 [0.960, 0.987] | 460/491 = 0.937 [0.912, 0.955] |
| **Failing, pooled** | 1728/1964 = 0.880 [0.865, 0.893] | 1935/1964 = 0.985 [0.979, 0.990] | 1809/1964 = 0.921 [0.908, 0.932] |
| **Passing, pooled** | 2830/2946 = 0.961 [0.953, 0.967] | 2937/2946 = 0.997 [0.994, 0.998] | 2898/2946 = 0.984 [0.979, 0.988] |

## What the failures look like

| Group, condition | Failures | Timeouts | Key picked up | Never entered the right room | Box never in view | No key pick-up and box never in view | Mean steps holding the key | Ended in the left room, hand empty |
|---|---|---|---|---|---|---|---|---|
| Failing, unchanged | 236 | 236 | 194 | 199 | 0 | 0 | 6.1 | 208 |
| Failing, K | 29 | 29 | 0 | 17 | 0 | 0 | 0 | 22 |
| Failing, T | 155 | 155 | 133 | 128 | 0 | 0 | 10.0 | 138 |
| Passing, unchanged | 116 | 116 | 113 | 66 | 0 | 0 | 9.7 | 82 |
| Passing, K | 9 | 9 | 0 | 5 | 0 | 0 | 0 | 5 |
| Passing, T | 48 | 48 | 48 | 23 | 0 | 0 | 14.6 | 35 |

Base rates from the unchanged successes:

| Group | Successes | Key picked up | Never entered the right room | Mean first step in the right room | Mean first step with the box in view | Mean steps |
|---|---|---|---|---|---|---|
| Failing | 1,728 | 568 (33 %) | 179 (10 %) | 13.2 | 1.3 | 20.2 |
| Passing | 2,830 | 822 (29 %) | 275 (10 %) | 10.1 | 1.2 | 16.4 |

Descriptive notes:
- Every valid episode starts `pickup_box` in the left room, and the box is in view within about
  one step on average. Every failure in every condition is a timeout; none ran into the env's 288-step limit.
- A key pick-up also occurs in about a third of the successes, so a pick-up alone does not mark a
  failure.
- Under T, 81 failing-seed episodes (68 passing) succeed after step 120; 155 failing-seed episodes
  still time out at 240, 133 of them with a key pick-up.
- Under K, 29 failures remain on the failing seeds, 16 of them on seed 46 and 11 on seed 52.
- Seed 52 differs from the other failing seeds unchanged: a key pick-up in 26 of its 53 failures.
- Seed 42 (exploratory, in no pooled number) behaves like the passing seeds.

## Checks

- **Gate:** 11 of 11 models reproduce W1a's sampled handoff episodes exactly, in each of the three runs.
- **Load checks** (from `eval_w1a.py`): every model's weights hash to the `policy_sha256` its run
  recorded, and its argmax held-out score equals its training run's. Run-store models are read
  from the main checkout's `runs/training/` (`ENS491_RUNS_ROOT`); seed 42 from `runs/v2/0-4_pickup_box/`.
- **K:** the observation regenerated through the env's own wrappers equals the chain's before the
  key is removed (checked every episode). In every valid episode the hand is empty, so the key is
  on the grid and was removed; no K episode picked up a key.
- **T:** within the first 120 steps every T episode equals its unchanged episode (every unchanged
  success recurs with the same steps; no T success within 120 steps is new).

## Deviations and choices the plan did not spell out

Deviations from the plan: none in models, seeds, sampling seed, interventions, readings or thresholds.

Choices (also in the result file under `implementation_choices`):
1. **Gate** compares episode for episode (validity, success, steps), stricter than the per-seed
   counts the plan asks for.
2. **K:** applied only to valid episodes; the key's cell is set empty and the observation
   regenerated through the env's observation wrappers.
3. **T:** a second node built from the same spec with timeout 240, sharing the loaded policy
   object. Its `spec_hash` differs; it is never saved.
4. **Box in view:** H-7a.5's definition (a box in the agent's 7×7 view); step 0 = at the start.
   **Right room:** agent x > door x. **Steps holding the key:** env steps after which a key is carried.
5. **Seed 42** is traced and labelled exploratory; it is in no pooled number or reading.
6. **Work location:** a separate worktree from `origin/main`; steps 1 and 2 wrote their outputs to
   a scratch folder only, and step 3's full output is the committed result file.

## Files

- Result: [`runs/v2/w1a_handoff_diagnostic.json`](../../runs/v2/w1a_handoff_diagnostic.json)
  (per-model summaries for each condition, readings, and every episode in compact form); log
  `runs/v2/w1a_handoff_diagnostic.log`.
- Script: [`scripts/diagnose_w1a_handoff.py`](../../scripts/diagnose_w1a_handoff.py).
- Plan: [`2026-11-15_W1a_handoff_diagnostic_plan.md`](2026-11-15_W1a_handoff_diagnostic_plan.md).
