# Run summary — v2 migration, Step 0 → H-5 (2026-10-01)

Executed `docs/plans/2026-10-01_v2_H0-H5_plan.md` end to end on branch
`docs/architecture-v2` (branched off `fix/declare-missing-deps`, so `acd5ffd` stays in the base).
**All seven tasks passed their gates.** Nothing was escalated; no stop-and-report condition occurred.

Headline: the hierarchy works end to end. Three primitives and one composite are FROZEN, and
`{1,1} unlock_door` runs as a real SMDP — 2 decisions spanning 11 env steps, with `get_key` closing
SUCCESS at step 5 and `open_door` taking over.

---

## Gate results

| Task | Gate | Result |
|---|---|---|
| Step 0 | branch off `fix/declare-missing-deps`, clean tree | ✅ `acd5ffd` verified an ancestor of HEAD |
| H-0 | `pytest` 58 + 1 new | ✅ 59 passed |
| H-0.5 | deps resolve, MaskablePPO imports, tests green | ✅ 59 passed; sb3-contrib 2.9.0 |
| H-1 | new tests pass, v1 tests still green | ✅ 108 passed (49 new) |
| H-2 | predicates verified against real minigrid 3.1.0 | ✅ 150 passed (42 new) |
| H-3 | a success rate for `col01` under the v2 criterion | ✅ 171 passed (21 new); col01 50/50 |
| H-4 | 3 FROZEN primitives + parent-reachable scores | ✅ 189 passed; gate passed after one retrain |
| H-5 | `{1,1} unlock_door` FROZEN with both C1 controls | ✅ 190 passed; held-out SR 0.98 |

Seven commits, one per task, each carrying its gate result.

---

## Trained nodes

All success rates are on the **held-out** seed set (20000–20049), which is never used for stopping.

| Node | Kind | Env(s) | Steps | Held-out SR | Mean length |
|---|---|---|---|---|---|
| `{0,1} get_key` | primitive | Unlock + DoorKey-6x6 | 180,000 | **1.00** (50/50) | 5.8 |
| `{0,2} open_door` | primitive | Unlock + DoorKey-6x6 | 100,000 | **1.00** (50/50) | 6.2 |
| `{0,3} go_to_goal` | primitive | DoorKey-6x6 | 200,000 | **0.92** (46/50) | 16.1 |
| `{1,1} unlock_door` | composite (n=1) | Unlock | 6,000 decisions | **0.98** (49/50) | 19.2 |

Throughput ~1,050 env steps/s on CPU with 8 `DummyVecEnv`s; ~61 SMDP decisions/s for the composite.
No GPU on this machine (Intel Arc integrated only), and none needed.

---

## What the C1 controls actually show

Same 50 held-out seeds, all five arms:

| arm | SR | mean env steps |
|---|---|---|
| random over all children (mask clipped) | 0.94 | 83.0 |
| random over available children | 0.90 | 85.3 |
| **random, initiation masking on** | **0.98** | **19.2** |
| learned composite | 0.98 | 19.2 |
| learned composite + initiation masking | 0.98 | 19.2 |

**A random policy with initiation masking matches the learned composite exactly.** At depth 1 the
hand-written mask solves the task outright, so success rate alone does not demonstrate that
composition helped — which is precisely why the plan made these controls mandatory.

What the learned run does support:

- **It discovered the ordering constraint rather than being told it.** Trained under a goal-only
  mask, it went 0.22 → 1.00 between 2k and 4k decisions. Of the 146 decisions (out of 304) where the
  strict mask *would* have withheld a child, the learned policy had already declined that child in
  all 146.
- **Efficiency.** 19.2 env steps against 85.3 for random-over-available: 4.4× fewer.
- With the strict mask the composite is STABLE at 4,000 decisions instead of 6,000 — the mask buys
  convergence speed, not final performance.

This is the honest form of C1 at this depth. H-6 (depth 2, `solve_doorkey`) is where the claim gets a
harder test, because the mask no longer determines the order on its own.

---

## Deviations from the plan, and why

1. **`randomize_carrying()` hook added (not in the plan).** The F29 gate caught `go_to_goal` scoring
   0.96 on its own hook distribution but **0.52** from the state `unlock_door` actually hands it.
   Cause, confirmed not guessed: MiniGrid writes the carried object into the agent's *own*
   observation cell (`gen_obs_grid` sets `grid[3,6] = self.carrying`). `go_to_goal`'s hook never gave
   it anything to hold — 0/200 resets — so that cell read object index 1 in every training state and
   reads 5 at the handoff. The policy had never seen that input. Fixed by randomizing the carried
   object 50/50, which is the plan's prescribed "widen the mix" response. The real handoff went
   **0.52 → 1.00**.

2. **`initiation` predicates declared on the primitives (gap in my own H-4).** The C1 ablation
   produced byte-identical numbers; checking rather than reporting showed the strict mask differed
   from the goal-only mask in **0 of 304** decisions, because `build_node()` never passed
   `initiation` and everything defaulted to `ALWAYS`. This was a real gap — `initiation` is half the
   task representation v2 §2 requires — not merely a broken control. Now: `get_key` carries nothing,
   `open_door` carries the key, `go_to_goal` needs the door open. Divergence is now 146/304.

3. **Reset hook folded into `spec_hash`.** The hook determines the training state distribution, so
   two nodes with the same goal and timeout but different hooks are not interchangeable. Adding it
   (plus `initiation`) correctly invalidated all three checkpoints; the registry refused to load
   them, which is F18 working as designed. All three were retrained with unchanged results.

4. **`initiation_masking` threaded through the evaluation path.** The ablation was training with one
   mask and scoring with the other, so the arms were not comparable.

5. **F29 gate scoped to genuinely reachable states.** Two cases are reported but excluded, with
   reasons: `[probe] door open, empty-handed` (SR 0.66 — no parent produces this; `unlock_door`
   always leaves the agent holding the key) and `[expected fail] door still shut` (SR 0.00 — the goal
   is unreachable and timing out is correct, and is the signal the parent learns from).

6. **PPO hyperparameters overridden.** SB3's defaults are wrong here: `n_steps=2048` × 8 envs buys
   only ~36 gradient updates in a 600k budget, and `ent_coef=0` gives no exploration pressure on
   goals that pay nothing until the key is picked up. Now `n_steps=256`, `ent_coef=0.01`.

7. **`runs/pn4/NOTES.md` force-added.** `runs/*` is gitignored, but that file is a record, not an
   artifact, and the durable version of the 0.75/0.85 note belongs in the repo.

8. **`stabilization.py` deprecated, not deleted** — per review correction R1, since `pn_trainer.py`
   imports it and `src/continual_learning/__init__.py` re-exports it.

---

## Things worth knowing before H-6

- **`go_to_goal` is weak from the fixed default start** (SR 0.66) while scoring 0.92 from randomized
  starts. No parent produces that exact state, so it does not gate — but it means the randomized
  training distribution is *easier* than some states in the env, and the held-out 0.92 is slightly
  optimistic for left-room starts. If H-6 underperforms, this is the first place to look.
- **The legacy `col01` re-evaluation is a harness check, not evidence.** `MiniGrid-Empty-8x8-v0` is
  layout-deterministic — start (1,1) and goal (6,6) for every seed, verified over 20 seeds — so its
  50/50 is one episode repeated 50 times and the Wilson bound there is vacuous. Recorded in the
  output JSON, not just here.
- **The Wilson bound earned its place twice in live runs.** `get_key` at 140k and `go_to_goal` at
  140k/160k both hit SR exactly 0.900 (45/50, LB 0.786) and were correctly *not* counted. Under the
  plain "≥0.90 twice" rule both would have started a streak on a lucky evaluation.
- **`{1,1} unlock_door`'s timeout is 240** against Unlock's own 288, with children at 120 and 60. The
  F13 invariant is checked at construction and caught two bad fixtures while I was writing tests.
- **Open supervisor questions Q8 and Q9** were added in H-0.5 and are unanswered: Q8 asks how to frame
  the masking contribution in the report, Q9 asks whether to retrain `col01` for the PN comparison
  now that observations are normalized.

---

## Repository state

Branch `docs/architecture-v2`, 7 commits ahead of `fix/declare-missing-deps`:

```
6eed565 H-5: {1,1} unlock_door trained and frozen -- the first composite
e205eb3 H-4: three primitives trained and frozen, and the F29 gate catches a real bug
9704a88 H-3: EvaluationMonitor, device selection, and the legacy col01 re-evaluation
3b878b2 H-2: observation pipeline, predicates, reset hooks and SubgoalEnv
bbb4512 H-0.5: apply Part A amendments to the v2 design docs; pin dependencies
e77d0ab H-0: fix frozen-column determinism bug; correct lateral docstring
48ffbce docs: lock Target Architecture v2 and the H-0..H-5 plan
```

New code: `src/skills/` (node, state, predicates, executor, registry), `src/envs/` (obs, subgoal_env,
option_env), `src/training/` (device, evaluation, node_trainer). 190 tests pass; the 59 v1 tests are
all still green, and no v1-only module was extended.

Artifacts in `runs/v2/`: `primitives.json`, `unlock_door.json`, `c1_controls.json`,
`parent_reachable_eval.json`, `legacy_col01_eval.json`, and the training logs.
