# Session handoff — 2026-10-02 (H-6 corrections, H-6.9, H-7 up to the A2 stop)

For the next agent picking this up. Read this first, then the documents it points to. Everything
below is committed and pushed on `docs/architecture-v2`, and the working tree was clean at hand-off.

## TL;DR

- **H-7 is stopped at H-7.3** on plan assumption **A2**, which the plan marks as a stop:
  `{0,4} pickup_box` was not stable within 600,000 env steps on any of three seeds (held-out
  0.28 / 0.40 / 0.24, still rising). Every H-7 composite needs `pickup_box` as a FROZEN child, so
  H-7.4–H-7.8 cannot run until the author decides how to proceed (§5).
- Done before the stop: the H-6 summary corrections, the H-6.9 diagnosis, H-7 Part A (v2
  amendments), the H-7.1 pre-flight, and all nine flat PPO baselines (H-7.2, collected).
- No CLAUDE.md stop-and-report condition occurred. No design change was made outside v2 and the
  H-7 plan's Part A.

## Documents to read, in order

1. `CLAUDE.md`: hard rules. v2 overrides every other doc.
2. `docs/plans/2026-10-02_H7_plan.md`: the plan being executed (assumptions A1–A8, which ones are
   "stop", the budgets, the wording rules for results).
3. `docs/plans/2026-10-02_H7_summary.md`: **the H-7 stop report**: gate table, measurements,
   deviations, success criteria one by one.
4. `docs/plans/2026-10-02_H6_summary.md`: §9 C.9 (this session's corrections) and §10 (H-6.9).
5. `docs/ENS491_Project_State.md`: H-6.9 and H-7 rows updated.
6. `docs/ENS491_Target_Architecture_v2.md`: §11 and §13, amended for H-7 (Part A).

`docs/HANDOFF.md` and `docs/SUPERVISOR_BRIEF_2026-10.md` are advisory-session documents. They are
listed in `.git/info/exclude`, quote superseded figures, and are the author's to update. Leave them
alone.

## Commits of this session (oldest first)

| commit | what |
|---|---|
| `e7be388` | docs: H-7 plan (un-excluded from `.git/info/exclude` and committed) |
| `5a4c353` | three corrections to the H-6 summary and Project_State |
| `79426d2` | H-6.9: choice-task diagnosis, no verdict |
| `273a79d` | H-7.0: Part A amendments to v2 |
| `80aa039` | H-7.1: pre-flight |
| `3530ea4` | H-7.2: flat baselines launched |
| `3ad70db` | H-7.3: `pickup_box` not stable, A2 stop |
| `e1fe937` | H-7 summary (stop report) |
| `554a6ff` | H-7.2 collected: six censored flat baselines |
| *(this file)* | session handoff |

## 1. Step B: corrections to the H-6 summary (`5a4c353`)

Each was checked against `runs/v2/` first, then recorded in H-6 summary §9 C.9.

1. **§4, strict-mask choice runs.** Only seed 42 is constant (blue first 49/50, stopping success
   20/50 from update 6 to 99). Seeds 43 and 44 pick blue first in 29 and 32 of 50; their stopping
   success sits near 20/50 until about update 50, then rises to 25–30/50. Seed 43 picks the right
   key in 36/50. A5 still fails on all three seeds. The new wording: no run learned the choice within
   100 updates; under the strict mask two of three runs were leaving the one-key plateau; the cause is
   not established. Project_State's "No, on the one task built to test it" was rewritten to match.
2. **§3, hyperparameters.** Since H-6.8 composites use `n_steps` 64 and flat PPO uses 256. This is
   now stated beside the table, together with the fact that flat PPO's update size was never
   revisited.
3. **"Depth wins 1.3–1.4×"** now always carries the with-children totals: 661–667k for
   `solve_doorkey` against 544–561k for the flat option set.

## 2. Step C: H-6.9 (`79426d2`, H-6 summary §10)

`scripts/diagnose_choice.py` → `runs/v2/choice_diagnosis.json`.

- **Sanity, 400 resets:** the door is in view and the strict mask leaves exactly the two key
  children, **400/400** on both. No bug. The door appears at 18 distinct view cells.
- **Decodability**, 5,000 decision-1 observations on seeds 30000–34999, tested on the last 1,000.
  The classifier is the selector's own `MaskableActorCriticPolicy` actor head, trained supervised,
  mean of 3 initialisations: **0.498 / 0.731 / 0.940 / 0.981 / 0.997** at 250 / 500 / 1,000 / 2,000 /
  4,000 examples. Logistic regression plateaus at about 0.91. A hand-written door-colour reader
  scores 1.000.
- **Verdict: none.** The rule was ≥ 0.95 at ≤ 1,000 examples → RL optimisation; < 0.95 at 4,000 →
  representation limit; anything else → no verdict. The result is in between, so no verdict was
  drawn and **the two diagnostic RL arms were not run**. The script has `--stage arm` ready for them.
- A fixed-budget fit, added post hoc and not used for the verdict, reaches 0.962 at 1,000. The
  primary fit's early stopping halted the 250 and 500 runs at train accuracy 0.62–0.74.

## 3. Step D: H-7

### Step 0 and H-7.0
- Step 0 passed: HEAD `79426d2`, regression 8 of 8 at delta 0.000.
- **H-7.0 (`273a79d`):** v2 amended with A.1–A.7 (T6–T9b specs, four baseline rows, the R1–R4
  table, `chain_ceiling`, `weak_heldout`, n_steps 64 restated in §5.2) plus an H-7 block in §13.

### H-7.1, pre-flight (`80aa039`, `runs/v2/h7_preflight.json`)
- New library code:
  - `door_approach_blocked()` in `src/skills/predicates.py`.
  - `relocate_blocker()` in `src/envs/subgoal_env.py`, with a guard: the ball may not cut the agent
    off from the key, the door or a reachable box.
  - `tests/test_h7_hooks.py` (12 tests, including the V9 regression).
- `scripts/h7_nodes.py` holds **every H-7 node spec and builder**. Always build nodes from it so
  `spec_hash` cannot drift.
- **V1–V7 all reproduce** on this env.
- **A1: `{1,1} unlock_door` transfers to UnlockPickup untouched at 0.98** (LB 0.895). On
  BlockedUnlockPickup from episode start it scores 0.00, as expected.
- **Solvability: 0/500 unsolvable** for both new hooks, by exhaustive BFS over MiniGrid's rules.
  This is deviation 1: the plan's flood-fill proxy flagged 13 + 3 layouts that are actually
  solvable by picking up the obstacle.
- **Full-hand rows** (not gating). In BlockedUnlockPickup after `relocate_blocker`, the frozen
  **`get_key` picks up the ball instead of the key in 29/50** (SR 0.38), `unlock_door` scores 0.34,
  and `open_door` with the key 0.74. No child ever drops anything. This strongly suggests H-7.5's
  A5 will take the **rebuild branch**.

### H-7.2, flat baselines (`3530ea4`, collected in `554a6ff`)
- `scripts/train_flat_baseline.py` got `--task {doorkey,unlock,unlockpickup,blocked}`. `doorkey`
  is the default and its node ids are unchanged.

| flat PPO | 42 | 43 | 44 |
|---|---|---|---|
| Unlock (240) | STABLE 450k, held-out 0.96 | 350k, 0.94 | 450k, 0.92 |
| UnlockPickup (288) | **> 3,000,000, best SR 0.00** | same | same |
| BlockedUnlockPickup (576) | **> 3,000,000, best SR 0.00** | same | same |

UnlockPickup and BlockedUnlockPickup scored 0/50 at all 30 evaluations on every seed; these runs
are censored. The hierarchy's own cost for Unlock is 411,817 (at `n_steps` 256; H-7.4's
re-measurement at 64 was not reached). `{1,B} flat_unlockpickup_options` has not been run; it
needs a frozen `pickup_box`.

### H-7.3, `{0,4} pickup_box`: the stop (`3ad70db`)
- `scripts/train_h7_primitives.py`, budget 600,000, `eval_every` 20,000, standard rule.

| seed | best stopping SR | held-out (LB) |
|---|---|---|
| 42 | 0.30 | 0.28 (0.175) |
| 43 | 0.40 | 0.40 (0.276) |
| 44 | 0.44 | 0.24 (0.143) |

  All three were near 0 until 280–380k steps and still rising at the budget.
- Seed 42 is saved as **TRAINING (not frozen)** in `runs/v2/0-4_pickup_box/`; seeds 43 and 44 were
  `--no-freeze`.
- Diagnosis (`scripts/diagnose_pickup_box.py` → `runs/v2/pickup_box_diagnosis.json`): the seed-42
  checkpoint scores 0.00–0.10 from the env's own start pose with the door open, and **0.10 at the
  real handoff after `unlock_door`**. The training hook draws the pose over both rooms, so many
  episodes start beside the box; the policy mostly learned the short case, not crossing the door
  with the key. That is a reading of one checkpoint, not a tested cause.

## 4. Written but never run

These need a frozen `pickup_box`. None of them has run, not even with `--smoke`.

- `scripts/h7_composite.py`: the shared composite experiment. It trains goal-only, then strict;
  runs the five arms, `mask_divergence` and `selection_sequences`; does a depth check; and writes
  `chain_ceiling`.
- `scripts/train_unlock_pickup.py`: T7. Its `chain_ceiling` reads the `pickup_box` F29 row, which
  does not exist yet.
- `scripts/train_blocked_pickup.py --variant {a,b} [--rebuilt]`: T9a and T9b. `--rebuilt` imports
  `h7_rebuild.load_rebuilt`, **which does not exist yet**; write it if A5 takes the rebuild branch.
  The `_b` node specs are in plan H-7.5, and v2 §11 must be amended first.
- `scripts/eval_h7_preflight.py --stage pickup_box | blocked | full_hand`: the later stages exist
  but have not been run (`full_hand` ran once, before the H-7 primitives existed).
- `selection_sequences()` in `src/training/c1_controls.py` has a unit test. `weak_heldout` is added
  in `freeze_and_save` (A.6).

## 5. The open decision (the author's, not the agent's)

The plan gives `pickup_box` no fallback and forbids per-node stopping rules, so every option below is
a plan or spec change:

1. **More budget**: all three curves were rising at 600k.
2. **A different reset hook**, e.g. anchor the start pose in the left room, where the real handoff
   happens. This renames the hook's spec string, changes v2 §11 T6 / A.1, and must be amended in
   v2 first.
3. **Report as is**: the cost the reuse argument has to carry.

After a `pickup_box` is FROZEN, resume at H-7.3's remaining step: `eval_h7_preflight.py --stage
pickup_box`, which must give a gating row ≥ 0.90. Then H-7.4 (`train_unlock_pickup.py`, the options
baseline, and the `unlock_door` re-measurement at `n_steps` 64), H-7.5, H-7.6, H-7.8. H-7.7's
PPO half is already done.

## 6. Environment notes (learned the hard way this session)

- **Interpreter:** `C:\Users\memir\miniconda3\envs\ens491\python.exe` (`conda` is not on PATH).
  Windows 11, Intel Core Ultra 7 155H (22 logical CPUs), CPU only.
- **Long runs:** launch detached with PowerShell `Start-Process cmd.exe /c "python -u ... > log
  2>&1"`. A Bash `run_in_background` command is killed after 2 hours.
- **Power:** on battery, Windows caps everything at about 4.8 cores, and a **BelowNormal-priority
  process is starved to about 0.08 cores**. Plugged in, a process gets about 1–2 cores. Check
  `Win32_Battery.BatteryStatus` (2 = on AC) before long runs.
- **Throughput:** about 600 env steps/s per flat PPO process when nine run concurrently plugged in.
  A failing evaluation costs 12,000 / 14,400 / 28,800 env steps on Unlock / UnlockPickup / Blocked.
- **Bash-tool heredocs** containing Python with nested quotes sometimes fail to parse
  ("unexpected EOF"). Write edit scripts to the scratchpad and run them instead.
- `*.log` is git-ignored; tracked logs are force-added (`git add -f`).
- **Workflow:** gates are self-checks, one commit per task, `--smoke` before every full run, push
  after every commit, full suite after every task (248 tests at hand-off).
