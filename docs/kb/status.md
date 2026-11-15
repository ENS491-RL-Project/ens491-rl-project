# Status

Where each track stands and what the open PRs close. Last updated: 2026-11-15.

Task-level status is under [Roadmap](#roadmap); measurements are in [`results.md`](results.md).
Migrated from `ENS491_Project_State.md` on 2026-11-15; its last full text is in
[`archive/ENS491_Project_State_2026-11-15.md`](../archive/ENS491_Project_State_2026-11-15.md).

## Tracks

| Track | Done | Stopped / open | Next | Source |
|---|---|---|---|---|
| H Hierarchy (critical path) | H-0…H-6, H-6.8, H-6.9. 9 frozen nodes; `{2,1} solve_doorkey` depth 3, held-out 1.00. Training/evaluation parity (PR #15, 2026-11-12); training parity audit and correct update counter (PR #16, 2026-11-15); every training run archived under `runs/training/`, freeze needs STABLE, no registry overwrite (PR #17, 2026-11-15) | H-7 and H-7a stopped: `{0,4} pickup_box` not STABLE under two specifications. W1a (2026-11-15): spec 2 scored as it acts (sampled) **fails, 6/10 seeds** (the handoff row fails on 4). W1a handoff diagnostic (2026-11-15, evaluation only): reading **Key** (removing the dropped key lifts the failing seeds to 0.985; doubling the timeout to 0.921). W1a spec 3 (2026-11-15, decision 0011, Accepted 2026-11-15): half of training at the real handoff, W1a's test unchanged, **passes, 7/10 seeds** (handoff row passes on all ten; the three failures are on the own-hook row). W1b-1 (2026-11-15, decision 0014 `Proposed`): spec-3 seed 45 frozen into the git-ignored `runs/w1b/` (not registered); `{2,2} unlock_pickup` **STABLE under 0004 on 3 of 3 seeds**, fresh 486/500 each, held-out 49/50 (flat 0/50); the three seeds act identically, and random selection under the strict mask ties it (reuse shown, a learned choice not). Nothing in the canonical registry. H-7b (BlockedUnlockPickup) not run | Emir decides 0014; if accepted, a separate PR registers the child and the parent (seed 43) and edits v2 §9, §11; then W2. W1b-2 (sampling, 0012, flat with saved models) as an improvement question ([W1b-1 summary](../plans/2026-11-15_W1b-1_summary.md), [open-questions](open-questions.md)) | [Roadmap](#roadmap) Track H; [`plans/2026-10-02_H7_summary.md`](../plans/2026-10-02_H7_summary.md) |
| C Continual learning / PN | C-2 as W7 (2026-11-15): `go_to_goal` with no lateral, a lateral from the frozen `{0,2} open_door` ([0013](decisions/0013-c2-lateral-source-open-door.md)) and a lateral from an untrained network, ten seeds each. **No measurable benefit**: median steps to STABLE 190,464 / 178,176 / 184,320, no arm beats another; PN forward transfer not claimed on this evidence ([summary](../plans/2026-11-15_W7_summary.md)) | C-1, C-3 not started | Whether a second C-2 pair is worth trying is Emir's call (W7 plan, "After"); no tuning or re-run of this pair | [Roadmap](#roadmap) Track C |
| R Recognition | R-2 in part: GRU task identification on 9 known tasks, 85 % task-episode macro accuracy at N ≥ 10 (PR #18, 2026-11-15) | R-1, R-3 not started. R-2: old-task accuracy and switching latency not measured | Task registry (R-3) and root selection ([open-questions](open-questions.md)) | [Roadmap](#roadmap) Track R; [`results.md`](results.md) |
| G Growth | — | G-1…G-3 not started | Required this semester ([0007](decisions/0007-automatic-growth-in-scope.md)) | [Roadmap](#roadmap) Track G |
| UI | — | UI-1, UI-2 not started | — | [Roadmap](#roadmap) Track UI |

## Open PRs

| PR | Author | What it closes | Review | KB follow-up when merged |
|---|---|---|---|---|
| #5 refactor src/ structure | TulayEce | v1-era `src/` restructure, open since 2026-05-11 | Not reviewed | Decide: close or rebase onto v2 |

## Waiting on a decision

- Emir sets [0009](decisions/0009-max-decisions-per-frame.md)'s status (PR #15 merged with the per-frame design).
- Owner of the strict-mask dead-end row in [open-questions](open-questions.md).
- Emir sets [0014](decisions/0014-freeze-without-live-stable.md)'s status (freezing a primitive that never met 0004; first use `pickup_box` spec 3, seed 45). Its experiment, [W1b-1](../plans/2026-11-15_W1b-1_summary.md), ran on 2026-11-15 while 0014 is `Proposed`: passes, 3 of 3 seeds.

## Roadmap

Task definitions are in v2 §12. ⬜ not started · 🔄 in progress · ✅ done

### Track H — Hierarchy (critical path)

| # | Task | SP | Status |
|---|---|---|---|
| H-0 | Fix deterministic flag; document 0.75 / 0.85 mismatch | 1 | ✅ 2026-10-01 — 59/59 tests pass |
| H-0.5 | Apply Part A amendments to v2 / CLAUDE.md / supervisor questions; pin deps, add sb3-contrib | 2 | ✅ 2026-10-01 — reqs resolve, MaskablePPO imports, 59/59 |
| H-1 | `SkillNode`, `Frame`, `Executor` + unit tests (depth-5 synthetic graph) | 8 | ✅ 2026-10-01 — 108/108 tests (49 new) |
| H-2 | Predicates + `SubgoalEnv` for T1, T2, T4; verify env IDs | 5 | ✅ 2026-10-01 — 150/150 tests; env IDs + budgets verified |
| H-3 | `EvaluationMonitor`; re-evaluate legacy `col01` | 3 | ✅ 2026-10-01 — 171/171; col01 50/50 (but Empty-8x8 is layout-deterministic) |
| H-4 | Train and freeze T1 `get_key`, T2 `open_door`, T4 `go_to_goal` | 5 | ✅ 2026-10-01 — all three FROZEN; held-out 1.00 / 1.00 / 0.92; F29 gate passed after retraining go_to_goal |
| H-5 | `OptionEnv` + MaskablePPO; T3 `unlock_door` | 8 | ✅ 2026-10-01 — `{1,1} unlock_door` FROZEN, held-out 0.98 in 6k decisions; both C1 controls run |
| H-6 | T5 `solve_doorkey` + C1 baselines (first hierarchical result) | 8 | ✅ 2026-10-02 — `{2,1} solve_doorkey` FROZEN, held-out **1.00** on 3 seeds, depth-3 stack confirmed. Flat PPO 300–450k env steps vs the composite's **~53k own / ~664k with children** (H-6.8 figures). The genuine-choice task gives a **negative** result after 100 policy updates (§4 of the summary) |
| H-7a | Amendment 1: UnlockPickup only, `pickup_box` split into `drop_held` + `pickup_box` | — | ⛔ **Stopped at H-7a.2 on A2b** (2026-10-02). `{0,9} drop_held` FROZEN (30,000 steps on all 3 seeds; 1.00 at the handoff). `{0,4} pickup_box` (spec 2) **not stable in 1,000,000 on any seed** (held-out 0.70 / 0.66 / 0.64): 0.96 from the box's room, 0.58–0.60 from the left room / real handoff — the failures are in left-room starts, measured twice; why is not established (close-out §2, K.4). **H-7a.5 (evaluation only, one checkpoint):** at the real handoff all 20 deterministic failures are state repeats (16 left/right, 3 pickup/drop, 1 forward into a blocked cell), so the pre-set reading is **loop**; the same 50 seeds under sampling give 140/150 (not a gate, nothing frozen). H-7a summary §10. **Execution-mode audit (evaluation only, seeds 40000–40049):** under `STOCHASTIC` every selector and every primitive samples; 5 of 28 rows are mode-sensitive (≥ 0.10, deterministic lower in all five); 39 of 39 deterministic failures of frozen nodes are state repeats; the scripted UnlockPickup sequence scores 32/50 deterministic and 147/150 all-sampled (TRAINING `pickup_box`; not a gate); flat PPO on UnlockPickup is not available in either mode (no model saved). H-7a summary §12. `unlock_door` re-measured at `n_steps` 64: 32,869–34,877 own. `unlock_pickup` and the option-set baseline not run. No third specification (amendment rule); the decision is the advisor's. H-7b (BlockedUnlockPickup) not planned yet. See `plans/2026-10-02_H7_summary.md` |
| W1a | `pickup_box` spec 2 scored sampled: fixed budget, final models, 500 fresh seeds per row | — | ⛔ 2026-11-15 — **fails (6/10)**. Seeds 43–47 gave exactly 3 of 5, so the declared extension (48–52) ran; 6 of 10 pass, the rule needs 7. Own-hook row passes on all ten (sampled 0.918–0.994); the four failing seeds fail at the handoff (0.874–0.892). No run met 0004's argmax rule. Nothing frozen or registered. Next per the plan: a memory arm or an `epsilon` mode. See `plans/2026-11-15_W1a_summary.md` |
| W1a-D | Why some `pickup_box` models time out at the handoff (evaluation only) | — | ✅ 2026-11-15 — reading **Key**. Failing seeds pooled: unchanged 0.880, key removed 0.985, 240 steps 0.921; passing seeds 0.961 / 0.997 / 0.984 (Key holds for both). Counterfactuals, not results for `pickup_box`. See `plans/2026-11-15_W1a_diagnostic_summary.md` |
| W1a-S3 | `pickup_box` spec 3: half of training at the real handoff; W1a's test unchanged on fresh ranges | — | ✅ 2026-11-15 — **passes (7/10)** at the threshold, after the declared extension (3 of 5 first). Handoff 0.984–1.000 on all ten; failing seeds 44, 47, 50 on the own-hook row (0.898 / 0.778 / 0.896). Decision 0011 Accepted 2026-11-15; nothing frozen or registered. See `plans/2026-11-15_W1a_spec3_summary.md` |
| W1b-1 | Freeze one spec-3 `pickup_box` (decision 0014, `Proposed`), train `{2,2} unlock_pickup` under 0004 unchanged, argmax | — | ✅ 2026-11-15 — **passes, 3/3 seeds**: STABLE after 2 updates, fresh 54000–54499 486/500 each, held-out 49/50 (flat 0/50). One result in effect (identical selectors); random under the strict mask ties it. Child in the git-ignored `runs/w1b/`, nothing registered. See `plans/2026-11-15_W1b-1_summary.md` |
| H-7 | T6–T9b reuse and depth experiment | 8 | ⛔ **Stopped at H-7.3** (2026-10-02); superseded by H-7a above. H-7.0–H-7.2 done: v2 amended (Part A), pre-flight passed (V1–V7 reproduce; 0/500 unsolvable layouts; A1 `unlock_door` transfers to UnlockPickup at **0.98**), nine flat baselines launched. `{0,4} pickup_box` **not stable in 600,000 env steps on any seed** (held-out 0.28 / 0.40 / 0.24, still rising) — plan A2 is a stop. Flat PPO: Unlock STABLE at 350–450k; UnlockPickup and BlockedUnlockPickup **> 3,000,000, best SR 0.00** on all 3 seeds (censored). See `plans/2026-10-02_H7_summary.md` |
| H-6.9 | Why the choice selector did not learn | 1 | ✅ 2026-10-02 — task built as designed (400/400); the selector's network reads the door colour supervised at 0.940 from 1,000 examples and 0.997 from 4,000; between the pre-set thresholds, so no verdict. H-6 summary §10 |
| H-6.8 | Advisor corrections to H-6: fair choice run, distinct-policy evaluations, stale numbers | 3 | ✅ 2026-10-02 — v2 §5.3 gets an `eval_every` floor; composites move to `n_steps = 64`; choice task rerun at 100 updates (A5 fails, reported); `{0,2} open_door` re-measured at 0.83; regression 8 of 8. H-6 summary §9 |

### Track C — Continual learning / PN

| # | Task | SP | Status |
|---|---|---|---|
| C-1 | FourRooms diagnosis with plain PPO | 2 | ⬜ |
| C-2 | Forward transfer: laterals vs none | 5 | ✅ 2026-11-15 — W7: no measurable benefit on `go_to_goal` ← `open_door` (B 9/10, A and C 10/10 STABLE; medians 178,176 vs 190,464 vs 184,320); a reduction of 25 % or more would have shown 8 times in 10. [W7 summary](../plans/2026-11-15_W7_summary.md) |
| C-3 | Retention check after each new task | 2 | ⬜ |

### Track R — Recognition

| # | Task | SP | Status |
|---|---|---|---|
| R-1 | AE normalization, calibration, ROC-AUC; single vs per-task AE | 5 | ⬜ |
| R-2 | GRU dataset, accuracy vs N, old-task accuracy | 5 | 🔄 2026-11-15 (PR #18) — dataset and accuracy vs N done: 9 tasks, task-episode macro 63 / 83 / 85 / 85 / 85 % at N = 1 / 5 / 10 / 20 / 50 (3 seeds); old-task accuracy not measured |
| R-3 | Task registry: GRU task id → root node | 3 | ⬜ |

### Track G — Growth (after H-6)

| # | Task | SP | Status |
|---|---|---|---|
| G-1 | Lifecycle API: declare → train → stabilize → freeze → register | 5 | ⬜ |
| G-2 | Composite-first heuristic + ablation | 8 | ⬜ |
| G-3 | AE-triggered task declaration | 5 | ⬜ |

### Track UI

| # | Task | SP | Status |
|---|---|---|---|
| UI-1 | GUI on real runs: DAG, live stack, traces, node stats | 8 | ⬜ |
| UI-2 | Import external policy as a frozen primitive | 3 | ⬜ |

### Stretch

| # | Task | SP | Status |
|---|---|---|---|
| P-1 | Planning over initiation / goal predicates | 13 | ⬜ |

### Critical path

```
H-0 → H-1 → H-2 → H-3 → H-4 → H-5 → H-6 → H-7
```

In parallel, never blocking the critical path: `C-1, C-2, C-3`, `R-1, R-2, R-3`, `UI-1` (after H-1). Growth (`G-*`) starts after H-6.

### Story points

| SP | Meaning |
|----|---------|
| 1 | Trivial. Already know exactly how to do it. |
| 2 | Straightforward. Minor unknowns. |
| 3 | Moderate. A few unknowns. |
| 5 | Significant. Design decision + implementation together. |
| 8 | Complex. Multiple unknowns, requires iteration. |
| 13 | Very complex. Should probably be broken into subtasks. |
