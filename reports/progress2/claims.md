# Progress Report II — claims table

Every number the report uses, one row each. A number with no row does not enter the report.
The **Sentence** column is filled in at step 4 (the docx), with the section it appears in.

**Cut-off (named in the report text):** `main` @ `d7be0a2` (2026-11-15 20:22 +0300, after PRs 32
and 33), the single cut-off for every result. This supersedes the earlier two-commit cut-off
(`2bc131c` + branch `97238b3`) and the brief's "`unlock_pickup` not run" rule. W1b-1 numbers come
from `docs/plans/2026-11-15_W1b-1_summary.md` and its JSONs (section 16 below).

**Conventions.** `S` = sampled (the policy samples its action), `A` = argmax (deterministic).
S and A numbers never share a comparison or a table column. "Held-out" = 50 episodes on env
seeds 20000–20049, argmax, unless stated otherwise. Paths are relative to the repo root.
`(B)` = read from branch `97238b3`, not on `main`.

---

## 0. Repo state

| ID | Value | What it is | Source (file → field) | Sentence |
|---|---|---|---|---|
| R1 | `d7be0a2` | `main` commit read (single cut-off) | `git log -1 origin/main` | §2 intro, §2.4 |
| R2 | — | superseded: W1b-1 is merged into `main` (PR 32), so no branch commit is used | — | §2.4, Table 11 |
| R3 | 290 passed | test suite, `pytest tests/ -q`, run 2026-11-15 on `d7be0a2` in the worktree | pytest output | §2.3.10 |
| R4 | 13 | decision files (0001–0011, 0013, 0014); 0012 reserved, not written | `docs/kb/decisions/` | §2.2 |
| R5 | 11 | FROZEN nodes in the registry (9 + `pickup_box` spec 3 + `unlock_pickup`) | `runs/v2/*/meta.json` → `status == "frozen"` | §2.1, §2.3.1, Fig. 2 |
| R6 | cpu | device of every run JSON read here | every `…json` → `device` | §1.4, §4 intro |

## 1. Decision table (Section 2.2)

| ID | Decision | Status as in file | Source | Sentence |
|---|---|---|---|---|
| D01 | 0001 v1 → v2 redesign | Accepted | `docs/kb/decisions/0001-v1-to-v2-redesign.md` → `Status:` | Table 2 |
| D02 | 0002 DAG, not tree | Accepted by Emir; supervisor confirmation not recorded | `0002-dag-not-tree.md` | Table 2 |
| D03 | 0003 PN off the critical path | Accepted | `0003-…md` | Table 2 |
| D04 | 0004 stabilisation rule | Accepted (amended 2026-10-02) | `0004-…md` | Table 2 |
| D05 | 0005 fixed manual task order | Accepted as Phase A only | `0005-…md` | Table 2 |
| D06 | 0006 claim: compose and reuse | Accepted as the direction; evidence partial | `0006-…md` | Table 2 |
| D07 | 0007 automatic growth in scope | Accepted | `0007-…md` | Table 2 |
| D08 | 0008 graphify code graph | Accepted | `0008-…md` | Table 2 |
| D09 | 0009 `max_decisions` per frame | Accepted 2026-11-12, provided PR #15 keeps the design | `0009-…md` | Table 2 |
| D10 | 0010 decide empirically | Accepted | `0010-…md` | Table 2 |
| D11 | 0011 `pickup_box` spec 3 | Accepted (2026-11-15) | `0011-…md` | Table 2 |
| D13 | 0013 C-2 lateral source `open_door` | Accepted | `0013-…md` | Table 2 |
| D14 | 0014 freeze without live STABLE | Accepted (2026-11-15) | `0014-…md` | Table 2 |

## 2. Frozen library (Section 2.3.1, Table 2)

Held-out = `heldout.successes / n_episodes` (argmax). Cost = `steps_trained` (primitive) or
`env_steps_own` (composite) of the frozen checkpoint. Seed 42 for all.

| ID | Node | Level n | Held-out | Cost (env steps) | Wall s | Source | Sentence |
|---|---|---|---|---|---|---|---|
| L1 | `{0,1} get_key` | 0 | 50/50 = 1.00 | 180,000 | 173.2 | `runs/v2/0-1_get_key/meta.json` | Table 3, Fig. 2 |
| L2 | `{0,2} open_door` | 0 | 50/50 = 1.00 | 100,000 | 89.6 | `0-2_open_door/meta.json` | Table 3, Fig. 2 |
| L3 | `{0,3} go_to_goal` | 0 | 46/50 = 0.92 | 200,000 | 179.0 | `0-3_go_to_goal/meta.json` | Table 3, Fig. 2 |
| L4 | `{0,6} get_key_yellow` | 0 | 47/50 = 0.94 | 220,000 | 280.5 | `0-6_get_key_yellow/meta.json` | Table 3, Fig. 2 |
| L5 | `{0,7} get_key_blue` | 0 | 43/50 = 0.86 | 240,000 | 299.0 | `0-7_get_key_blue/meta.json` | Table 3, Fig. 2 |
| L6 | `{0,8} open_door_tk` | 0 | 47/50 = 0.94 | 100,000 | 127.7 | `0-8_open_door_tk/meta.json` | Table 3, Fig. 2 |
| L7 | `{0,9} drop_held` | 0 | 50/50 = 1.00 | 30,000 | 61.2 | `0-9_drop_held/meta.json` | Table 3, Fig. 2 |
| L8 | `{1,1} unlock_door` | 1 | 49/50 = 0.98 | 131,817 own (+280,000 children = 411,817) | 72.5 | `1-1_unlock_door/meta.json` → `env_steps_own`, `env_steps_children`, `env_steps_total` | Table 3, Fig. 2 |
| L9 | `{2,1} solve_doorkey` | 2 | 50/50 = 1.00 | 162,138 own (+611,817 children = 773,955) | 126.5 | `2-1_solve_doorkey/meta.json` | Table 3, Fig. 2 |
| L10 | stack depth 3 | — | root `solve_doorkey` → `unlock_door` → primitive | — | — | `2-1_solve_doorkey/meta.json` → `children`; `1-1_unlock_door/meta.json` → `children` | §2.1, §2.3.2 |
| L11 | not frozen: `{0,4} pickup_box` (spec 2, TRAINING), `{1,2} unlock_matching` (TRAINING) | — | 35/50 = 0.70; 39/50 = 0.78 | 1,001,472; 1,280,768 own | — | `0-4_pickup_box/meta.json`, `1-2_unlock_matching/meta.json` → `status` | Fig. 2 (39/50); §2.3.4 (0.78–0.80); §2.3.6 (0.70) |
| L12 | `drop_held` on 3 seeds | 0 | 1.00 / 1.00 / 0.98 | 30,000 each | — | `runs/v2/drop_held_s{42,43,44}.json` | Table 3 note |
| L13 | `unlock_door` transfers untouched into UnlockPickup | — | 49/50 = 0.98 | — | — | `runs/v2/h7_preflight.json` → `base.rows[0]` | §2.3.1, Table 1 |
| L14 | `open_door` in the two-key env / its replacement `open_door_tk` | — | 0.83 / 0.98 | — | — | results.md H-6.8 row; `choice_f29_open_door.json`, `choice_f29_open_door_tk.json` | §2.3.1 |
| L15 | stabilisation rule | — | SR ≥ 0.90 and Wilson 95 % lower ≥ 0.80 (binds at 46/50), twice, ≥ 1 policy update apart, 50 episodes on seeds 10000–10049 | — | — | `docs/kb/decisions/0004-…md`; every meta → `criterion` | §1.4, §2.4 |

## 3. `solve_doorkey` cost (Section 2.3.2, Table 3)

| ID | Value | What it counts | Source | Sentence |
|---|---|---|---|---|
| K1 | 1.00 (50/50) on 3 seeds | held-out, argmax, H-6.8 runs | `runs/v2/solve_doorkey_h68_s{42,43,44}.json` → `composite.heldout` | §2.3.2 |
| K2 | 48,887 – 55,659 | own env steps, H-6.8 re-measurement (`n_steps` 64) — status.md's "~53k own" | same → `composite.env_steps_own` (43: 48,887; 44: 53,930; 42: 55,659) | Table 4, §4.1 |
| K3 | 660,704 – 667,476 | own + children as built (611,817) — status.md's "~664k with children" | same → `composite.env_steps_total` | Table 4 |
| K4 | 1,536 decisions | decisions to STABLE, all 3 seeds | same → `composite.decisions_to_stable` | §2.3.2 |
| K5 | 162,138 own / 773,955 total | the frozen checkpoint as built (`n_steps` 256) | `h7_amortisation.json` → `as_built[1]`; `2-1_solve_doorkey/meta.json` | Table 4, §4.5 |
| K6 | 561,756 – 570,536 | cumulative through DoorKey with composites re-measured at `n_steps` 64 (Unlock row 312,869–314,877 + DoorKey row 248,887–255,659) | `h7_amortisation.json` → `rows[1].hierarchy_cumulative`, `rows[0]` | Table 4 |
| K7 | 300,000 – 450,000 | flat PPO on DoorKey-6x6, steps to STABLE (44: 300k, 43: 350k, 42: 450k) | `runs/v2/flat_ppo_s{42,43,44}.json` → `report.steps_trained` | Table 4, Table 6 |
| K8 | 650,000 – 900,000 | flat cumulative through DoorKey (flat Unlock + flat DoorKey) | `h7_amortisation.json` → `rows[1].flat_cumulative` | Table 4, §4.5 |
| K9 | 350,000 – 450,000 | flat PPO on Unlock to STABLE | `flat_unlock_ppo_s*.json` → `report.steps_trained` | Table 4, Table 6 |
| K10 | 312,869 – 314,877 | hierarchy through Unlock (`get_key` + `open_door` + `unlock_door` re-measured 32,869–34,877) | `h7_amortisation.json` → `rows[0]` | §4.1 (unlock_door 32,869–34,877) |
| K11 | 64,000 – 80,951 own; 544,000 – 560,951 total | flat option set `{1,B}` (same 3 skills, depth 1) | `runs/v2/flat_options_h68_s*.json` | Table 4 |
| K12 | 5 caveats | amortisation caveats (shared primitive training, one-seed primitives, eval not counted, repeated runs not counted, hand-written subgoals/hooks) | `h7_amortisation.json` → `caveats` | Table 4 note |

## 4. Composition vs the mask (Section 2.3.3, new small table)

All argmax, held-out 50 episodes. "Strict" = initiation mask on (only the next child in the
subgoal order is selectable). "Goal-only" = the project's default mask.

| ID | Row | `unlock_door` (H-5, depth 1) SR / mean env steps | `solve_doorkey` (H-6, depth 2) SR / mean env steps, 3 seeds | Source | Sentence |
|---|---|---|---|---|---|
| M1 | learned, goal-only | 0.98 / 19.16 | 1.00 / 14.24 | `c1_controls.json` → `controls[3]`; `solve_doorkey_s*.json` → `arms.learned` | Table 5 |
| M2 | random over available children, goal-only | 0.90 / 85.34 | 0.82 – 0.92 / 113.3 – 143.1 | `controls[1]`; `arms.random_over_available` | Table 5 |
| M3 | random over all children (mask clipped) | 0.94 / 82.96 | 0.86 – 0.92 / 112.6 – 134.6 | `controls[0]`; `arms.random_over_all` | Table 5 |
| M4 | random, strict mask | 0.98 / 19.16 | 1.00 / 14.24 | `controls[2]`; `arms.random_strict_mask` | Table 5 |
| M5 | learned, strict mask | 0.98 / 19.16 | 1.00 / 14.24 | `controls[4]`; `arms.learned_strict_mask` | Table 5 |
| M6 | step efficiency, learned vs random-over-available | 85.34 / 19.16 = 4.4× | 113–143 / 14.2 ≈ 8× | derived from M1, M2 | §2.3.3 ("4.4 times", "8 to 10 times": 113.3/14.24 – 143.1/14.24) |
| M7 | divergence count: withheld choices taken | — | 0 of 20 (40 root decisions, 20 episodes) | `solve_doorkey_s*.json` → `mask_divergence` | §2.3.3 |
| M8 | random selector before training (`solve_doorkey` first evaluation) | — | 0.40 / 0.10 / 0.78 (seeds 42/43/44) | H6 summary §3 item 3; `solve_doorkey_h68_s42.json` → `criterion.history[0]` = 0.40 | §2.3.3 |
| M9 | strict-mask trained `solve_doorkey`: STABLE at the first allowed point | — | 1,024 decisions, 7,240–7,280 own env steps | `solve_doorkey_h68_s42.json` → `trained_with_strict_mask` (7,280); H6 summary §3 | §2.3.3 |

## 5. Choice task (Section 2.3.4)

| ID | Value | What it is | Source | Sentence |
|---|---|---|---|---|
| C1 | 100 policy updates, 3 seeds × 2 masks | budget of the choice runs | results.md H-6/H-6.8 row; `unlock_matching_s*.json` | §2.3.4 |
| C2 | not STABLE; held-out 0.78 / 0.80 / 0.80 | `unlock_matching`, seeds 42/43/44 | `unlock_matching_s*.json` → `heldout` | §2.3.4 |
| C3 | 43/50 | `get_key_<matching>` handed the decoy key swaps to the matching key (a wrong pick is recoverable) | `choice_full_hand.json` → `rows[0].swapped_to_the_matching_key` | §2.3.4 |
| C4 | 0.72 | best first-key accuracy over 7 strict-mask runs | `choice_conditioning.json` → `rows[*].first_key_accuracy` (max) | §2.3.4 |
| C5 | 3 of 7; gaps +0.43 / +0.35 / +0.32, Fisher p 0.004 / 0.021 / 0.021, uncorrected | runs with a weak colour dependence of the first key | `choice_conditioning.json` → `rows[1,5,6]` | §2.3.4 |
| C6 | 0.940 (1,000 examples), 0.997 (4,000) | the selector's network reads the door colour when trained supervised | results.md H-6.9 row; `choice_diagnosis.json` | §2.3.4 |
| C7 | 1,280,768 / 1,283,740 / 1,248,538 | own env steps of the 3 `unlock_matching` runs | `unlock_matching_s*.json` → `env_steps_own` | not used |

## 6. Flat PPO on UnlockPickup and Blocked (Section 2.3.5, Table 4)

| ID | Value | Source | Sentence |
|---|---|---|---|
| F1 | 0/50 held-out, best stopping SR 0.00, after 3,000,320 env steps, seeds 42–44, not STABLE | `flat_unlockpickup_ppo_s*.json` → `report.heldout`, `report.steps_trained`, `report.criterion.history` | §2.3.5, Table 6 |
| F2 | 0/50 held-out, best stopping SR 0.00, after 3,000,320, seeds 42–44 | `flat_blocked_ppo_s*.json` | §2.3.5, Table 6 |
| F3 | 9,000,960 | three UnlockPickup flat runs, total (censored) | `h7_amortisation.json` → `including_withdrawn_spec.totals_over_all_runs.flat_ppo_unlockpickup_three_runs` | §2.3.5, §4.5 |
| F4 | 3,000,000 budget, `eval_every` 100,000, timeout 288 | flat UnlockPickup config | `flat_unlockpickup_ppo_s42.json` → `budget`, `eval_every`, `report.timeout` | Table 6 |
| F5 | no flat model saved | flat UnlockPickup cannot be re-scored on fresh seeds | W1b-1 plan "What this plan does not settle"; status.md H-7a row | §2.3.5 |
| F6 | > 1,593,228 – 1,602,008 vs > 3,650,320 – 3,900,320 | cumulative through UnlockPickup, hierarchy vs flat, both lower bounds | `h7_amortisation.json` → `rows[2]` | not used |

## 7. `pickup_box` (Section 2.3.6, Fig. 3 and Table 5)

### 7a. History (argmax held-out, 50 episodes)

| ID | Value | What it is | Source | Sentence |
|---|---|---|---|---|
| P1 | 0.28 / 0.40 / 0.24, not STABLE in 600,064 | spec 1 (H-7), seeds 42/43/44 | `pickup_box_attempt1_s*.json` → `heldout`, `steps_trained` | §2.3.6 |
| P2 | 0.70 / 0.66 / 0.64, not STABLE in 1,001,472 | spec 2 (H-7a, October), seeds 42/43/44 | `pickup_box_s{42,43,44}.json` | §2.3.6 |
| P3 | 20/20 argmax failures are state repeats; 140/150 sampled | spec 2 seed 42 at the real handoff (H-7a.5) | results.md H-7a.5 row; `pickup_box_spec2_trace.json` | §2.3.6 |
| P4 | 32/50 argmax, 147/150 all sampled | scripted UnlockPickup sequence, `pickup_box` TRAINING (execution-mode audit) | `execution_mode_audit.json` → `E2_scripted_sequence.variants[0].deterministic`, `variants[2].stochastic` | §3.2 |
| P5 | 39/39 | deterministic failures of frozen nodes that are state repeats (audit) | results.md 2026-10-02 audit row | §3.2 |

### 7b. W1a (spec 2, sampled test). Rule: a row passes at S ≥ 0.90 with ≥ 475/500 valid; a seed passes on both rows; ≥ 7 of 10.

| ID | Value | Source | Sentence |
|---|---|---|---|
| W1 | fails, 6/10 seeds (3 of 5 first → declared extension 48–52) | `w1a_pickup_box_sampled.json` → `verdict`; `w1a_pickup_box_sampled_ext.json` → `verdict.seeds_passing` = 6 | §2.3.6, §2.4 |
| W2 | own hook S 0.918 – 0.994 (passes on all ten) | `…models.<seed>.rows.own_hook.sampled.success_rate` | §2.3.6 |
| W3 | handoff S, failing seeds 46/47/50/52: 0.874 / 0.876 / 0.878 / 0.892 | `…rows.handoff.sampled.success_rate` | §2.3.6 (range 0.874–0.892) |
| W4 | handoff S, passing seeds: 0.918 – 0.982 | same | Fig. 3 |
| W5 | handoff chain valid in 491/500 | `…rows.handoff.sampled.valid` | not used |
| W6 | own hook seeds 50000–50499, handoff 51000–51499, sampling seed 0 | `own_hook_seeds`, `handoff_seeds`, `sampling_seed` | Fig. 3 caption |
| W7a | A, separately: own hook 0.684 – 0.766, handoff 0.379 – 0.713; every argmax failure a state repeat | `…argmax.success_rate`, `failures_with_state_repeat == failures` | Table 7 |
| W8 | fixed budget 1,001,472 per run; 10 runs = 10,014,720 | `pickup_box_w1a_s*.json` → `steps_trained`; W1a summary "Cost" | §2.4 |

Per-seed W1a values (Fig. 3, left panel; S only):

| Seed | 43 | 44 | 45 | 46 | 47 | 48 | 49 | 50 | 51 | 52 |
|---|---|---|---|---|---|---|---|---|---|---|
| own hook S | 0.994 | 0.980 | 0.972 | 0.918 | 0.964 | 0.970 | 0.968 | 0.956 | 0.946 | 0.948 |
| handoff S | 0.982 | 0.919 | 0.972 | 0.874 | 0.876 | 0.957 | 0.965 | 0.878 | 0.970 | 0.892 |
| passes | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✗ |

### 7c. Handoff diagnostic (evaluation only; K and T are counterfactuals, not results)

| ID | Value | Source | Sentence |
|---|---|---|---|
| G1 | reading **Key** | `w1a_handoff_diagnostic.json` → `readings.failing.reading.reading` | §2.3.6, §2.4 |
| G2 | failing seeds pooled: unchanged 1728/1964 = 0.880, K (key removed) 0.985, T (240 steps) 0.921 | `readings.failing.{unchanged,K,T}` | §2.3.6 |
| G3 | passing seeds pooled: 0.961 / K 0.997 / T 0.984 | `readings.passing` | not used |
| G4 | rule: Key if K ≥ 0.95; Time if T ≥ 0.95 | `readings.rules` | §2.3.6 ("reading rule fixed in advance") |
| G5 | key never orthogonally adjacent to the door (0/491); chain uses 16.8 env steps on average (max 32) | `w1a_handoff_key_position.json`; results.md row | not used |

### 7d. Spec 3 (half of training at the real handoff; W1a's test unchanged on fresh ranges)

| ID | Value | Source | Sentence |
|---|---|---|---|
| S1 | passes, 7/10 at the threshold (3 of 5 first → declared extension) | `w1a_spec3_sampled.json` → `verdict`; `w1a_spec3_sampled_ext.json` → `verdict.seeds_passing` = 7 | §2.3.6, §2.4 |
| S2 | handoff S 0.984 – 1.000 on all ten (488/500 valid) | `…models.<seed>.test.handoff.sampled` | §2.3.6 |
| S3 | failing seeds 44 / 47 / 50 fail on own hook: S 0.898 / 0.778 / 0.896; all ten 0.778 – 0.968 | `…test.own_hook.sampled.success_rate` | §2.3.6 |
| S4 | test ranges: own hook 52000–52499, handoff 53000–53499 | `test_ranges` | Fig. 3 caption |
| S5 | A, separately: handoff 0.902 – 1.000, own hook 0.586 – 0.718; every argmax failure a state repeat | `…test.*.argmax` | Table 7, §2.3.6 |
| S6 | 10 × 1,001,472 = 10,014,720 env steps; chain steps in training resets 3,715,701 (own line) | `pickup_box_s3_s*.json` → `steps_trained`; spec-3 summary "Cost" | §4.5 (inside P7) |
| S7 | spec-2 models reproduce W1a episode for episode (gate) | `w1a_spec3_sampled.json` → `gate_spec2_reproduces_w1a.*.passed` | not used |
| S8 | spec hash `6c6428043d97ac75` | `spec3_hash` | not used |

Per-seed spec-3 values (Fig. 3, right panel; S only):

| Seed | 43 | 44 | 45 | 46 | 47 | 48 | 49 | 50 | 51 | 52 |
|---|---|---|---|---|---|---|---|---|---|---|
| own hook S | 0.912 | 0.898 | 0.918 | 0.958 | 0.778 | 0.964 | 0.950 | 0.896 | 0.968 | 0.954 |
| handoff S | 0.988 | 1.000 | 1.000 | 0.996 | 0.984 | 1.000 | 1.000 | 0.998 | 1.000 | 1.000 |
| passes | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ |

Argmax table (Table 5, separate from Fig. 3):

| Seed | 43 | 44 | 45 | 46 | 47 | 48 | 49 | 50 | 51 | 52 |
|---|---|---|---|---|---|---|---|---|---|---|
| W1a own hook A | 0.706 | 0.766 | 0.754 | 0.704 | 0.742 | 0.748 | 0.748 | 0.752 | 0.684 | 0.704 |
| W1a handoff A | 0.713 | 0.560 | 0.654 | 0.379 | 0.576 | 0.652 | 0.703 | 0.491 | 0.664 | 0.654 |
| spec 3 own hook A | 0.656 | 0.640 | 0.610 | 0.614 | 0.586 | 0.718 | 0.678 | 0.606 | 0.658 | 0.704 |
| spec 3 handoff A | 0.951 | 0.996 | 0.994 | 1.000 | 0.902 | 0.980 | 1.000 | 0.961 | 0.994 | 0.998 |

(W1a and spec 3 are on different env-seed ranges: W1a 50000–51499, spec 3 52000–53499.)

### 7e. Research cost of `pickup_box`

| ID | Value | Source | Sentence |
|---|---|---|---|
| P6 | 4,804,608 | six October runs: spec 1 3 × 600,064 = 1,800,192 + spec 2 3 × 1,001,472 = 3,004,416 | `h7_amortisation.json` → `including_withdrawn_spec.totals_over_all_runs.pickup_box_six_runs`; W1b-1 plan cost table | §4.5 (inside P7) |
| P7 | 28,549,749 ≈ 28.5M | research total = P6 + W1a 10,014,720 + spec 3 10,014,720 + chain 3,715,701 | W1b-1 plan (branch) "Cost reporting"; sum checked | §3.2, §4.5 |
| P8 | 515,000 | chain steps inside spec-3 stopping evaluations (not in P7; evaluation is counted on neither side) | spec-3 summary "Cost" | not used |

## 8. W1b-1 (Section 2.4; state from branch `97238b3`)

| ID | Value | What it is | Source | Sentence |
|---|---|---|---|---|
| B1 | done | step 1: `pickup_box` spec 3, training seed 45, frozen from its fixed-budget model under 0014 (Proposed); `met_0004: false`; checkpoint git-ignored | (B) `runs/v2/w1b_pickup_box_freeze.json` → `seed`, `meta.met_0004`, `meta.frozen_under`, `checkpoint` | Table 11 |
| B2 | freezing rule: (a) W1a test passed and (b) Wilson 95 % lower bound of A handoff ≥ 0.95 on ≥ 475 valid; smallest seed meeting both | rule | (B) `…freeze.json` → `gate_p2_rule.rule` | Table 11 |
| B3 | seeds meeting the rule: 45, 46, 48, 49, 51, 52 (6 of 10) | | (B) `gate_p2_rule.seeds.*.meets_rule`; branch W1b-1 summary | Table 11 |
| B4 | seed 45: S own hook 459/500, S handoff 488/488 (rule a) | | (B) `meta.acceptance.a_w1a_test_sampled` | Table 11 |
| B5 | seed 45: A handoff 485/488, Wilson lower 0.982 (rule b) | | (B) `meta.acceptance.b_argmax_handoff` | Table 11 |
| B6 | seed 45: A own hook 305/500 = 0.610, all 195 failures state repeats (known limit) | | (B) `meta.acceptance.argmax_own_hook_known_limit` | Table 11 |
| B7 | done | step 2: `unlock_pickup` script and **ceiling gate (a pre-check, not a result for `unlock_pickup`)** | (B) `runs/v2/w1b_unlock_pickup_gate.json` | Table 11 |
| B8 | 48/50 stopping seeds (10000–10049); 49/50 held-out (20000–20049) | ceiling gate: scripted `unlock_door` → `drop_held` → `pickup_box`, argmax. **Label in text: "ceiling gate, a pre-check, not a result for unlock_pickup"** | (B) `…gate.json` → `stopping_10000_10049.successes`, `heldout_20000_20049.successes` | Table 11 |
| B9 | 46/50 | gate threshold on the stopping set (below → fail, reading "chain") | branch plan, step 2 | Table 11 |
| B10 | ≈ 0.97 | expected chain ceiling (488/500 × 485/488) | (B) `…gate.json` → `chain_ceiling.value` | Table 11 |
| B11 | done (superseded by N4–N6) | step 3: `unlock_pickup` trained on seeds 43, 44, 45 | `main` @ `d7be0a2`, W1b-1 summary | Table 11, §2.4 |
| B12 | budget 40,960 decisions, `eval_every` 512, 8 envs, `n_steps` 64, timeout 288, `max_decisions` 8, goal-only mask | design | branch plan "The parent" | Table 11 |
| B13 | a seed passes: STABLE under 0004 within budget **and** fresh row A ≥ 0.90 (450/500) on 54000–54499; W1b-1 passes at ≥ 2 of 3 seeds; no extension | pass rule | branch plan "Criterion" | Table 11 |
| B14 | addendum before step 3: random-over-available control on 54000–54499, not in the verdict, reported as step efficiency | | branch plan "Addendum before step 3" (commit `97238b3`) | Table 11 |
| B15 | fail readings: gate < 46/50 → chain; gate passed and ≥ 2 seeds > 5/50 below gate → selector; else undetermined | | branch plan "What follows" | Table 11 |
| B16 | recipe cost ≈ 2,319,000 + own (with 10/6 selection factor); 1,403,535 + own without | cost framing fixed before the run (no own value yet) | branch plan "Cost reporting" | not used |

## 9. Lateral ablation W7 (Section 2.3.7)

`go_to_goal`, 10 training seeds per arm (1000–1900), budget 401,408, `eval_every` 4,096, one torch thread.

| ID | Arm | STABLE | Median steps to STABLE | Min–max | Final score median (A, 60000–60499) | Source | Sentence |
|---|---|---|---|---|---|---|---|
| V1 | A no lateral | 10/10 | 190,464 | 143,360 – 221,184 | 0.901 | `runs/v2/w7/w7_lateral.json` → `arms.A` | Table 9 |
| V2 | B lateral from frozen `open_door` | 9/10 (seed 1200 censored) | 178,176 | 155,648 – 401,408 (censored) | 0.875 | `arms.B` | Table 9 |
| V3 | C lateral from untrained network | 10/10 | 184,320 | 147,456 – 196,608 | 0.919 | `arms.C` | Table 9 |
| V4 | B vs A: p 0.469, median ratio 0.977; B vs C: p 0.354; A vs B: p 0.559 (rule p < 0.05 and ratio ≤ 0.80) | — | — | — | — | `tests.*` | §2.3.8 |
| V5 | detectable effect: a true 25 % reduction passes 0.80 of the time (20 %: 0.47; 30 %: 0.955); sd of log steps (A) 0.143 | — | — | — | — | `power.pass_rate_by_reduction`, `power.sd_log` | §2.3.8 |
| V6 | 15 of 30 final scores below 0.90 | — | — | — | — | results.md W7 row; `arms.*.final_score.per_seed` | Table 9 note |
| V7 | 5,656,576 env steps for 30 runs | — | — | — | — | `cost.training_env_steps_30_runs` | Table 9 note |
| V8 | Fig. 4 PNG is 1634 × 624 px | legibility check at page width | — | — | — | file | Fig. 4 dropped (illegible at page width); W7 given as Table 9 |

## 10. GRU (Section 2.3.8, Table 6)

Task-episode macro accuracy, mean over seeds 42–44, 9 tasks, CPU.

| ID | N | Mean | Per seed (42/43/44) | Source | Sentence |
|---|---|---|---|---|---|
| U1 | 1 | 63.3 % | 62.9 / 63.7 / 63.2 | `runs/v2/gru/task_seq1_s*_metrics.json` → `task_episode_macro_accuracy` | Table 10, §2.3.9 |
| U2 | 5 | 83.2 % | 83.0 / 83.2 / 83.5 | `task_seq5_s*` | Table 10, §2.3.9 |
| U3 | 10 | 84.9 % | 84.1 / 85.6 / 85.0 | `task_seq10_s*` | Table 10, §2.3.9 |
| U4 | 20 | 85.3 % | 86.1 / 85.5 / 84.3 | `task_seq20_s*` | Table 10, §2.3.9 |
| U5 | 50 | 85.1 % | 85.6 / 84.8 / 85.0 | `task_seq50_s*` | Table 10, §2.3.9 |
| U6 | 825 test episodes (70/15/15 split, 3,850 / 825 / 825) | | | `test_episodes`; GRU validation summary | §2.3.9 |
| U7 | uniform guess 11 % (1/9), stated as not the right baseline (start states differ per task; R-0 not done) | | | results.md R-2 row; roadmap W6 R-0 | §2.3.9 |
| U8 | weakest: `unlock` 70 % at N = 20; `get_key` lowest at N = 50, 66.1 % (0.746 / 0.681 / 0.557) | | | results.md R-2 row; `per_task_episode_balanced_accuracy.unlock` (s42: 0.699) | §2.3.9 |
| U9 | not measured: old-task accuracy, switching latency | | | status.md Track R | §2.3.9 |

## 11. Evaluation infrastructure (Section 2.3.9)

| ID | Value | Source | Sentence |
|---|---|---|---|
| E1 | 9 frozen nodes, 450 held-out episodes identical before/after the parity change | results.md 2026-11-05 PR #15 row | §2.3.10 |
| E2 | 18 paired old/new training runs; buffers and weights identical at every completed update | results.md PR #16 row | §2.3.10 |
| E3 | 290 tests pass | R3 | §2.3.10 |
| E4 | every training attempt archived under `runs/training/`; no registry overwrite; freeze needs STABLE (PR #17) | status.md Track H | §2.3.10 |

## 12. Compute and energy (Section 4, Table 9)

`wall_seconds` per process, summed over the training-run JSONs in `runs/v2/` listed below. Runs were
often launched in parallel batches (e.g. five at a time), so these are **process-hours, not
machine-hours**. Not counted: smoke runs, bug-repeated runs, runs only in `runs/training/`,
evaluation-only scripts (except where listed). A lower bound on compute since PR-I.

| ID | Group | Files | Wall s | Hours | Sentence |
|---|---|---|---|---|---|
| X1 | library primitives + `unlock_door` as built (`solve_doorkey` seed 42 is in X6) | 7 metas (`get_key`…`open_door_tk`, `unlock_door`) | 1,221.5 | 0.34 | Table 13 |
| X2 | `pickup_box` research (spec 1 ×3, spec 2 Oct ×3, W1a ×10, spec 3 ×10) | `pickup_box_attempt1_s*`, `pickup_box_s4?`, `pickup_box_w1a_s*`, `pickup_box_s3_s*` | 31,041.6 | 8.62 | Table 13, §4.3 |
| X3 | flat PPO ×12 (DoorKey, Unlock, UnlockPickup, Blocked) | `flat_*ppo_s*.json` | 35,410.8 | 9.84 | Table 13 |
| X4 | of which flat UnlockPickup + Blocked ×6 | `flat_unlockpickup_ppo_s*`, `flat_blocked_ppo_s*` | 30,451.3 | 8.46 | Table 13, §4.3 |
| X5 | W7 ×30 | `runs/v2/w7/runs/*.json` | 8,881.6 | 2.47 | Table 13 |
| X6 | other (drop_held ×3, solve_doorkey ×6, unlock_door H-7a.3 ×3, flat options ×3, choice ×3 + 4 arms) | listed files | 16,222.1 | 4.51 | Table 13 |
| X7 | **total** | X1 + X2 + X3 + X5 + X6 | 92,777.6 | **25.8** | Table 13, §4.6 ("about 26") |
| X8 | `solve_doorkey` stack as built: `get_key` + `open_door` + `go_to_goal` + `unlock_door` + `solve_doorkey` | 5 metas | 640.8 | 0.18 | not used |
| X9 | W1a + spec-3 scoring (evaluation) | `w1a_*sampled*.json` → `wall_seconds` | 4,422.1 | 1.23 | not used |
| X10 | energy at an **assumed 90 W** average draw per process (confirmed by Emir): 25.8 h × 90 W ≈ **2.3 kWh** (2.32) | assumption, labelled | — | — | Table 13 note |
| X11 | PR-I / Proposal estimate: 10–30 GPU-hours, 1.5–4.5 kWh, 0.4 kg CO₂/kWh, $3–$15 per run | PR-I §4.4 | — | — | Table 13 note |
| X12 | 2.32 kWh × 0.4 kg CO₂/kWh ≈ 0.9 kg CO₂e (same grid factor as PR-I) | derived from X10, X11 | — | — | Table 13 note |

## 13. Sustainability comparison (Section 4.5)

| ID | Value | Source | Sentence |
|---|---|---|---|
| Z1 | reuse through DoorKey: 773,955 as built vs flat 650,000 – 900,000 → about the same | K5, K8 | §4.5 |
| Z2 | `pickup_box` research ≈ 28.5M vs flat UnlockPickup 9.0M, on a task flat never solved | P7, F3 | §4.5 |
| Z3 | no efficiency gain claimed yet | decision 0006; roadmap B3 | §4.5 |

## 14. Timetable and roadmap (Sections 3.3, 5)

| ID | Value | Source | Sentence |
|---|---|---|---|
| T1 | PR-I §5.8: FourRooms W14; AE calibration W14; BWT/FWT logging W15; lateral baseline W15; MetaController upgrade W16; GRU baseline + AE–GRU handoff W16; dynamic options W17; dashboard W17; Phase 2 integration W19; no-AE/no-GRU ablations W21; Phase 3 scope review W22 (11 rows) | PR-I docx §5.8 table | Table 12 |
| T2 | this report week 7, Draft Final Report week 12, Final Report week 14 (confirmed by Emir) | brief | §5, Table 16 |
| T3 | PR-I date May 24th, 2026; this report November 15th, 2026 | PR-I cover; brief | cover; Table 12 intro |


## 15. Numbers added while writing (step 4)

| ID | Value | Source | Sentence |
|---|---|---|---|
| A1 | ≈ 1.2 million (1,201,817) env steps for the seven primitives and `unlock_door` | sum of L1–L8 own costs | §2.3.1 |
| A2 | depth-3 stack within two days of the redesign (H-0 on 2026-10-01, H-6 `solve_doorkey` FROZEN 2026-10-02) | `docs/kb/status.md` roadmap rows H-0…H-6 | §3.1 |
| A3 | GRU: nine tasks = seven primitive skills + two root tasks | roadmap §2 W6 | §2.3.9 |
| A4 | "six months" since PR-I (24 May → 15 November 2026) | PR-I cover; this report | §4.6 |
| A5 | 450 held-out episodes identical (9 nodes × 50) | E1 | §2.3.10 |
| A6 | text width 6.3 in; figures placed at 6.3 in | PR-I `sectPr` (A4, 1417-twip margins) | figure layout |


## 16. W1b-1 and the 11-node library (added with the `d7be0a2` update)

| ID | Value | Source (file → field) | Sentence |
|---|---|---|---|
| N1 | 11 frozen nodes | `runs/v2/*/meta.json` → `status == "frozen"`; `status.md` Track H | §1.3, §1.4, §2.1, §2.3.1, Fig. 2, Table 3 |
| N2 | `{0,4} pickup_box` spec 3, training seed 45, FROZEN under 0014; handoff argmax 485/488 (Wilson lower 0.982), own hook argmax 305/500; 1,001,472 steps + 372,063 chain steps | `runs/v2/0-4_pickup_box/meta.json` → `acceptance`, `steps_trained`, `chain_env_steps_in_training_resets`; `w1b_pickup_box_freeze.json` | Table 3, Fig. 2, §2.3.7 |
| N3 | `{2,2} unlock_pickup` seed 43 registered: held-out 49/50, own 60,205, children 1,443,289 | `runs/v2/2-2_unlock_pickup/meta.json` → `heldout`, `env_steps_own`, `env_steps_children`; `w1b_register.json` | Table 3, Fig. 2, §2.3.7 |
| N4 | STABLE on 3 of 3 seeds after 1,536 decisions (2 policy updates) | `w1b_unlock_pickup_s{43,44,45}.json` → `decisions_to_stable`; W1b-1 summary | §2.3.7, Table 8 |
| N5 | fresh 486/500 = 0.972, Wilson 95 % 0.9536–0.9832 (0.954–0.983), mean 31.05 steps, each seed | `w1b_unlock_pickup_s*_fresh.json` → `fresh` | §2.3.7, Table 8, §2.4 |
| N6 | held-out 49/50 each seed | `w1b_unlock_pickup_s*.json` → `arms.learned`; meta `heldout` | §2.3.7, Table 8 |
| N7 | the three selectors act identically (same held-out and fresh episodes) | W1b-1 summary, "The three seeds are one result" | §2.3.7, §2.4 |
| N8 | random under the strict mask 49/50 (ties learned) | `w1b_unlock_pickup_s*.json` → `arms.random_strict_mask` | §2.3.7, §3.2 |
| N9 | random over available on fresh seeds: 359 / 375 / 359 of 500; 148.4 / 139.4 / 146.8 mean steps ("139–148") | `…_fresh.json` → `random_over_available_control` | §2.3.7, Table 8 |
| N10 | 2.8 % fresh loss is in the chain before the handoff; gate: every episode reaching the handoff succeeded (48/48, 49/49) | W1b-1 summary; `w1b_unlock_pickup_gate.json` | §2.3.7, §2.4 |
| N11 | own env steps 55,412–60,205 | `w1b_unlock_pickup_s*.json` → `env_steps_own` | Table 8 note, §4.1, Table 14 |
| N12 | recipe 2,374,637–2,379,430 (2.37–2.38M); without factor 1,458,947–1,463,740; reused library 411,817; with library ≈ 2.79M (2,786,454–2,791,247) | W1b-1 summary "Cost"; sum checked | Table 8, §2.3.7, §4.5, Table 14, Table 1 (B3) |
| N13 | 10/6 factor: 6 of 10 spec-3 runs met the freezing rule | B3; W1b-1 summary | Table 8 note, Table 14 |
| N14 | predicted chain ceiling 0.970 (488/500 × 485/488) against fresh 0.972 | `w1b_unlock_pickup_gate.json` → `chain_ceiling.value`; N5 | §2.4 (second example, step 6) |
| N15 | 0014 Accepted 2026-11-15; child and seed-43 composite registered after it | `0014-…md` → `Status`; `w1b_register.json` | Table 2, §2.4 |
| N16 | `unlock_door` has two parents (`solve_doorkey`, `unlock_pickup`) | meta `children` of both | §2.1, Table 1 (B1, B2) |
| N17 | flat UnlockPickup 0/50 on the same held-out seeds, > 3,000,320 per run, censored | F1 | §2.3.7, Table 8, Table 14 |
| N18 | `unlock_pickup` wall 46.4 / 42.9 / 42.6 s → "other" group 25 runs, 4.54 h; total still 25.8 h (92,909.5 s) | `w1b_unlock_pickup_s*.json` → `wall_seconds` | Table 13 |
| N19 | W2 first: T9a `{2,3}` (`move_blocker`, `unlock_door`, `pickup_box`) vs T9b `{3,1}` (`move_blocker`, `unlock_pickup`); `pickup_box` needs its own handoff row under T9a | v2 §11 table; `status.md` Track H "Next" | Table 15, Table 16 |

### 17. Second review round (15 November, evening)

| # | Sentence in the report | Source | Where |
|---|---|---|---|
| P1 | `pickup_box` seed 45 held-out 41/50 (argmax, mixed hook); 485/488 is on the selection range 53000–53499, so a selection figure | `w1a_spec3_sampled.json` → `models.45.load_check_heldout_mixed_hook.recorded` = [41, 27.18]; `meta.acceptance.ranges.handoff` | Table 3, Fig. 2 |
| P2 | Choice task: 3 seeds × 2 masks; then 4 strict-mask runs: arm (a) s42 repeats H-6.8 s42 exactly, arm (b) s42–44 at `ent_coef` 0.05; six distinct strict runs; weak dependence p 0.0037 (0.01), 0.021, 0.0214 (0.05) | `choice_conditioning.json` → `rows` | §2.3.4 |
| P3 | Key state "rarely produced in training; how rarely was not measured" | decision 0011 lines 50–52; W1a spec-3 plan lines 27–28 | §2.3.6 |
| P4 | Handoff rows: 491 valid (W1a), 488 valid (spec 3) of 500 | W1b-1 plan line 60; spec-3 summary line 37 | §2.3.6, Fig. 3 |
| P5 | 2.8 % loss before the handoff stated as an inference (fresh JSON has no handoff-reached field) | `w1b_unlock_pickup_s*_fresh.json` (no such field); gate 48/48, 49/49 | §2.3.7, §2.4 |
| P6 | Cost: DoorKey alone not cheaper (660,704–667,476 vs 300,000–450,000); Unlock + DoorKey about the same (561,756–570,536 re-measured, 773,955 as built, vs 650,000–900,000) | Table 4 rows (see section 3) | §2.3.2, §3.1, §4.5 |
| P7 | Flat PPO evaluated every 50,000 steps (UnlockPickup 100,000) | `scripts/train_flat_baseline.py` `TASKS` | Table 4 note |
| P8 | No commits June–September 2026 (summer break) | `git log --all` by month: 2026-05 20, 2026-10 58 | §2.1, §3.1 |
| P9 | GRU rollouts deterministic, env seeds 20000–20499 | `docs/plans/2026-11-13_gru_task_identification_validation.md` line 11 | §2.3.9 |
| P10 | 0009: "Accepted 12 November, provided the final pull request kept the design; merged unchanged the same day" | `0009-…md` lines 3, 26 | Table 2 |
| P11 | Compute: every training result file that records a device says `cpu`; one single-seed `mps` GRU check not used; W7 fixes one thread per run because the thread count changes results | results.md R-2 row; W7 summary lines 111–115 | §1.4 |
| P12 | A child's timeout may not exceed its parent's (validated at creation) | CLAUDE.md "Validate at creation"; roadmap W5 | §2.1 |
| P13 | Timetable dates: PR-II week 7 = 15 Nov (Sunday); Draft Final week 12 = 20 Dec; Final week 14 = 3 Jan 2027; presentation upload in the first week of finals = 10 Jan; assumes no break weeks | ENS 491/492 Student Handbook, course-flow table (items 2–5) | Table 16 |

---

## KB/JSON disagreements found (for the report-back)

1. **`solve_doorkey` update count.** `solve_doorkey_h68_s*.json` → `composite.policy_updates = 3`;
   results.md / open-questions (PR #16 audit) say 2 actual `train()` calls, because the counter
   included the discarded in-flight rollout. The KB explains the JSON field; the report quotes
   decisions (1,536) and env steps, not the update count.
2. **Label of 4,804,608.** The spec-3 summary's cost table calls it "The six withdrawn runs
   (spec 1)"; the JSON total is spec 1 (3 × 600,064) + spec 2 October (3 × 1,001,472). The W1b-1
   plan (branch) relabels it correctly. The brief repeats "spec 1".
3. **0009 status.** status.md "Waiting on a decision" says "Emir sets 0009's status"; the file
   says "Accepted 2026-11-12, provided the final PR #15 keeps this design".
4. **"~53k own / ~664k with children" (status.md H-6 row)** = the H-6.8 re-measurement
   (`n_steps` 64); the frozen checkpoint is 162,138 / 773,955 (`n_steps` 256). Not a contradiction,
   two different runs; the report gives both with what each counts.
5. Not a disagreement (per Emir): status.md "Nothing frozen" vs the W1b branch is a branch difference.
6. Resolved by the `d7be0a2` update: W1b-1 is merged and reported as completed (§2.3.7).
7. **`runs/v2/0-4_pickup_box/meta.json` → `frozen_under: "0014 (Proposed)"`**, while decision 0014 is `Accepted` (2026-11-15) and `w1b_register.json` says "0014 (Accepted 2026-11-15)". The meta was written at freezing time and not updated at registration. The report follows the decision file.
8. The W1b-1 summary still says 0014 is `Proposed` (written before acceptance); `status.md`, `open-questions.md` and the decision file say Accepted. Not a numeric disagreement.

## Brief items the repo contradicts or qualifies

1. `runs/v2/w1b_pickup_box_freeze.json` is **not on `main`**; it is on `w1b/freeze-pickup-box` (resolved: cut-off rule above).
2. The brief says W1b-1 step 1 only; the branch has step 2 done (gate 48/50, 49/50) and an addendum (resolved).
3. "≈ 28.5M = spec 1, W1a on spec 2, spec 3 with chain": the 28,549,749 total also includes the three October spec-2 runs (3,004,416).
4. `python-docx` is not installed in the `ens491` env; step 4 will edit the docx XML with the standard library (no install).
