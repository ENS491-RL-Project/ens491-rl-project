# Status

Where each track stands and what the open PRs close. Last updated: 2026-11-12.

Until PRs #15–#17 merge and the next KB PR migrates it, [`ENS491_Project_State.md`](../ENS491_Project_State.md)
stays authoritative; where the two disagree, Project_State wins. Measurements are in [`results.md`](results.md).

## Tracks

| Track | Done | Stopped / open | Next | Source |
|---|---|---|---|---|
| H Hierarchy (critical path) | H-0…H-6, H-6.8, H-6.9. 9 frozen nodes; `{2,1} solve_doorkey` depth 3, held-out 1.00 | H-7 and H-7a stopped: `{0,4} pickup_box` not STABLE under two specifications. `unlock_pickup` and H-7b (BlockedUnlockPickup) not run | Execution-mode decision, then UnlockPickup ([open-questions](open-questions.md)) | Project_State Track H; [`plans/2026-10-02_H7_summary.md`](../plans/2026-10-02_H7_summary.md) |
| C Continual learning / PN | — | C-1, C-2, C-3 not started | C-2 on primitives first ([0003](decisions/0003-progressive-networks-off-critical-path.md)) | Project_State Track C |
| R Recognition | — | R-1, R-3 not started. R-2: GRU work in open PRs #14, #18 | Review #14/#18; root selection after R-1 ([open-questions](open-questions.md)) | Project_State Track R; PRs #14, #18 |
| G Growth | — | G-1…G-3 not started | Required this semester ([0007](decisions/0007-automatic-growth-in-scope.md)) | Project_State Track G |
| UI | — | UI-1, UI-2 not started | — | Project_State Track UI |

## Open PRs

| PR | Author | What it closes | KB follow-up when merged |
|---|---|---|---|
| #15 Fix/train eval parity | mehmetpekpak | `OptionEnv` and `Executor` apply the same rules (parent timeout, strict mask at every depth, `max_decisions` per frame); amends v2 §4.4, §5.2, §13 | Confirm [0009](decisions/0009-max-decisions-per-frame.md); add its parity measurements to `results.md` |
| #16 Fix/checkpoint and update count | mehmetpekpak | `load_into` fails on a missing weights file; `policy_updates` counts real `train()` calls; adds a training-parity audit plan | Note that `policy_updates` in older JSON reports is one too high; add the audit to `results.md` |
| #17 Feat/training run provenance | mehmetpekpak | Run provenance store (`run_store.py`); also carries the #15 and #16 changes | Review after #15 and #16; check whether its v2 change touches a 🔒 rule (needs a decision file) |
| #18 GRU policy identification | elifsu42 | GRU pipeline over 11 PPO policies; reported test window accuracy 73.38%, macro 77.33% (identifies policies, not task ids) | Add to `results.md`; update the GRU row in [open-questions](open-questions.md) |
| #14 GRU pilot training | elifsu42 | Pilot GRU training and PPO rollout tests | Same as #18 |
| #5 refactor src/ structure | TulayEce | v1-era `src/` restructure, open since 2026-05-11 | Decide: close or rebase onto v2 |

## Next KB PR (after #15–#17 merge)

- Move Project_State's remaining content into `status.md` and `results.md`; turn Project_State into a pointer or archive it.
- Remove the "Project_State stays authoritative" line here and in `CLAUDE.md` (Docs and Decisions sections).
- After #15: `CLAUDE.md`'s `max_decisions` rule says it is counted per composite frame; Emir sets [0009](decisions/0009-max-decisions-per-frame.md)'s status.
- After #17: if it adds a hard rule (run provenance), add it to `CLAUDE.md`.
