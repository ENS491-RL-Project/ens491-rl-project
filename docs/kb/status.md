# Status

Where each track stands and what the open PRs close. Last updated: 2026-11-15.

Until Project_State is migrated (KB sync PR #22), [`ENS491_Project_State.md`](../ENS491_Project_State.md)
stays authoritative; where the two disagree, Project_State wins. Measurements are in [`results.md`](results.md).

## Tracks

| Track | Done | Stopped / open | Next | Source |
|---|---|---|---|---|
| H Hierarchy (critical path) | H-0…H-6, H-6.8, H-6.9. 9 frozen nodes; `{2,1} solve_doorkey` depth 3, held-out 1.00. Training/evaluation parity (PR #15, 2026-11-12); training parity audit and correct update counter (PR #16, 2026-11-15); every training run archived under `runs/training/`, freeze needs STABLE, no registry overwrite (PR #17, 2026-11-15) | H-7 and H-7a stopped: `{0,4} pickup_box` not STABLE under two specifications. `unlock_pickup` and H-7b (BlockedUnlockPickup) not run | Execution-mode decision, then UnlockPickup ([open-questions](open-questions.md)) | Project_State Track H; [`plans/2026-10-02_H7_summary.md`](../plans/2026-10-02_H7_summary.md) |
| C Continual learning / PN | — | C-1, C-2, C-3 not started | C-2 on primitives first ([0003](decisions/0003-progressive-networks-off-critical-path.md)) | Project_State Track C |
| R Recognition | R-2 in part: GRU task identification on 9 known tasks, 85 % task-episode macro accuracy at N ≥ 10 (PR #18, 2026-11-15) | R-1, R-3 not started. R-2: old-task accuracy and switching latency not measured | Task registry (R-3) and root selection ([open-questions](open-questions.md)) | Project_State Track R; [`results.md`](results.md) |
| G Growth | — | G-1…G-3 not started | Required this semester ([0007](decisions/0007-automatic-growth-in-scope.md)) | Project_State Track G |
| UI | — | UI-1, UI-2 not started | — | Project_State Track UI |

## Open PRs

| PR | Author | What it closes | Review | KB follow-up when merged |
|---|---|---|---|---|
| #5 refactor src/ structure | TulayEce | v1-era `src/` restructure, open since 2026-05-11 | Not reviewed | Decide: close or rebase onto v2 |

## Still to do in the KB sync PR (#22)

- Move Project_State's remaining content into `status.md` and `results.md`; turn Project_State into a pointer or archive it.
- Remove the "Project_State stays authoritative" line here and in `CLAUDE.md` (Docs and Decisions sections).
- Emir sets [0009](decisions/0009-max-decisions-per-frame.md)'s status (PR #15 merged with the per-frame design).
