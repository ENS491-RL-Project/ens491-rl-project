# Status

Where each track stands and what the open PRs close. Last updated: 2026-11-15.

Until Project_State is migrated (KB sync PR #22), [`ENS491_Project_State.md`](../ENS491_Project_State.md)
stays authoritative; where the two disagree, Project_State wins. Measurements are in [`results.md`](results.md).

## Tracks

| Track | Done | Stopped / open | Next | Source |
|---|---|---|---|---|
| H Hierarchy (critical path) | H-0…H-6, H-6.8, H-6.9. 9 frozen nodes; `{2,1} solve_doorkey` depth 3, held-out 1.00. Training/evaluation parity (PR #15, 2026-11-12); training parity audit and correct update counter (PR #16, 2026-11-15); every training run archived under `runs/training/`, freeze needs STABLE, no registry overwrite (PR #17, 2026-11-15) | H-7 and H-7a stopped: `{0,4} pickup_box` not STABLE under two specifications. `unlock_pickup` and H-7b (BlockedUnlockPickup) not run | Execution-mode decision, then UnlockPickup ([open-questions](open-questions.md)) | Project_State Track H; [`plans/2026-10-02_H7_summary.md`](../plans/2026-10-02_H7_summary.md) |
| C Continual learning / PN | — | C-1, C-2, C-3 not started | C-2 on primitives first ([0003](decisions/0003-progressive-networks-off-critical-path.md)) | Project_State Track C |
| R Recognition | — | R-1, R-3 not started. R-2: GRU work in open PR #18 (contains #14) | Finish #18 (3-seed results);  root selection after R-1 ([open-questions](open-questions.md)) | Project_State Track R; PR #18 |
| G Growth | — | G-1…G-3 not started | Required this semester ([0007](decisions/0007-automatic-growth-in-scope.md)) | Project_State Track G |
| UI | — | UI-1, UI-2 not started | — | Project_State Track UI |

## Open PRs

| PR | Author | What it closes | Review | KB follow-up when merged |
|---|---|---|---|---|
| #18 GRU task identification | elifsu42; finished by Emir | GRU pipeline over 11 PPO policies, mapped to 9 task labels; sequence-length ablation N = 1…50 | Review fixes pushed 2026-11-13 (device, output paths, tests, results files). 4 of the 11 policies are identifiable from the first observation. A CPU reproduction matches N = 1 and 5 within 1 point but not N ≥ 10, so the 3-seed run is in progress | Add to `results.md`; update the GRU row in [open-questions](open-questions.md); close #14 |
| #14 GRU pilot training | elifsu42 | Pilot GRU training and PPO rollout tests | Its only commit is in #18 | Close when #18 merges |
| #5 refactor src/ structure | TulayEce | v1-era `src/` restructure, open since 2026-05-11 | Not reviewed | Decide: close or rebase onto v2 |

## Still to do in the KB sync PR (#22)

- Move Project_State's remaining content into `status.md` and `results.md`; turn Project_State into a pointer or archive it.
- Remove the "Project_State stays authoritative" line here and in `CLAUDE.md` (Docs and Decisions sections).
- Emir sets [0009](decisions/0009-max-decisions-per-frame.md)'s status (PR #15 merged with the per-frame design).
