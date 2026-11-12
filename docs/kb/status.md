# Status

Where each track stands and what the open PRs close. Last updated: 2026-11-12.

Until PRs #16 and #17 merge and Project_State is migrated, [`ENS491_Project_State.md`](../ENS491_Project_State.md)
stays authoritative; where the two disagree, Project_State wins. Measurements are in [`results.md`](results.md).

## Tracks

| Track | Done | Stopped / open | Next | Source |
|---|---|---|---|---|
| H Hierarchy (critical path) | H-0…H-6, H-6.8, H-6.9. 9 frozen nodes; `{2,1} solve_doorkey` depth 3, held-out 1.00. Training/evaluation parity (PR #15, 2026-11-12) | H-7 and H-7a stopped: `{0,4} pickup_box` not STABLE under two specifications. `unlock_pickup` and H-7b (BlockedUnlockPickup) not run | Execution-mode decision, then UnlockPickup ([open-questions](open-questions.md)) | Project_State Track H; [`plans/2026-10-02_H7_summary.md`](../plans/2026-10-02_H7_summary.md) |
| C Continual learning / PN | — | C-1, C-2, C-3 not started | C-2 on primitives first ([0003](decisions/0003-progressive-networks-off-critical-path.md)) | Project_State Track C |
| R Recognition | — | R-1, R-3 not started. R-2: GRU work in open PRs #14, #18 | Review #18 (#14 is contained in it); root selection after R-1 ([open-questions](open-questions.md)) | Project_State Track R; PRs #14, #18 |
| G Growth | — | G-1…G-3 not started | Required this semester ([0007](decisions/0007-automatic-growth-in-scope.md)) | Project_State Track G |
| UI | — | UI-1, UI-2 not started | — | Project_State Track UI |

## Open PRs

| PR | Author | What it closes | Review (2026-11-12) | KB follow-up when merged |
|---|---|---|---|---|
| #16 Fix/checkpoint and update count | mehmetpekpak | `load_into` fails on a missing weights file; `policy_updates` counts real `train()` calls; adds a paired training-parity audit (18 runs) | Mergeable; small fixes asked (audit output path, script style, plan language) | Add the audit to `results.md`; `solve_doorkey` STABLE in 2 updates, not 3. Older JSON reports overcount by one only when early stopping fell on a rollout boundary; do not subtract one from every report |
| #17 Feat/training run provenance | mehmetpekpak | Run provenance store (`run_store.py`); `freeze()` requires STABLE; `registry.save` refuses to overwrite; frozen nodes cannot be retrained. Also carries #16 | Not mergeable as is: every training call fails on Windows (`os.fsync` on a read-only file); overwrite check runs only after training. Merge after #16 | Add its rules to `CLAUDE.md` Hard rules; no 🔒 rule changes, so no decision file |
| #18 GRU policy identification | elifsu42 | GRU pipeline over 11 PPO policies; reported test window accuracy 73.38%, macro 77.33% (identifies policies, not task ids) | Runs against current `main`. 4 of the 11 policies are identifiable from the first observation alone; seq-len 1 result and the three hard pairs asked for; device, output paths, results file | Add to `results.md`; update the GRU row in [open-questions](open-questions.md) |
| #14 GRU pilot training | elifsu42 | Pilot GRU training and PPO rollout tests | Its only commit is in #18 | Close when #18 merges |
| #5 refactor src/ structure | TulayEce | v1-era `src/` restructure, open since 2026-05-11 | Not reviewed | Decide: close or rebase onto v2 |

## Still to do in the KB sync PR (after #16–#18 merge)

- Move Project_State's remaining content into `status.md` and `results.md`; turn Project_State into a pointer or archive it.
- Remove the "Project_State stays authoritative" line here and in `CLAUDE.md` (Docs and Decisions sections).
- Emir sets [0009](decisions/0009-max-decisions-per-frame.md)'s status (PR #15 merged with the per-frame design).
