# Review of the H-0 → H-5 plan (2026-10-01)

> Companion to `2026-10-01_v2_H0-H5_plan.md` (copied from `~/.claude/plans/shiny-hugging-stallman.md`).
> Status: **all corrections below have been merged into the plan** (revision of 2026-10-01, items R1–R5).
> The plan file is now the single source of truth; this review is kept as a record only.

---

## Corrections to the plan (verified against the code)

1. **Do not delete `src/continual_learning/stabilization.py` in H-3.**
   `pn_trainer.py:19` imports `StabilizationMonitor`, and `ColumnTrainer` (including `load_policy_into`, which H-3 uses to re-evaluate `col01`) depends on it. Deleting it breaks `pn_trainer.py`, `scripts/train_pn4.py` and the trainer tests.
   → Mark it deprecated in H-3; delete it only when `ColumnTrainer` itself is replaced.

2. **F11's Wilson threshold is stricter than intended.**
   With N = 50 evaluation episodes, the Wilson 95 % lower bound is:

   | Successes / 50 | SR | Wilson LB |
   |---|---|---|
   | 45 | 0.90 | 0.786 |
   | 46 | 0.92 | 0.812 |
   | 47 | 0.94 | 0.838 |
   | 48 | 0.96 | 0.865 |

   "SR ≥ 0.90 and LB ≥ 0.85" therefore means ≥ 48/50 (96 %), twice in a row.
   → Use **SR ≥ 0.90 and Wilson LB ≥ 0.80** at N = 50 (effectively ≥ 46/50), or keep LB ≥ 0.85 with N = 100 (≥ 92/100).

3. **Node id of `go_to_goal`.** The H-4 table says `{0,4}`; v2 §11 defines it as `{0,3}` (third primitive created). Use `{0,3}` everywhere, including run directories.

## Fact to confirm before execution

- The plan reports `torch 2.14.1+cpu` (no CUDA) and mentions a "MacBook Pro (M5 Pro)" target. Confirm the MacBook is real. Run `nvidia-smi`: if the RTX 3070 Ti is present, torch was installed without CUDA. CPU remains the right default for these MLP policies either way (F31).

## Answers to the plan's "Decision" items (recommended; edit before resuming if you disagree)

| Item | Answer |
|---|---|
| F10: a node's reward comes from its own goal predicate, never the env's reward | **Yes** |
| F11: fixed held-out eval seeds + Wilson bound + separate freeze-time seed set | **Yes, with correction 2** |
| F16: `ChannelNormalize` on for all new nodes; legacy `col01` retrained if used as a lateral source | **Yes** |
| F20: single lateral source until C-2; multiple sources are summed, never concatenated | **Yes** |
| F23: C-2 retrains its own lateral source under the current pipeline | **Yes** (not on the critical path) |

All items marked "Baked in" in Part A are accepted.

## Resume prompt (paste into Claude Code, in default or plan mode, not auto mode)

```
Execute docs/plans/2026-10-01_v2_H0-H5_plan.md starting at Step 0 (branch + docs commit), then H-0.
Stop after each task's gate and report before starting the next one.
For H-0.5, show me the diff of the doc amendments before committing.
```

Note: the git recipe that used to be here branched off `main` and would have dropped commit `acd5ffd`
(matplotlib/pandas fix). The plan's Step 0 branches off `fix/declare-missing-deps` instead (R5).
