# 0009 — `max_decisions` is counted per composite frame, not per tree

Status: Accepted 2026-11-12, provided the final PR #15 keeps this design
Date: 2026-11-05   Approved by: Emir (2026-11-12)   Source: PR #15 (Pekpak, in progress); v2 §4.4 and §13 as amended in PR #15 (not on main yet)
Reviewed by Emir: yes — 2026-11-12

## Decision
Each composite counts its own selections against its own `max_decisions`. When a frame hits its cap it closes with TIMEOUT and its parent selects again. Previously evaluation capped all decisions in the whole tree at the root's value.

## Why
- Training (`OptionEnv`) already counted per composite; evaluation (`Executor`) counted per tree. The two disagreed.
- Measured: the 9 frozen nodes give identical results on their 50 held-out seeds before and after.

## Rejected alternatives
- Keep the global cap and change `OptionEnv` instead: not evaluated in the PR.

## Evidence
- PR #15: 9 frozen nodes × 50 held-out seeds identical before/after; `unlock_door` and `solve_doorkey` training measured (Project_State note in the PR).
- Rationale Emir must be able to give: training (`OptionEnv`) already counted per composite; evaluation must use the same rule.

## Affects
[`src/skills/executor.py`](../../../src/skills/executor.py), [`src/envs/option_env.py`](../../../src/envs/option_env.py). Deep trees now get more decisions in evaluation (caps multiply).

## Open point
Recheck when PR #15 is final.

History: Decide in PR #15's discussion, then set Status here. The two other parity rules in PR #15 (parent timeout cuts the child; strict mask at every depth) do not change a locked rule.
