# 0001 — Replace the v1 column/meta-controller design with v2 skill nodes

Status: Accepted
Date: 2026-10-01 (design) · 2026-10-04 (merged in the personal repo) · 2026-10-20 (PR #12, team repo)   Approved by: Emir (merge to main)   Source: v2 §0, §13; PR #12; chat 'RL project agent durumu' (30 Sep–1 Oct)
Reviewed by Emir: partly — whole redesign approved 2026-10-01; individual choices are 0002–0006

## Decision
Everything the agent knows is a skill node (primitive or composite) with initiation, goal, timeout and policy. There is no per-layer MetaController object; a composite's policy is its meta-controller. Composites are trained as SMDPs (`OptionEnv` + MaskablePPO).

## Why
- v1 stalled for four months (June–September 2026). Progressive Networks sat on the critical path ahead of the hierarchy and the second column never stabilised on FourRooms.
- The proposal had demoted the multi-level hierarchy to a 'Phase 3 stretch goal', although the supervisor's 7 April notes asked for an unbounded recursive controller as the priority, and on 28 April he advised solving the hierarchy before Progressive Network details.
- v1 left key questions open (how layers are created, how termination propagates) while code was written around them; `Column.sub_layer` was always empty.

## Rejected alternatives
- Keep v1 and finish PN-3 first: rejected, it kept the hierarchy blocked behind FourRooms.

## Evidence
- v1 outcome: one stable column (Empty-8x8), FourRooms column never stabilised: [`runs/pn4/NOTES.md`](../../../runs/pn4/NOTES.md).
- v2 outcome: 9 frozen nodes, `solve_doorkey` 50/50 at depth 3: [`plans/2026-10-01_run_summary.md`](../../plans/2026-10-01_run_summary.md), [`plans/2026-10-02_H6_summary.md`](../../plans/2026-10-02_H6_summary.md); the 9th frozen node (`drop_held`): [`plans/2026-10-02_H7_summary.md`](../../plans/2026-10-02_H7_summary.md).

## Affects
All of [`src/skills/`](../../../src/skills/), [`src/envs/option_env.py`](../../../src/envs/option_env.py), [`src/training/`](../../../src/training/). Old design archived in `docs/archive/ENS491_Recursive_Hierarchy_Design.md`.

## Open point
Emir approved the redesign as a whole ('tamam', 1 October). Individual choices inside it are listed as separate decisions (0002–0006) so each can be confirmed on its own.
