# 0002 — The skill hierarchy is a DAG, not a tree

Status: Accepted by Emir; supervisor confirmation not recorded
Date: 2026-04-28 (idea) · 2026-10-01 (v2)   Approved by: Emir   Source: v2 §3; SUPERVISOR_QUESTIONS Q2; chat 'Önceki konuşmaları hatırlama' (28 Apr)
Reviewed by Emir: yes — his idea, 2026-04-28

## Decision
A composite may use any earlier FROZEN node as a child, from any level, and a node may have several parents. Its level is `1 + max(child levels)`, derived, never assigned.

## Why
- Reuse is the point of the project. Emir observed on 28 April that the same primitive serves several meta-controllers, so the structure is a graph, not a tree.
- Acyclic by construction: a node can only use nodes frozen before it.
- Children from any level remove the need for dummy wrappers that 'lift' a primitive to the right layer.

## Rejected alternatives
- Tree with one parent per column (v1, `ENS491_Recursive_Hierarchy_Design.md` §3.1): sharing impossible.

## Evidence
- `{2,1} solve_doorkey` uses `{1,1} unlock_door` (level 1) and `{0,3} go_to_goal` (level 0) directly: [`plans/2026-10-02_H6_summary.md`](../../plans/2026-10-02_H6_summary.md).

## Affects
v2 §2–§3; executor availability is memoised per step because naive recursion is exponential on a DAG.

## Open point
Ask the supervisor (Q2) and record the answer here.
