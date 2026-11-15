# 0005 — Tasks are declared by hand in a fixed bottom-up order (growth Phase A)

Status: Accepted as Phase A only — moving to Phase B is required (0007)
Date: 2026-10-01   Approved by: Emir (2026-11-12)   Source: v2 §8; chat 'Architecture policy training setup' (6 Oct); SUPERVISOR_QUESTIONS Q1, Q6
Reviewed by Emir: yes — 2026-11-12

## Decision
The experimenter declares each task (env, reset hook, goal, timeout, node kind, children). Order: get_key → open_door → go_to_goal → unlock_door → solve_doorkey → … Everything after declaration (train → evaluate → stabilise → freeze → register) is automatic. Each node is trained in isolation on its own episodes.

## Why
- Separates 'can the hierarchy compose' from 'can the agent decide when to grow'. The second is Track G.
- Automatic growth stays in scope this semester (see 0007); Phase A is the first step, not the end state.

## Rejected alternatives
- Curriculum learning / automatic ordering now: deferred to Phase B (composite-first heuristic, Q3).

## Evidence
No comparison run. Phase B (composite-first heuristic, G-1…G-3) is the next step, not an optional one.
Related results in [`results.md`](../results.md): H-4 primitives T1, T2, T4.

## Affects
All `scripts/train_*.py`; Track G.

## Open point
The report must state plainly that tasks and subgoal predicates are hand-defined in Phase A.

History: The supervisor asked about exactly this on 6 October (fixed order vs curriculum, episode definition, when training stops). His reaction is not recorded.
