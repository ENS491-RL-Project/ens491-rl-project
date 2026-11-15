# 0006 — The claim: compose while learning, reuse compositions, without forgetting

Status: Accepted as the direction (2026-11-12); evidence partial
Date: 2026-10-02 (first framing) · 2026-11-12 (Emir's formulation)
Reviewed by Emir: yes — 2026-11-12
Source: Emir, 2026-11-12; H-6/H-7 summaries; SUPERVISOR_BRIEF Q10

## Decision
The project's main claim, in Emir's words: the agent **composes skills while it learns**, the
**compositions themselves are reused** later as building blocks, and it does this **without forgetting**.
- Primary: composition and reuse of compositions (cost of new tasks given the library).
- No-forgetting is part of the claim, measured on the whole system as the library grows, not only
  on frozen skills (which cannot forget by construction). Where forgetting can happen is the
  recognition layer (AE/GRU, v2 §6.3).
- PN forward transfer is claimed only if C-2 shows it (0003).
- Not claimed: the hierarchy learns a single task faster than flat PPO.

## Why
- On DoorKey-6x6, counting child training, the hierarchy from scratch was not cheaper than flat PPO.
- Composing already-frozen skills was much cheaper; a composite was reused one level up
  (`unlock_door` inside `solve_doorkey`).

## Evidence
- Have: amortisation measurements ([`plans/2026-10-02_H7_summary.md`](../../plans/2026-10-02_H7_summary.md)), `unlock_door` reused at level 2.
- Missing: (1) composition *while learning*, i.e. new tasks arriving in a sequence and being solved by new
  composites without a human declaring children: Phase B (0005, 0007); (2) a composite reused by
  more than one later task; (3) retention of the recognition layer across the sequence (R-2);
  (4) possibly a flat baseline with memory, for a fair cost comparison.
- Results in [`results.md`](../results.md): H-7 amortisation; H-7 `pickup_box` spec 1 and flat baselines;
  H-6 / H-6.8 `{2,1} solve_doorkey` vs flat PPO.

## Rejected alternatives
- "Hierarchy learns faster" (original framing): not supported by our measurements.
- "No forgetting" as the headline: trivial for frozen skills alone.

## Affects
Report framing; evaluation plan v2 §11; Tracks G and R.
