# 0003 — Progressive Networks are postponed off the critical path, not dropped

Status: Accepted — hierarchy first; PN measured empirically, primitives first
Date: 2026-09-30 (proposed) · 2026-10-05 (re-examined)   Approved by: Emir (2026-11-12)   Source: v2 §7; chat 'Handoff document review process' (5 Oct); supervisor advice 28 Apr; meeting notes 7 Apr
Reviewed by Emir: yes — 2026-10-05 (order) and 2026-11-12 (scope)

## Decision
Skills are frozen, so forgetting is prevented without PN. PN's contribution is measured as forward transfer (steps to STABLE with vs without laterals, C-2). PN is an independent track, not on the critical path.

## Why
- Supervisor, 28 April: focus on the hierarchy first; adding PN afterwards is easy.
- Putting PN first is what stalled v1 (see 0001).

## Rejected alternatives
- PN as the backbone of continual learning (proposal objective 3.1.1 (proposal is not in the repo)).
- Dropping PN from the report: rejected, it contradicts the supervisor ('later', not 'never').

## Evidence
C-2 ran as W7 (2026-11-15, [`results.md`](../results.md)): `go_to_goal` with no lateral, with a lateral from the frozen `{0,2} open_door`, and with a lateral from an untrained network, ten seeds each. **No measurable benefit**: no arm beats another on steps to STABLE (medians 190,464 / 178,176 / 184,320); a reduction of 25 % or more would have passed 8 times in 10. One pair only ([W7 summary](../../plans/2026-11-15_W7_summary.md)).
Related results in [`results.md`](../results.md) (background, not C-2): v1 PN-4; H-3 legacy `col01`.

## Affects
Track C in v2 §12. C-2 did not need the PN-4 Empty-8x8 column ([`runs/pn4`](../../../runs/pn4)) retrained: its source is the frozen `{0,2} open_door` ([0013](0013-c2-lateral-source-open-door.md)).

## Open point
Decided 2026-11-12: start C-2 with laterals between primitives. Laterals between composites are decided by a later experiment, not by default. Whether PN is part of the reported contribution follows from C-2's result (see 0006).

History: 1) v2 allows laterals between primitives only. Emir's April design had laterals at every level (`{n,m-1} → {n,m}`); this narrowing was made without his review. 2) The early decision log named PN + options + recognition as the contribution. Whether the report claims PN as part of the contribution is open (see open-questions).
