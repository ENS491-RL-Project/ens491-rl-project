# 0013 — C-2 uses a frozen v2 primitive as the lateral source, with an untrained-source control

Status: Accepted
Date: 2026-11-15   Source: [`plans/2026-11-15_W7_pn_lateral_plan.md`](../../plans/2026-11-15_W7_pn_lateral_plan.md) (revision 7)
Reviewed by Emir: yes — 2026-11-15 (line set at his instruction in chat)

## Decision
- C-2's lateral source is the frozen `{0,2} open_door`, not the legacy `runs/pn4/col01` (v2 §10,
  §12) and not a newly retrained source (v2 §7 [F23]).
- C-2 has a control arm whose lateral source is an untrained network. It is not a node, is never
  registered, and is allowed in this experiment only. v2 §7's rule for library nodes is unchanged.

## Why
- `open_door` is already trained under the normalised pipeline; F23's reason (pipeline match) holds.
- Without the control, a faster lateral arm cannot be told apart from a larger network.

## Rejected alternatives
- Retraining `col01` on Empty-8x8: a source unrelated to the curriculum; extra work for no gain.
- Empty-8x8 → FourRooms (Progress Report I §5.2): FourRooms never stabilised (C-1 not done).

## Evidence
None, and W7 does not provide any: W7 uses this source, it does not compare sources. The
alternatives (legacy `col01` retrained, another frozen primitive, all earlier primitives) are not
measured. The experiment that would settle the choice is the empirical row "Lateral sources" in
[`open-questions.md`](../open-questions.md), which is only worth running if W7 shows transfer.
Experiment that used this source: W7 ([`results.md`](../results.md), row of 2026-11-15); it is not evidence for the choice.

## Affects
v2 §7 [F23], §10, §12 and §13 wording (corrected in the same PR as this file); open-questions
"Retrain the PN-4 column…" (becomes moot; removed in W7's result PR) and "Fresh test seed set"
(60000–60499 is taken by W7); decision [0003](0003-progressive-networks-off-critical-path.md)
"Affects"; [`status.md`](../status.md) Track C.
