# 0015 — A final test set of 200 reserved seeds, opened once per locked candidate

Status: Proposed
Date: 2026-11-12   Approved by: —   Source: v2 §5.3 and §13 (amended 2026-11-12); [protocol](../../protocols/2026-11-12_final_and_parity_experiments.md)
Reviewed by Emir: no

## Decision
A third seed set, used only for the final evaluation: the 200 seed IDs in [`final_test_v1.json`](../../protocols/final_test_v1.json). Before any final episode runs, one configuration per task/arm and one checkpoint per training seed (42–44) are locked: the lock pins the seed manifest, node and child weights, env/reset-hook/mask settings and evaluator sources. Each candidate is evaluated once; a second opening is refused.

## Why
- The stopping seeds (10000+) and the held-out seeds (20000+) were read repeatedly during development, so neither can carry an "untouched" claim.
- No final score may choose the next training configuration.

## Rejected alternatives
- A contiguous range from 50000: proposed first (2026-11-12) and dropped; 50000–59999 was then reserved for W1 and 60000–60499 used by W7.

## Evidence
- The 200 IDs were drawn after auditing every integer in 107 historical JSON files (100,116 excluded values). Checked 2026-11-15: none lies in 50000–59999 or 60000–60499 (range 271,136,090–1,072,709,092).
- `tests/test_final_protocol.py`: lock mismatches and a second opening are refused.
- Not yet run: no candidate locked, no final episode played.

## Affects
[`src/training/final_protocol.py`](../../../src/training/final_protocol.py), [`scripts/eval_final.py`](../../../scripts/eval_final.py), [`scripts/final_test_protocol.py`](../../../scripts/final_test_protocol.py).

## Open point
Seed separation is not layout separation: two seeds can give the same MiniGrid layout. Local guards cannot stop someone deleting the claim files. JSON files added after 2026-11-12 were not re-audited beyond the two ranges above.
