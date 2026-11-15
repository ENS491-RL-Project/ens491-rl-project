# 0016 — A parent's checkpoint records its children's weight hashes; `spec_hash` stays as it is

Status: Accepted 2026-11-15
Date: 2026-11-15   Approved by: Emir (2026-11-15, relayed by Mehmet P.)   Source: code audit, 2026-11 (finding "`spec_hash` does not verify behaviour")
Reviewed by Emir: yes — 2026-11-15

## Decision
- `spec_hash` is unchanged: it covers what a node was asked to do (env, predicate names, timeout,
  obs pipeline, children's hashes), not what its weights are.
- `NodeRegistry.save` writes, for a composite, each child's `policy_hash`
  ([`run_store.py`](../../../src/training/run_store.py)) into the parent's `meta.json`.
  `load_into` compares them with the children it is given and raises on a difference.
  A `meta.json` without the field loads as today.

## Why
- A predicate is hashed by its name only, and a child's weights are not hashed at all
  ([`node.py`](../../../src/skills/node.py) `spec()`). A parent can therefore be loaded on top of a
  child with the same spec and different weights (another seed, a retrain), and nothing reports it.
- The final test protocol (branch `feat/final-test-protocol`, not merged) pins child weights in
  its lock; ordinary registry loads have no such check.

## Rejected alternatives
- Adding weights to `spec_hash`: every existing checkpoint in `runs/v2/` would fail with
  `SpecMismatch` and need a migration, for the same protection.
- Hashing predicate source code: changes with formatting and Python version; machines differ.

## Evidence
Run 2026-11-15, after acceptance ([`results.md`](../results.md)):
- `scripts/eval_regression.py`: the 9 frozen nodes it loads reproduce their recorded held-out
  scores exactly. Their `meta.json` files predate this decision and carry no child hashes, so this
  shows only that old checkpoints still load; no existing parent is pinned until it is saved again.
- `tests/test_node_registry.py::test_parent_refuses_a_child_with_other_weights`: a parent loads on
  the children it was saved with, is refused on a same-spec child with other weights, and a
  checkpoint without the field loads.
- The check found one such load in the tests (`test_load_frozen_composite_round_trips` rebuilt the
  children with new random weights); the test now reuses the saved children.

## Affects
[`src/skills/registry.py`](../../../src/skills/registry.py), `tests/test_node_registry.py`; v2 §2
(`spec_hash` 🔒, one sentence added).

## Open point
A renamed-but-identical predicate still changes `spec_hash`, and a changed predicate body under the
same name still does not. This decision does not address that.
