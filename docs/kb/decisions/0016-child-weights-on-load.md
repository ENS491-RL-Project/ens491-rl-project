# 0016 — A parent's checkpoint records its children's weight hashes; `spec_hash` stays as it is

Status: Proposed
Date: 2026-11-15   Approved by: —   Source: code audit, 2026-11 (finding "`spec_hash` does not verify behaviour")
Reviewed by Emir: no

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
Not run. The experiment that settles it: with the check implemented, (a) every frozen node in
`runs/v2/` loads and `scripts/eval_regression.py` gives the recorded numbers, and (b) a parent
loaded on a child whose weights were swapped for another seed's is refused. To be recorded in
[`results.md`](../results.md).

## Affects
[`src/skills/registry.py`](../../../src/skills/registry.py), `tests/test_node_registry.py`; v2 §2
(`spec_hash` 🔒, wording only). No code is changed until this is Accepted.

## Open point
A renamed-but-identical predicate still changes `spec_hash`, and a changed predicate body under the
same name still does not. This decision does not address that.
