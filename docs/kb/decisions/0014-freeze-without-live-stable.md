# 0014 — A primitive that never met 0004 in training may be frozen on a fixed-budget test

Status: Proposed
Date: 2026-11-15   Source: [`plans/2026-11-15_W1b-1_unlock_pickup_plan.md`](../../plans/2026-11-15_W1b-1_unlock_pickup_plan.md) (revision 3)
Reviewed by Emir: not yet

## Decision (proposed)
A primitive whose training run ended with its budget exhausted, never STABLE under
[0004](0004-stabilisation-rule.md), may be frozen from its **fixed-budget final model** if both hold:

- **(a)** it passed a two-row test whose criterion was fixed before training (here W1a's test:
  sampled success ≥ 0.90 on ≥ 475 of 500 valid fresh episodes, from its own hook and from the
  parent's real handoff);
- **(b)** the Wilson 95 % lower bound of its **argmax** success from the parent's real handoff is
  ≥ 0.95 on ≥ 475 valid fresh episodes.

When several models meet (a) and (b), the smallest-numbered training seed is taken. The mode the
node was accepted in and its weak rows are written to its `meta.json` (`met_0004: false`,
`frozen_under`, `execution_mode: argmax`).

Scope: **primitives only, argmax only.** The acceptance is per parent: (b) is measured from one
parent's handoff (here `{2,2} unlock_pickup`), recorded in `meta.json`. Using the node under a
different parent needs its own (b) row first. A stopping rule under sampling is 0012's (reserved, not
written) and is not decided here. No code under `src/` changes: `freeze()` still requires STABLE,
and the freezing script sets STABLE itself in one line that cites this file.

First use: `{0,4} pickup_box` spec 3 ([0011](0011-pickup-box-spec-3.md)), training seed 45, run
`20261115T125406-03312e12d82048218c20b91fa6cad0fc`: (a) 459/500 = 0.918 own hook, 488/488 handoff;
(b) 485/488, lower bound 0.982. Seeds meeting both: 45, 46, 48, 49, 51, 52 (6 of 10).

**This experiment runs while 0014 is `Proposed`, with Emir's go-ahead, under `scripts/` only.**

## The rules this sets aside
- v2 §5.3–5.4 🔒: STABLE is two consecutive evaluations under 0004; "STABLE → FROZEN".
- v2 §9: "budget exhausted → ABANDONED (kept in the log, never used)". Seed 45 is such a run.
- v2 §13, run-persistence amendment (2026-11-05): "`SkillNode.freeze()` requires STABLE".
- Decision [0004](0004-stabilisation-rule.md), for this node.
- CLAUDE.md hard rule "`freeze()` requires STABLE".

"Only FROZEN nodes may be children" (v2 §9 🔒) is kept: the parent is trained on a FROZEN child.

## Why
- `pickup_box` met 0004 in none of 26 runs over three specifications (24 distinct models: W1a's
  seeds 43 and 44 repeat the October spec-2 runs exactly). All of seed 45's 195 argmax
  failures from its own hook are state repeats (loops); from the handoff it fails 3 of 488.
- 0.95, not 0.90: the parent needs 46/50 on fixed seeds. Its ceiling is about (chain reaches the
  handoff, 488/500) × (child's argmax handoff rate). A 0.90 child gives ≈ 0.88; seed 43 (lower
  bound 0.928) ≈ 0.93, which yields 46+/50 about 73 % of the time; seed 45 ≈ 0.97 (about 98 %).
- **Why (a) is kept, although this parent only runs the child argmax at the handoff.** (a) is the
  test spec 3 was accepted on (0011); a model that failed it is not an instance of the
  specification that passed. Without (a) the rule would select seed 44 ((b): 486/488, lower bound
  0.985; (a): sampled own hook 0.898, fails).
- **The limit of (a).** (a) is a *sampled* measurement (own hook 0.918), while the node is frozen
  and run *argmax*, where the same model scores 305/500 = 0.610 from its own hook. So (a) does not
  show that the node as frozen here is general; it shows it would be if run sampled (W1b-2's
  question). (a) rests on the first argument alone.
- Emir confirmed on 2026-11-15, after seeing that (a) removes seed 44, that (a) stays.

**Disclosure.** The ten argmax handoff rates were known when this rule was written, so the rule
and the seed it selects are not blind. Every `unlock_pickup` result is still unseen.

## The checkpoint
- While `Proposed`: only under the git-ignored `runs/w1b/`. Its identity (source `run_id`,
  `policy_sha256`, `spec_hash`, the rule over all ten seeds) is in the tracked
  `runs/v2/w1b_pickup_box_freeze.json`. `runs/v2/0-4_pickup_box/` (spec 2) is not touched.
- If rejected: `runs/w1b/` is deleted; W1b-1's results stay in `results.md` as the record of a
  rejected experiment.
- If accepted: the acceptance PR writes the checkpoint to a tracked path.

## Rejected alternatives
- **(b) alone:** selects a model that failed the test its specification was accepted on.
- **A lower 0004 threshold for primitives:** errors compound up the hierarchy (0004).
- **Waiting for a stopping rule under sampling (0012):** needs trainer changes; W1b-1 asks first
  whether the existing rule and mode are enough for the parent.
- **A larger budget or a fourth specification:** excluded by 0011.

## Evidence
- None yet for the rule itself. The experiment that settles it is
  [W1b-1](../../plans/2026-11-15_W1b-1_unlock_pickup_plan.md): with this child frozen, does
  `{2,2} unlock_pickup` reach STABLE under 0004 unchanged on at least 2 of 3 seeds, each with a
  fresh row ≥ 0.90 on env seeds 54000–54499? A ceiling gate below 46/50 reads "chain" and fails it.
  Only the "chain" reading counts against this rule; a "selector" reading (gate passed, parent
  not STABLE) says nothing about it either way.
- Result of W1b-1 (2026-11-15): the ceiling gate passed (48/50 stopping, 49/50 held-out), so the
  "chain" reading did not occur; `{2,2} unlock_pickup` is STABLE under 0004 on 3 of 3 seeds, fresh
  486/500 on each. The three seeds act identically (one result).
  [Summary](../../plans/2026-11-15_W1b-1_summary.md), [`results.md`](../results.md).
- Behind the selection, in [`results.md`](../results.md): W1a spec 3 (7/10),
  `runs/v2/w1a_spec3_sampled.json` and `w1a_spec3_sampled_ext.json`.

## Affects
- If accepted: v2 §5.4, §9 (a second route to FROZEN, and what ABANDONED means), §11 (T6, T7), the
  §13 amendment's wording, CLAUDE.md "Skill nodes", tracked registration of the child and parent.
- While `Proposed`: new files under `scripts/`, `runs/w1b/` (ignored) and result JSONs under `runs/v2/`.
