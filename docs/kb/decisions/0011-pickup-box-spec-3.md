# 0011 — `pickup_box` specification 3: half of training starts at the real handoff

Status: Accepted
Date: 2026-11-15   Source: [`plans/2026-11-15_W1a_spec3_plan.md`](../../plans/2026-11-15_W1a_spec3_plan.md) (revision 2); W1a and its handoff diagnostic
Reviewed by Emir: yes — 2026-11-15 (line set at his instruction in chat)

## Decision
`{0,4} pickup_box` (T6) gets a third specification. Spec 3 = spec 2 with **half of the training
episodes started at the real handoff**:

- with probability 0.5: spec 2's hook, unchanged (`open_door`, `key_off_door`,
  `randomize_agent_pose(any)`). Same distribution as spec 2, not the same layouts: the coin flip
  consumes a draw from the env's RNG;
- with probability 0.5: UnlockPickup from the env's own reset, then the frozen `{1,1} unlock_door`
  and `{0,9} drop_held` run to their ends (argmax, as they are frozen today); `pickup_box` starts in
  the state they leave. If the chain does not leave the handoff state (door open, hand empty), the
  layout is rejected and redrawn (`UnsuitableLayout`).

Everything else is spec 2: goal `carrying:box`, initiation `and(carrying:none, door:open)`, timeout
120, `MiniGrid-UnlockPickup-v0`, PPO configuration, budget 1,000,000 env steps, `eval_every` 20,000.
The hook has its own spec string, so spec 3 has its own `spec_hash`; spec 2 and its hash are
untouched.

Executing frozen nodes inside a reset is a new mechanism, not a precedent (`open_door_tk` built its
parent's distribution with a hand-written hook). It relies on `DummyVecEnv`, and the hook must
build the 147-dim normalised observation itself, because `SubgoalEnv` hands it the raw env.

**This experiment runs while 0011 is `Proposed`, with Emir's go-ahead, under `scripts/` only.**
No file under `src/` changes, v2 is not edited, and no spec-3 model is frozen or registered while
this decision is open.

## The rule this sets aside
v2 §11, H-7 amendment 1: *"One re-specification only. If T6b or the new T6 does not stabilise
within its budget (100,000 and 1,000,000 env steps), that is the finding and H-7a stops."*

Spec 2 was that one re-specification, and it did not stabilise. Spec 3 is therefore a second
re-specification of T6, which the clause forbids in so many words. This decision sets the clause
aside once, for this reason:

- **W1a** (criterion fixed before any result): spec 2, scored as it acts (sampled), passes on 6 of
  10 training seeds; the rule needs 7. All ten pass from their own start states; the four that
  fail (46, 47, 50, 52) fail at the handoff (0.874–0.892).
- **Handoff diagnostic** (readings fixed before any result): with the dropped key removed from the
  grid, the four failing seeds go from 0.880 to 0.985 pooled, and all ten are at 0.967 or above;
  doubling the time gives 0.921. Reading: **Key**, a property of the skill.
- **Key-position measurement** (491 valid handoff states, env seeds 51000–51499): the key is never
  orthogonally adjacent to the door (0/491). In all 491 states the agent stands on the cell in
  front of the door, facing along the wall (264 south, 227 north, never the door), with the key on
  the cell directly in front of it. Spec 2's random-pose hook produces this configuration only
  rarely; how rarely was not measured.

So the mismatch is the start state: one narrow, always-the-same situation that spec 2 hardly
trained in. The diagnostic shows the key's presence there causes the failures; it does not show
that training in that situation fixes them. The experiment in the plan tests that.

All three specifications (spec 1 withdrawn, spec 2, spec 3) stay in every table, each with its cost.

## If this fails
Spec 3 is the **last specification of `pickup_box` in this line**. A "no further attempt" rule has
now been set aside once; this one is not to be. The choice left is Emir's and is recorded as its own
decision: continue with a passing model under a declared selection rule and full cost accounting,
or stop the UnlockPickup line.

## Rejected alternatives
- **Train at the handoff only:** with the mixture, the old half keeps spec 2's distribution, so a
  change in the result is attributable to the added half and the own-hook row of the test stays
  meaningful.
- **A longer timeout:** the diagnostic's Time reading did not hold on the failing seeds (0.921 < 0.95).
- **A memory arm:** the diagnostic's Search reading did not hold (0 of 236 failing-seed failures had
  no key pick-up and the box never in view).

## Evidence
- This plan: [`plans/2026-11-15_W1a_spec3_plan.md`](../../plans/2026-11-15_W1a_spec3_plan.md)
  (W1a's test, unchanged, on fresh seed ranges; criterion fixed before training).
- Behind it, in [`results.md`](../results.md): W1a (spec 2 sampled, 6/10), the W1a handoff
  diagnostic (reading Key), the handoff key-position measurement.
- Result of this plan (2026-11-15): **spec 3 passes, 7 of 10 seeds** (the extension ran after
  exactly 3 of 5); the handoff row passes on all ten, the three failing seeds fail on the own-hook
  row. [Summary](../../plans/2026-11-15_W1a_spec3_summary.md), `results.md`.
- Accepted on that result (2026-11-15), with two limits: the pass is **at the threshold** (7 of
  10; the rule needs 7), and **generality is reduced** against spec 2: spec 2 passed the own-hook
  row on all ten seeds, spec 3 fails it on three (44, 47, 50: 0.898 / 0.778 / 0.896).
- Acceptance freezes nothing. No spec-3 model is frozen or registered, and v2 §11 is not edited,
  until the freezing rule for a model that never met 0004 is decided in its own decision file.

## Affects
- If accepted: T6's rows in v2 §11, the node table ("reset with the door open, the hand empty and no
  key on the door's approach cell") and the H-7 specification table (reset hook), and the
  "One re-specification only" clause above.
- While `Proposed`: new files under `scripts/` only.
- Numbering: the decision reserved for W1b (sampled stopping rule; decision 0004, v2 §5.3), called
  0011 in [`plans/2026-11-15_roadmap.md`](../../plans/2026-11-15_roadmap.md) and the W1a plan,
  becomes **0012**.
