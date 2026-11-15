# W3 — Can a selector learn a choice that depends on the observation, when a wrong pick is final?

> Written 2026-11-15 by Mehmet P. with Claude Code, against `fix/audit-followups-main` at `d031676`.
> Node, budget, seeds, gates and criterion below are fixed before the first training run
> (decision 0010). Smoke runs are not runs of this plan.
>
> Informs: `open-questions.md` "Fix or replace the choice task (`unlock_matching`)" (owner Emir) and
> the answered row "Can a composite condition its child choice on the observation?"; Q8; roadmap W3,
> claim part A2. **Changes `scripts/train_choice_task.py` only.** Nothing is frozen or registered.

## What was wrong (H-6 summary §4, §10)

`{1,2} unlock_matching` has 6 decisions. A wrong first key is recoverable: `get_key_<colour>`
drops and swaps in 43 of 50 episodes, so the goal-only arm measured recovery, not choice, and the
reward arrived four decisions after the choice.

## The node

`{1,4} unlock_matching_2d`: the same env (DoorKey-6x6), the same reset hook (`two_keys`), the same
three frozen children (`{0,6} get_key_yellow`, `{0,7} get_key_blue`, `{0,8} open_door_tk`), the
same goal and timeout, and **`max_decisions = 2`**. The only successful episode is "matching key,
then the opener"; a wrong key or a wasted pick ends it. The reward follows the choice by one
decision. No new env, hook or primitive.

The children's held-out scores are 0.94 / 0.86 / 0.94, so the ceiling is about 0.81–0.88 and the
node cannot be STABLE under 0004. STABLE is not the criterion and the node is not a library node.

## Gates before training (50 held-out seeds, `--stage gate2d`)

| Gate | Passes if |
|---|---|
| A wrong key is final (the premise that was false in H-6) | forced wrong key, then the opener: success ≤ 1/50 |
| Ceiling | matching key, then the opener: success ≥ 0.80 |
| The mask does not give the answer | the goal-only mask at decision 1 is the same set for both door colours |
| Colour balance | each colour in at least 30 % of the episodes |

If a gate fails, training does not start and the failure is reported.

## Runs

Training seeds 42, 43, 44; 51,200 decisions = 100 policy updates (`n_steps` 64 × 8 envs), as H-6.8;
`--no-freeze`. Primary arm: goal-only masking. The strict-mask arm runs from the same command and
is reported; it is not part of the criterion.

## Criterion, fixed in advance (roadmap W3)

On 500 fresh seeds (80000–80499), goal-only arm, per training seed:
first-key accuracy ≥ 0.90 over the episodes in which a key is picked, each door colour ≥ 0.80, and
no key picked in at most 5 % of the episodes. **W3 passes if at least 2 of 3 training seeds meet it.**

Reported beside it on the same fresh seeds: success of the learned selector, of random over
available children, of "always yellow" and "always blue", and of the scripted ceiling.

| Outcome | Reading |
|---|---|
| Passes | A selector learned an observation-conditioned choice that no mask gives it (A2) |
| Fails | A2 stays "order discovery"; reported as the negative result |

## Limits

One task, two colours, three seeds. The colour is one channel value in one of 49 cells. The
selector sees the same 147-dim observation as before.

## Not in this plan

A new env with a physically irreversible choice; goal-predicate bits in the selector's input
(v2: policies see only the observation); retraining the key primitives.
