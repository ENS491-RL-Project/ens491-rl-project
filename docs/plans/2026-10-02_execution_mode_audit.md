# Execution-mode audit — what deterministic execution does across the library (evaluation only)

> Written 2026-10-02 17:40 (+03) by the advisor chat, against commit `0054c24`
> (`docs/plans/2026-10-02_H7_summary.md` §10, `runs/v2/pickup_box_spec2_trace.json`).
> The advisor chat cannot run the repo's code: where a real function signature or a measurement
> disagrees with this file, the measurement wins and is recorded as a deviation.
>
> **Nothing in this file authorises training or a rule change.** No policy is trained, continued or
> frozen. v2, the Executor, the stabilisation rule and every default stay as they are.
> `{0,4} pickup_box` stays TRAINING.

## 1. Why

H-7a.5 found that all 20 deterministic failures of the spec-2 `pickup_box` checkpoint at the handoff
are state repeats, and that the same checkpoint with sampled actions scores 140/150. v2 §5.3 [F17]
evaluates and executes every policy deterministically. PPO trains a stochastic policy, and under a
7×7 view with no memory its argmax can cycle between two observations.

Whether that rule should change is a design decision for the author and the supervisor. Before
they take it, two things have to be known that the files do not show:

1. What the two execution modes do to the nodes that are already frozen.
2. What they do to flat PPO, which has also been scored deterministically. A hierarchy number
   under sampling may only ever stand beside a flat number under sampling.

## 2. Seeds

A fresh set of 50, base **40000**. Confirm by search that no script or result file uses
40000–40049; if one does, take the next unused multiple of 10000 and record it. The held-out set
20000–20049 has now been looked at in both modes for one checkpoint and is not reused here.
Stochastic episodes use sampling seeds 0, 1, 2 with the per-episode torch seed H-7a.5 defined.

## 3. Measurements

- [ ] **E.0** State in the summary what `STOCHASTIC` execution does in `src/skills/executor.py` for
  a composite: which levels sample (the selector, the primitives, both). Run what exists. Do not
  modify the Executor.
- [ ] **E.1 Frozen nodes under their own hooks.** Each of the 9 frozen nodes, on each of its
  `env_ids`: 50 deterministic episodes and 3 × 50 stochastic. Per row: success rate with Wilson
  bounds, per sampling seed, mean steps of the successes in each mode, and for the deterministic
  failures the number that contain a state repeat (the H-7a.5 definition).
- [ ] **E.2 Handoff rows.** Every gating F29 row that exists under `runs/v2/` (list them with their
  source file), re-run on these seeds in both modes. Plus, labelled diagnostic, the scripted
  sequence `unlock_door → drop_held → pickup_box` (TRAINING checkpoint) on UnlockPickup from the
  env's own reset, in three variants: all three deterministic; only `pickup_box` sampled; all three
  sampled.
- [ ] **E.3 Flat PPO.** For each of the twelve flat PPO runs (DoorKey-6x6, Unlock, UnlockPickup,
  BlockedUnlockPickup × seeds 42, 43, 44): if its model was saved, both modes on the same seeds,
  from the env's own reset, against the task's goal. If it was not saved, say so, do not retrain,
  and report from its training log the highest rollout success or mean episode reward it ever
  reached (training rollouts are sampled).
- [ ] **E.4 The table.** `runs/v2/execution_mode_audit.json` and §12 of the H-7a summary:
  one row per item above with both modes side by side. Two flags, fixed here before the run:
  - a row is **mode-sensitive** if the pooled stochastic rate differs from the deterministic rate
    by 0.10 or more, in either direction;
  - over all deterministic failures of frozen nodes in E.1 and E.2, the share that are state repeats.

  And the UnlockPickup comparison stated once per mode on these seeds: scripted sequence against
  flat PPO, deterministic against deterministic, sampled against sampled.

## 4. Wording rules

- A stochastic success rate is not a gate and freezes nothing.
- No sentence proposes changing v2 or says what should be built.
- Never write that `pickup_box` "works" or is "solved".
- Where a flat model is missing, the comparison in that mode is "not available", not zero.

## 5. Verify

Record counts match the plan; `runs/v2/regression.json` unchanged at 9 of 9, delta 0.0; no file
under any `runs/v2/<node>/` checkpoint directory changed; suite green; one commit for this file
(unchanged), one for the script and result, one for the documents; origin = HEAD.

Then stop.
