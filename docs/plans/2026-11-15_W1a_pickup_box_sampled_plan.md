# W1a — Does `pickup_box` work when it acts by sampling? (fixed budget, evaluation at the end)

> **Revision 2**, 2026-11-15, by the advisor chat against `main` at `7322161`. Replaces revision 1
> of this file and W1 plans v1/v2. Revision 2 applies a review made in the CLI (see the end).
> The advisor chat did not run the repo's code: where a real signature or a measurement disagrees
> with this file, the measurement wins and is recorded as a deviation.
>
> Informs: `open-questions.md` rows "Do frozen skills act deterministically or by sampling?" and
> "How is UnlockPickup finished?". **Changes no rule and no file under `src/`.** Nothing is frozen
> or registered. No decision file is needed for this step.

## Why this shape

The first half of the execution-mode question is about one primitive and can be answered with the
trainer as it is: train to the fixed budget, keep the final model (run store, PR #17), score that one
model once. No new stopping rule, so no "stopped on a lucky evaluation" problem and no change to
decision 0004. The second half (selectors, a stopping rule under sampling, `unlock_pickup`, flat PPO
with saved models) needs trainer changes and is planned after this result, as W1b.

## Known before the run

- Argmax arm: spec 2 (hash `a529aa4e74808ba0`), seeds 42–44, 1,000,000 steps, not STABLE.
- The seed-42 final checkpoint has already been scored under sampling (140/150 at the handoff). It
  is **exploratory** here and counts toward nothing.
- Amendment 1 said "no third attempt". Spec, budget and settings are unchanged here; whether this
  enters the report as the UnlockPickup result is Emir's and the supervisor's call (Q12).

## Steps

- [ ] **0. Before anything long.** `--smoke` of the training command; print measured steps/s and the
  estimated wall time for 5 × 1,000,000 steps; check the machine is on AC power.
- [ ] **1. Train.** `scripts/train_h7_primitives.py --only pickup_box --seed <s> --no-freeze
  --out-name pickup_box_w1a_s<s>` for s = 43, 44, 45, 46, 47. Existing budget (1,000,000) and
  `eval_every` (20,000); do not pass `--budget`. A run that meets decision 0004 under the existing
  (argmax) rule stops by itself; record that, it is a result. For each seed record the run-store
  `run_id`, run directory and `policy_sha256`; check `model.zip` and `policy_state_dict.pt` exist.
  For 43 and 44, report whether the evaluation series equals the old `pickup_box_s43/s44.json`;
  a difference is a deviation (machine or code since October), not a stop.
- [ ] **2. Scoring script, validated first.** New `scripts/eval_w1a.py` (evaluation only), built
  from `scripts/audit_execution_mode.py` with the same per-episode seeding of sampled runs. It has
  `--smoke`. **Validation gate:** on the saved seed-42 checkpoint it must reproduce the recorded
  handoff result of H-7a.5 (`runs/v2/pickup_box_spec2_trace.json`: same seeds and sampling seeds,
  same per-seed counts). If it does not, stop and report; do not score the new models.
- [ ] **3. Score the final models.** For each model (43–47, and seed 42 labelled exploratory):
  - **own hook:** 500 distinct env seeds 50000–50499; argmax once, sampled once (sampling seed 0);
  - **handoff:** `unlock_door → drop_held → pickup_box` on UnlockPickup from the env's reset, 500
    distinct env seeds 51000–51499; the chain argmax (as those nodes are frozen today),
    `pickup_box` argmax once and sampled once. An episode is **valid** if the chain left the
    handoff state (door open, hand empty); invalid episodes are counted and excluded.
  Per row: valid episodes, successes, rate with Wilson bounds, mean steps of successes, timeouts,
  state repeats (argmax only). The result JSON carries each model's `run_id` and `policy_sha256`.
  Confirm first that 50000–51499 is unused in `scripts/`, `src/`, `tests/`, `runs/`.
- [ ] **4. Write-up.** `runs/v2/w1a_pickup_box_sampled.json`,
  `docs/plans/2026-11-15_W1a_summary.md`, one row in `results.md`. Then stop.

## Criterion (fixed now)

- A **row passes** if the sampled success rate over valid episodes is **≥ 0.90** and at least 475
  of the 500 episodes are valid. (450/500 has a Wilson lower bound of 0.87.) Fewer than 475 valid
  episodes is a problem with the chain, reported separately; the row then neither passes nor fails.
- A **training seed passes** if both of its rows pass.
- Over seeds 43–47:
  - **4 or 5 pass → W1a passes.** Next: write W1b.
  - **0, 1 or 2 pass → W1a fails.** That is the finding. Next plan is a memory arm or an `epsilon`
    mode, not a re-run.
  - **exactly 3 pass → undecided, with one extension declared now:** train seeds 48–52 the same
    way, score them the same way, and apply **at least 7 of 10** over all ten. No other extension.
- What this rule measures (independent episodes assumed): it passes almost surely when the true
  sampled success is 0.92 or more (probability 0.93 at 0.92), and fails almost surely at 0.88 or
  less. At exactly 0.90 it passes 3% of the time. So in effect it asks for "clearly above 0.90",
  which is the margin a child needs if a chain of three is to stay near 0.80.

- **Too few valid handoff episodes (added 2026-11-15 13:46 +03:00, by Emir, before any step-3
  result was seen; the scoring run was in progress and its output unread).** If a handoff row
  has fewer than 475 valid episodes, W1a's verdict is **"undeterminable (chain)"**: neither
  pass nor fail. The rates over the valid episodes are still reported. Threshold and rule are
  unchanged; the next step is decided separately. (The chain is argmax and the env seeds are the
  same for every model, so the valid count is the same for all seeds: this happens to all of
  them or to none.)

## Not allowed

- Changing the spec, hook, budget, thresholds or the seed list after any result is seen.
- Freezing or registering any of these models, or writing into `runs/v2/0-4_pickup_box/` or
  `runs/v2/baselines/`.
- Overwriting any existing file under `runs/v2/` (every output here has `w1a` in its name).
- Calling `pickup_box` "solved" or "working"; sampled and argmax numbers in one comparison.

## Cost reporting

Each model's cost is its full training (the budget, or the earlier stop if 0004 was met): about
5,000,000 env steps for this plan, 10,000,000 if the extension runs. The six withdrawn runs
(4,804,608 env steps) are listed beside it. Evaluation episodes are counted on neither side, and
the summary says so.

## Carried to W1b (not part of this plan)

- **Flat UnlockPickup with saved models** (moved out of W1a in revision 2): 3 × 3,000,000 steps,
  not part of W1a's criterion. `train_flat_baseline.py` needs `--which ppo`; without `--out-name` it
  overwrites `runs/v2/flat_unlockpickup_ppo_s42–44.json`; and a STABLE run calls `node.freeze()`
  and saves under `runs/v2/baselines/`. W1b must state the purpose, the cost and how these are handled.
- A sampled stopping rule changes decision 0004 and v2 §5.3, not only §5.4; 0011 must say so, and
  CLAUDE.md forbids implementing a `Proposed` decision, so Emir must first allow experimental code
  behind non-default flags.
  *Note (2026-11-15, after this plan ran): decision number 0011 went to `pickup_box` specification 3
  ([`decisions/0011`](../kb/decisions/0011-pickup-box-spec-3.md)); the W1b decision described here
  becomes 0012. The text above is left as written.*
- Confirmation must use fresh sampling seeds and distinct env seeds; equal stopping opportunities
  for flat and hierarchy.
- Sampled parity between `OptionEnv` and `Executor` needs per-node generators and adapters that
  sample from the distribution themselves; SB3 `predict()` takes no generator.
- A "promote a run-store checkpoint to a frozen node" path does not exist and must be written.
- The stabilised mode is logged in meta, not enforced at load (the 9 frozen nodes are all argmax).
- `train_unlock_pickup.py` fails on the handoff row label today; the pre-flight has no mode option.
- B vs C differs in one place during training (`unlock_door`'s selector) and two in evaluation;
  with 3 seeds per arm, require a gap of at least 2 seeds or fall through to the step-cost criterion.

## What changed from revision 1

From the CLI review: the flat step was not runnable as written and would have overwritten recorded
results, so it moved to W1b; the handoff criterion is now a rate over valid episodes; the pass rule
at 150 episodes left a wide undecided zone (4% pass at a true 0.90, and seed 42 sits at 0.93), so
rows now use 500 distinct seeds and the "exactly 3 of 5" case has one pre-declared extension; the
scoring script gets a smoke mode and a validation gate; each model is tied to its run id and policy
hash; the base commit is current `main` (the 21 commits since `d7fe1da` only add GRU files); a
wall-time estimate is required before the long runs.
