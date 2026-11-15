# W1b-1 — Freeze one spec-3 `pickup_box`, train `{2,2} unlock_pickup` with the rule as it stands

> **Revision 3**, 2026-11-15, by the advisor chat; written against `main` at `128d2b6`, checked by
> the executor against `b59bc2c` (0011 `Accepted`; PR #29 touches none of this plan's files).
> Revisions 2 and 3 apply the plan-executor's two reviews (lists at the end). The
> advisor chat read the repo and did not run its code: where a real signature or a measurement
> disagrees with this file, the measurement wins and is recorded as a deviation.
>
> Informs `open-questions.md` "How is UnlockPickup finished?" and roadmap rows B1, B3.
> It freezes a node that v2 calls ABANDONED, so **step 0 is two decision-file changes and Emir's
> explicit go-ahead in the session**. No file under `src/` changes. Nothing is written to a
> tracked registry path while 0014 is `Proposed`.
>
> Emir's answers in the advisor chat (2026-11-15): 0011 accepted (he sets the status line
> himself); freezing rule = W1a test passed **and** Wilson 95 % lower bound of the argmax handoff
> rate ≥ 0.95; model = smallest-numbered seed that meets the rule.

## The one question

With one spec-3 `pickup_box` frozen as a child, does `{2,2} unlock_pickup` (T7) reach STABLE under
decision 0004 **unchanged**, trained by the existing composite trainer in the existing mode
(every node argmax)?

Why this shape: it needs no trainer change and no new stopping rule. If it passes, UnlockPickup has
a composite scored in the same mode and on the same 50 held-out seeds as flat PPO's recorded 0/50.
Sampling (W1b-2) then becomes an improvement question, not a blocker.

**What this plan does not settle.** Roadmap decision 4 ("retrain flat UnlockPickup with saved
models", agreed, carried to W1b) is **not cancelled and not done here**; it stays with W1b-2.
Consequence, stated in the write-up: the flat comparison rests on the 50 held-out seeds only; the
fresh 500-seed row has no flat counterpart, because no flat model was saved.

## Fixed before any run

**The child.** Spec 3, training seed **45**: run `20261115T125406-03312e12d82048218c20b91fa6cad0fc`,
`policy_sha256` `b05ae151ac43c398…`, `spec_hash` `6c6428043d97ac75`.

| Check (from `w1a_spec3_sampled.json`, ranges 52000–53499) | Seed 45 |
|---|---|
| (a) W1a test, sampled: own hook / handoff | 459/500 = 0.918 / 488/488 = 1.000 → pass |
| (b) Argmax handoff | 485/488 = 0.994, Wilson lower bound 0.982 (rule: ≥ 0.95) |
| Argmax own hook (known limit, goes into the node's meta) | 305/500 = 0.610, every failure a state repeat |
| Decision 0004 during training | not met; budget exhausted (v2 §9: ABANDONED) |

Seeds meeting (a) and (b): 45, 46, 48, 49, 51, 52 (6 of 10). Seed 43 passes (a); its lower bound
is 0.928. Seed 44 meets (b) (486/488, lower bound 0.985) and fails (a) (sampled own hook 0.898).

**Why condition (a) is kept, although this parent only uses the child argmax at the handoff.**
(a) is the test spec 3 was accepted on (0011); a model that failed it is not an instance of "the
specification that passed". It is also the only evidence that the skill does anything away from
this one parent. Without (a) the rule would select seed 44. **The limit of that second argument,
which 0014 must state in so many words:** (a) is a *sampled* measurement (own hook 0.918), while
the child is frozen and run *argmax*, where the same model scores 0.610 from its own hook. So (a)
does not show that the node as frozen here is general; it shows that it would be if run sampled
(W1b-2's question). In this plan (a) rests on the first argument alone.
**Emir confirmed on 2026-11-15, after seeing that (a) removes seed 44, that (a) stays**: the rule
was chosen with (a) before its effect on the selection was looked at, and is not changed afterwards.

**Why the threshold is 0.95 and not 0.90.** The parent must score 46/50 (0.92) on a fixed seed set.
Its ceiling is about (chain leaves the handoff state: 488/500 and 491/500 measured) × (child's
argmax handoff rate). With seed 43 that is ≈ 0.93, and a true 0.93 gives 46+/50 only about 73 % of
the time; with a 0.90 child the ceiling is ≈ 0.88. With seed 45 it is ≈ 0.97 (about 98 %).

**Disclosure.** The ten argmax handoff rates were known when this rule was written, so the rule
and the seed it selects are not blind. What is still unseen is every `unlock_pickup` result. The
write-up says this in the same paragraph as the result.

**The parent.** T7 exactly as `h7_nodes.build_unlock_pickup` builds it (timeout 288,
`max_decisions` 8, goal-only mask), children `{1,1} unlock_door`, `{0,9} drop_held` (canonical,
frozen) and the frozen seed-45 `pickup_box`. Budget 40,960 decisions, `eval_every` 512, 8 envs,
50 evaluation episodes, `n_steps` 64: the composite defaults. Training seeds **43, 44, 45**.

**Not independent.** The three training seeds share the fixed stopping seeds (10000–10049) and the
same frozen children. They test whether the selector learns the order, not whether the children
are good enough; the second is close to one draw. Hence step 2's ceiling gate and no extension.

**Fresh seeds.** Env seeds **54000–54499**. Checked unused by the executor in `scripts/`, `src/`,
`tests/`, `docs/` and the JSON files under `runs/` (`runs/training/` included); log files were not
searched.

## Steps (stop after each and report)

- [x] **0a. 0011 is `Accepted` on `main`** (done, `b59bc2c`, PR 30). Status line set by the
  executor at Emir's instruction in the session, merged by Emir ("7/10, at the threshold" and
  "generality reduced" under Evidence). 0011's text says no spec-3 model is frozen
  while it is open, so **step 1 does not start until this is merged**. The session pulls `main`.
- [ ] **0b. Decision file `0014-freeze-without-live-stable.md`, `Proposed`.** (0012 stays reserved
  for the stopping rule under sampling, as 0011, `status.md`, the roadmap and both W1a plans say;
  0013 is taken.) It must contain:
  - the rule: a primitive whose training never met 0004 may be frozen from its fixed-budget final
    model if (a) it passed a two-row test whose criterion was fixed before training (here W1a's)
    and (b) the Wilson 95 % lower bound of its argmax success from the parent's real handoff is
    ≥ 0.95 on ≥ 475 valid fresh episodes; the mode it was accepted in and its weak rows go into
    `meta.json`; why (a) is kept (above);
  - **everything it sets aside:** v2 §5.3–5.4 ("STABLE → FROZEN"); v2 §9 ("budget exhausted →
    ABANDONED, kept in the log, never used"); the v2 §13 run-persistence amendment
    ("`freeze()` requires STABLE"); decision 0004; the CLAUDE.md hard rule of the same wording;
  - scope: primitives only, argmax only;
  - "Evidence: this plan"; the disclosure above;
  - **what happens to the checkpoint:** while `Proposed` it lives only under the git-ignored
    `runs/w1b/`; if 0014 is rejected that directory is deleted and W1b-1's results stay in
    `results.md` as the record of a rejected experiment; if accepted, the acceptance PR writes it
    to a tracked path;
  - the sentence "This experiment runs while 0014 is `Proposed`, with Emir's go-ahead, under
    `scripts/` only."
  **Stop. Continue only after Emir says so in the session.**
- [ ] **1. Promote the model.** New `scripts/w1b_freeze_pickup_box.py`, `--smoke` included.
  - Load seed 45 the way `eval_w1a_spec3.load_spec3_model` does (spec equality, sha, held-out
    reproduces).
  - **Gate P1:** its handoff rows on 53000–53499 reproduce the record episode for episode
    (argmax 485/488, sampled 488/488).
  - **Gate P2:** the script computes the rule over all ten seeds from **both** result files
    (`w1a_spec3_sampled.json` for 43–47, `w1a_spec3_sampled_ext.json` for 48–52) and applies
    "smallest seed that meets (a) and (b)"; it must return 45.
  - Set STABLE (one line, citing 0014), `freeze()`,
    `registry.save(run_dir=runs/w1b/0-4_pickup_box_spec3_s45)` (**git-ignored**). Meta carries:
    source `run_id`, sha, `steps_trained` 1,001,472, chain steps in training resets 372,063,
    `met_0004: false`, `frozen_under: "0014 (Proposed)"`, `execution_mode: argmax`,
    `accepted_for_parent: "{2,2} unlock_pickup"` (0014: acceptance is per parent), the table above.
  - The checkpoint is git-ignored, so its identity goes into a **tracked** file,
    `runs/v2/w1b_pickup_box_freeze.json`: source `run_id`, `policy_sha256`, `spec_hash`, the rule
    computed over all ten seeds, and the results of gates P1–P3.
  - **Gate P3:** load it back with `attach_sb3_primitive_policy` +
    `registry.load_into(run_dir=…)` (`load_frozen_primitive` takes no `run_dir`): FROZEN, same
    sha and `spec_hash`, 50 handoff episodes identical.
  - `runs/v2/0-4_pickup_box/` (spec 2, seed 42, TRAINING) is not touched; other scripts load it.
- [ ] **2. Composite script and ceiling gate.** New `scripts/w1b_unlock_pickup.py`.
  - Loads `unlock_door` and `drop_held` from the canonical registry and `pickup_box` as in gate
    P3 (not `load_h7_primitive("pickup_box")`, which loads spec 2); all must be FROZEN; builds T7;
    prints its new `spec_hash`.
  - Reuses `h7_composite.run` with `--no-freeze --skip-strict` forced. `run` does not return the
    trained node and calls `build()` several times: **the wrapper keeps the node from the first
    `build()` call**, which is the one `run` trains. `run` writes its output
    unconditionally, so **the wrapper** refuses to start if the output file exists, `--smoke`
    included (smoke outputs get their own names and are deleted afterwards). Known and accepted:
    `run` still calls `node.freeze()` in memory on a STABLE run; nothing is saved to the registry.
  - `chain_ceiling` is built from the recorded rows (488/500 × 485/488), not from `f29()`;
    `h7_preflight.json` is not rewritten.
  - **Ceiling gate (evaluation only):** the scripted sequence unlock_door → drop_held → pickup_box,
    argmax, on the stopping seeds 10000–10049 and the held-out seeds 20000–20049. If the stopping
    set gives fewer than 46/50, **stop: W1b-1 fails with reading "chain"**, and nothing is trained.
  - `--smoke`, then measured decisions/s and env steps per decision with a wall-time estimate.
- [ ] **3. Train and score in one process.** Seeds 43, 44, 45, outputs
  `runs/v2/w1b_unlock_pickup_s<seed>.json`. Record `run_id`, `policy_sha256`, `stable`,
  `decisions_to_stable`, `env_steps_own`, held-out 50.
  The fresh-seed row is scored **in the same process, right after `run` returns**, on the node
  the wrapper kept (no run-store loader for composites exists, and none is written here):
  argmax, once, on 54000–54499; successes, Wilson bounds, mean steps, timeouts.
  `run` writes its own file and cannot be extended, so this row goes into a second file,
  **`runs/v2/w1b_unlock_pickup_s<seed>_fresh.json`**, with the composite's `run_id` and
  `policy_sha256` (which must equal the run-store manifest's) and the child's `policy_sha256`.
  **If the process dies after training and before the fresh file is written** (rule fixed now):
  the same seed is run again once, same command, output `…_s<seed>_rerun.json`. The rerun counts
  only if its stopping-evaluation history and `policy_sha256` equal the first run's exactly; the
  fresh row is then scored from it. If they differ, or it dies again, that seed is "not
  determined" and counts as not passing. Nothing is overwritten.
- [ ] **4. Write-up.** `docs/plans/2026-11-15_W1b-1_summary.md`, rows in `results.md` and
  `status.md`, the cost table below. Then stop. 0014 stays `Proposed`.

## Criterion

- A **training seed passes** if its run is STABLE under 0004 within the budget **and** its fresh
  row is ≥ 0.90 (450/500).
- **W1b-1 passes** if at least 2 of 3 seeds pass. No extension, no second child model.
- Reported beside it, not part of the verdict: held-out 50 (the seeds of flat's 0/50), the ceiling
  gate's counts, the C1 evaluation arms, mask divergence and selection sequences from `run`
  (with `--skip-strict` the strict-mask arms are evaluations of the goal-only policy).

**Addendum before step 3** (2026-11-15, 19:40, after step 2; no training result seen, only the
10 episodes of step 2's smoke run on seed 42). On the same 500 fresh seeds (54000–54499) the
control "random over available children" (`RandomOverAvailable`, goal-only mask, children argmax)
is measured: successes and mean env steps. It is **not part of the verdict**. Its difference from
the learned selector is reported as step efficiency. Why: in the smoke run random selection
already scored 0.70–0.90 (10 episodes), so a passing W1b-1 shows reuse (`unlock_door` untouched,
a task flat PPO did not solve), not that the selector learned something the mask does not give
it; the write-up says so.

## What follows (fixed now)

- **Passes.** Allowed wording: "`{2,2} unlock_pickup` is STABLE under 0004 (argmax) on k of 3
  seeds, with a child frozen under 0014 (`Proposed`)". Emir decides 0014. Only after that, in a
  separate PR: tracked registration of the child and of the parent from the smallest passing
  training seed, v2 §9 and §11 (T6/T7), and where the spec-2 seed-42 record moves. Then W2;
  W1b-2 (sampling, 0012, flat with saved models) is planned as an improvement question.
- **Fails.** Reading by this table, then W1b-2. No re-run with another child seed, no larger
  budget, no fourth `pickup_box` design.

  | Observation | Reading |
  |---|---|
  | Ceiling gate below 46/50 on the stopping set | chain (children, argmax) |
  | Gate passed; ≥ 2 seeds end more than 5/50 below the gate's count | selector (order not learned in budget) |
  | Gate passed; otherwise | undetermined; both reported |

## Not allowed

- Changing the child, rule, thresholds, budget or seed lists after any result is seen.
- Any file under `src/`; any frozen checkpoint under a tracked path while 0014 is `Proposed`;
  writing to `runs/v2/0-4_pickup_box/`, `runs/v2/2-2_unlock_pickup/`, `runs/v2/baselines/` or
  `h7_preflight.json`; overwriting any existing file under `runs/v2/`.
- Sampling anywhere; the strict-mask trained arm; the option-set baseline. Flat is not retrained
  **in this plan** (decision 4 stands, with W1b-2).
- "Solved" or "works"; sampled and argmax numbers in one comparison.

## Cost reporting (rule fixed now)

Two framings in one table, neither presented alone; evaluation episodes on neither side.

| | Env steps |
|---|---|
| **Recipe, new nodes for this task:** `drop_held` 30,000 + `pickup_box` (1,001,472 + 372,063 chain) × 10/6 + `unlock_pickup` own | ≈ 2,319,000 + own |
| The same without the selection factor | 1,403,535 + own |
| Library reused, reported separately: `env_steps_children` of `unlock_door` | as recorded |
| **Research total on `pickup_box`:** six October runs 4,804,608 (spec 1, withdrawn: 3 × 600,064; spec 2 argmax, seeds 42–44: 3 × 1,001,472) + W1a spec 2 10,014,720 + spec 3 10,014,720 + chain 3,715,701 | 28,549,749 |
| Flat PPO, UnlockPickup: per run / three runs, 0/50, censored | > 3,000,320 / 9,000,960 |

The headline comparison with flat is the recipe row **with** the 10/6 factor (6 of 10 runs met the
freezing rule), per run against flat's per-run figure. The research total stands in the same
table. `children_env_steps` does not count chain steps; they are added by hand, on their own line.

## What changed from revision 2 (the executor's second review)

- The fresh row and the policy hashes get their own named file; the wrapper keeps the first
  `build()` node; the child's identity is recorded in a tracked file; a rule for a process that
  dies between training and fresh scoring; the limit of condition (a)'s second argument is stated.

## What changed from revision 1 (the executor's first review)

- 0011 must be `Accepted` in the file before anything is frozen: step 0a, and step 1 waits for it.
- The freezing rule is decision **0014**; 0012 keeps its reserved meaning.
- A tracked `status: frozen` checkpoint under a `Proposed` decision is gone: the checkpoint is
  git-ignored until 0014 is decided, and 0014 says what happens to it either way.
- 0014's list of rules set aside now includes v2 §9 (ABANDONED) and the §13 amendment.
- Revision 1 said flat need not be retrained and forbade it, silently reversing roadmap
  decision 4. Now: out of scope here, not cancelled, and the limit on the comparison is stated.
- Code: overwrite check moved to the wrapper (smoke included); child loaded with `run_dir`;
  gate P2 reads both result files; fresh-seed scoring happens in-process (no composite loader);
  the 4,804,608 line is relabelled (half of it is not "withdrawn"); `--no-freeze` still freezes
  in memory, noted.
- New paragraph: why condition (a) is kept although it removes seed 44.
