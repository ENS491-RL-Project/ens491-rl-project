# Open questions

Decisions that are not made yet, and what each one blocks. When one is decided, write a file in
[`decisions/`](decisions/) and remove it here. Last reviewed: 2026-11-15.

## Open decisions

Each is settled by an experiment where possible ([0010](decisions/0010-decide-empirically.md)).

| Question | Why it matters / what it blocks | How to settle | Owner | Source |
|---|---|---|---|---|
| Do frozen skills act deterministically or by sampling? | Deterministic play loops in partially observed rooms; under sampling every frozen node scored 150/150 in the audit (`solve_doorkey` 149/150), but `open_door` rose only 0.74→0.90 in E.2. Blocked `pickup_box`/UnlockPickup until decision 0014 settled it for argmax (W1b-1); still open for the train/eval rules and W1b-2. | Audit exists (evaluation only); still to run: training + final evaluation under each mode on the same seeds | Emir | [`plans/2026-10-02_execution_mode_audit.md`](../plans/2026-10-02_execution_mode_audit.md) |
| Phase B growth (composite-first heuristic) | Needed for "compose while learning" (0006); Phase A is temporary (0005). | G-1…G-3, v2 §8 | Emir | v2 §8, §12 |
| Lateral connections: forward transfer at all? | Decides whether PN is part of the claim (0003, 0006). W7 (C-2, 2026-11-15) found no measurable benefit on one pair (`go_to_goal` ← `open_door`), so PN is not claimed now; open only in that a second pair has not been tried. | Emir decides whether a second pair is worth trying (W7 plan, "After"); no tuning or re-run of W7's pair | Emir | [`decisions/0003`](decisions/0003-progressive-networks-off-critical-path.md), [W7 summary](../plans/2026-11-15_W7_summary.md) |
| Fresh test seed set for final evaluation | Only item of the evaluation-rigour list without a PR. Already taken: 50000–59999 (reserved for W1, [roadmap](../plans/2026-11-15_roadmap.md)) and 60000–60499 (W7's final score, [plan](../plans/2026-11-15_W7_pn_lateral_plan.md)); a new set must avoid both. | Define once, never used in training or earlier evaluation | Mehmet P. | team chat, 2026-11 |
| When does the GRU's task id select the root node? | v2 §6.2 designs it; code ignores the prediction. Offline accuracy is known (PR #18: 85 % task-episode macro at N ≥ 10 on 9 known tasks); switching latency, old-task retention and the task registry are not built or measured. | Integrate with a task registry (R-3) and measure end-to-end success and switching latency | Elif + Emir | PR #18, [`results.md`](results.md), v2 §6.2 |
| Does the flat baseline need memory? | A memoryless PPO may be an unfair baseline in partially observed tasks. **Answered for one variant by W4(a) (2026-11-16):** a flat PPO with a 64-unit LSTM is also 0/50 on UnlockPickup (3 seeds, 3M steps, every stopping evaluation 0) and slower on DoorKey (550–750k against 300–450k steps). Open: a frame-stack arm, tuned recurrent hyperparameters | Emir decides whether another memory variant is worth running; no re-run of W4(a)'s arm | Emir | [`decisions/0006`](decisions/0006-claim-compose-and-reuse.md), [W4(a) summary](../plans/2026-11-16_W4a_summary.md) |
| Fix or replace the choice task (`unlock_matching`) | A wrong key pick is recoverable (43/50 swaps), so the task cannot test conditioned choice. | Redesign so a wrong pick is unrecoverable; rerun | Emir | [`plans/2026-10-02_H6_summary.md`](../plans/2026-10-02_H6_summary.md) §4 |
| Should a frame that re-selects without an env step be closed early? | With per-frame caps (0009) the caps multiply down the stack. Under the strict mask a dead end (no applicable primitive) re-selects with no env step: one `act()` call made 266,304 selections (4.7 s) at depth 5, about 17M at depth 6 (extrapolated); the old global cap stopped it at 64 by construction. Goal-only masking is not affected; only the C1 ablation uses the strict mask today. A rule change touches v2 §4 🔒 and needs a decision file | Measure `act()` cost and outcome on strict-mask graphs of depth ≥ 4, with and without a no-progress rule | — | PR #15 review (2026-11-12), [`results.md`](results.md), [`decisions/0009`](decisions/0009-max-decisions-per-frame.md) |
| AE threshold: updated during use (Module_Interfaces) or calibrated then frozen (v2 §6.1)? | The two docs conflict; affects novelty detection and Track R. | Compare false-positive/negative rates of both on the curriculum tasks | Emir | Module_Interfaces AE section, v2 §6.1 |

## Supervisor questions

The supervisor is shown the evidence and consulted on scope and framing; design questions above are settled by experiment.
Q1–Q9: [`docs/SUPERVISOR_QUESTIONS.md`](../SUPERVISOR_QUESTIONS.md). Q10 (the central claim), Q11 (skills that do not transfer)
and Q12 (execution mode) were in the October brief, which was never committed.
**The 6 October meeting answered some of these; the answers were not written down.** Record them here.

| Q | Topic | Answer (supervisor's words) | Date |
|---|---|---|---|
| Q1 | Who defines a task | | |
| Q2 | DAG hierarchy | | |
| Q3 | New composite vs new primitive | | |
| Q4 | Continual-learning claim / PN | | |
| Q5 | Initiation sets | | |
| Q6 | Evaluation curriculum | | |
| Q7 | GRU continual training | | |
| Q8 | How much is masking | | |
| Q9 | Retrain the old column | | |
| Q10 | Central claim | | |
| Q11 | Skills that do not transfer | | |
| Q12 | Execution mode | | |

## Empirical questions (answered by experiment, not by decision)

Moved unchanged from Project_State ([archived](../archive/ENS491_Project_State_2026-11-15.md)). Section numbers refer to v2.

| Topic | Alternatives | Criterion | v2 |
|---|---|---|---|
| AE architecture | Undercomplete → Sparse AE → VQ-VAE | Separation of known vs novel tasks (ROC-AUC); ref. Meyer et al. 2024 | §6.1 |
| AE scope | Single global AE → one AE per task (base paper) | Novelty accuracy after each new task | §6.1 |
| AE threshold | Fixed per task (mean + kσ) → statistical test (KS / Wasserstein, ref. SWOKS) | False positive / negative rate | §6.1 |
| GRU sequence length | N = 5 → 20 → 50 | Accuracy vs switching latency | §6.2 |
| GRU continual training | Retrain on accumulated sequences → EWC on the GRU | Old-task accuracy after each new task | §6.3 |
| Lateral sources | Most recent primitive → all earlier primitives → nearest by AE latent | Steps to STABLE (forward transfer); only worth running if some pair shows transfer, which W7 did not (0013) | §7 |
| Execution mode of frozen nodes | Deterministic → stochastic | Success rate, loop frequency | §5.4 |
| SMDP discounting | Per-decision γ → exact γᵏ | Composite learning speed | §5.2 |
| Stabilization thresholds | SR 0.90 × 2 evals (default) → stricter / looser | Downstream reuse quality vs training cost | §5.3 |
| Growth order | Composite-first → primitive-first | Unnecessary nodes created, total steps | §8 |
| Primitive training distribution | Single layout → mix of all layouts where the goal is defined | Success inside parent tasks | §11 |
| Composite depth | T9a (level 2, primitives) → T9b (level 3, reuses `unlock_pickup`) | Steps to STABLE | §11 |

## Answered or partly answered by experiment

Rows up to "Stabilization thresholds" moved unchanged from Project_State (section "Answered by H-6", 2026-10-02); later rows were added here. Measurements are in [`results.md`](results.md).

| Question | Answer |
|---|---|
| **Q8 — how to frame the masking contribution** | On the §11 curriculum, success rate **cannot** separate composition from masking at any depth: every composite's subgoal order is strict, the initiation mask encodes exactly that order, and a random policy under it ties the learned composite (0.98 at depth 1 in H-5, 1.00 at depth 2 in H-6.3). Stronger than "ties": under the strict mask the *randomly initialised* selector already scores 1.00, so the mask does not accelerate learning, it removes the need for it. The report states that as a property of the curriculum and quantifies the learned policy's contribution two other ways: the **divergence count** (0 of 20 withheld choices taken, so the ordering was discovered rather than given) and **efficiency** (14.2 env steps vs 113–143 for random-over-available, 8×). A dedicated choice task was built to escape this; see H-6 summary §4 for its result and §9 for why the first attempt at it did not count |
| **Primitive training distribution** | The mix matters, measured both ways. The three H-4 primitives, trained on two layouts with randomized poses, transfer into DoorKey at 1.00 from real handoff states. `{0,2} open_door`, trained on a narrower mix, scores **0.83** in the two-key variant (re-measured in H-6.8 under the corrected hook) and had to be replaced by `{0,8} open_door_tk`, trained on the parent's own distribution, which scores 0.98 there |
| **Composite depth** | First data point, re-measured in H-6.8 over distinct-policy evaluations. `{2,1} solve_doorkey` (depth 2) is STABLE in **2** policy updates (the reports said 3: the counter included the rollout the stopping callback discards; fixed 2026-11-05) / 48,887–55,659 own env steps; `{1,B} flat_doorkey_options` (depth 1, same three skills) in 3–4 updates (old counter, likely one too many; not re-measured) / 64,000–80,951. Depth wins by ~1.3–1.4× on own env steps and loses with children counted (661–667k for `solve_doorkey` against 544–561k for the option set), both at held-out 1.00 on 3 seeds. The decisive comparison is still T9a vs T9b in H-7 |
| **How is UnlockPickup finished?** | **Decided** by [0011](decisions/0011-pickup-box-spec-3.md) (`pickup_box` spec 3) and [0014](decisions/0014-freeze-without-live-stable.md) (freeze a budget-exhausted primitive on a pre-fixed test plus an argmax handoff row), both Accepted 2026-11-15. W1b-1: spec-3 seed 45 frozen; `{2,2} unlock_pickup` STABLE under 0004 on 3 of 3 seeds (one result in effect), held-out 49/50, fresh 486/500; seed 43 registered. [Summary](../plans/2026-11-15_W1b-1_summary.md) |
| **Does reuse pay?** | **Partly answered by W1b-1 (2026-11-15):** UnlockPickup now has a frozen composite, `{2,2} unlock_pickup` (held-out 49/50; flat PPO 0/50 on the same seeds after 3,000,320 env steps per run, censored). Recipe cost about 2.38M env steps per run with the 10/6 selection factor (2.79M with the reused `unlock_door`); research total on `pickup_box` 28.5M ([summary](../plans/2026-11-15_W1b-1_summary.md)). T9a vs T9b (W2) is not run. State before W1b-1: H-6 could not answer it, and H-7a stopped before it could: `{0,4} pickup_box` failed to stabilise under two specifications (4.8M env steps over six runs), so UnlockPickup has no frozen composite. What exists: R1 holds on its one row (`unlock_door` 0.98 untouched on UnlockPickup); through DoorKey the hierarchy's cumulative cost is 561,756–570,536 env steps with its composites re-measured at `n_steps` 64 (an update size flat PPO never got) and **773,955 as built** (the frozen checkpoints, `n_steps` 256), against flat PPO's 650,000–900,000, under five caveats (`runs/v2/h7_amortisation.json`); on UnlockPickup both sides are lower bounds (> 1.59M against > 3.65M; flat PPO never scored); with the withdrawn `pickup_box` specification counted, > 2.19M against > 3.65M, so the hierarchy has still spent less over these three tasks, and neither has a stable solution for the third (close-out K.3). H-7 summary §3–§4 |
| **Can a composite condition its child choice on the observation?** | **Not shown; no run learned it within 100 updates, and the cause is not established.** `{1,2} unlock_matching`, 3 seeds × 2 mask conditions × 100 policy updates: every goal-only run converges to "always pick yellow" (50/50 episodes). Under the strict mask seed 42 picks blue first in 49/50, but seeds 43 and 44 (blue first in 29 and 32 of 50) were leaving the one-key plateau when the budget ran out — stopping success near 20/50 until about update 50, then 25–30/50; seed 43 picks the right key in 36/50 held-out episodes. A5 fails on all three seeds. H-6.9 (H-6 summary §10): the door is in view and the strict mask is correct in 400/400 resets; the selector's own network, trained supervised, reads the colour at 0.940 from 1,000 examples and 0.997 from 4,000 (logistic regression 0.91). Between the pre-set thresholds, so no verdict: not a representation limit, not cheap either. The task's own premise is also broken — a wrong pick is recoverable, because `get_key_<colour>` drops and swaps 43/50 times when handed a full hand. Fixable; see H-6 summary §4. **Exploratory arms (H-6 summary §10.1):** 4 more strict-mask runs (seed 42 unchanged; seeds 42–44 with `ent_coef` 0.05), 100 updates each, none stable; the final policies' probe gap p(yellow|yellow door) − p(yellow|blue door) is −0.01 / −0.04 / +0.08 / +0.05. Seed 42 unchanged collapsed on blue mid-run (0.97–0.98), ended at 0.87–0.88; the entropy bonus prevented collapse. **Counted in deterministic play over all seven strict-mask runs (close-out K.1, H-6 summary §10.1, `runs/v2/choice_conditioning.json`): three of seven, all on seeds 43 and 44, show a weak dependence of the first key on the door's colour (gaps +0.43 / +0.35 / +0.32, Fisher p 0.004 / 0.021 / 0.021, uncorrected for seven runs); seed 42 shows none in three runs; no run learned the choice (best first-key accuracy 0.72); the entropy bonus is not shown to matter, because seed 43 shows the dependence at 0.01 as well** |
| **Stabilization thresholds** | One further constraint, added in H-6.8: `eval_every` must be at least one policy update (`n_steps × n_envs`), or "two consecutive evaluations" scores the same network twice. Composites now use `n_steps = 64`. v2 §5.3 |
| **GRU sequence length** (partly) | Accuracy only, on 9 known tasks over 3 seeds (PR #18): task-episode macro 63 % at N = 1, 83 % at N = 5, 85 % at N = 10, 20 and 50 (within 0.5 points; seed ranges overlap). Switching latency, the other half of the criterion, is not measured. [`results.md`](results.md) |
