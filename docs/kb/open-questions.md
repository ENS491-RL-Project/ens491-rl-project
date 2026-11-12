# Open questions

Decisions that are not made yet, and what each one blocks. When one is decided, write a file in
[`decisions/`](decisions/) and remove it here. Last reviewed: 2026-11-12.

## Open decisions

Each is settled by an experiment where possible ([0010](decisions/0010-decide-empirically.md)).

| Question | Why it matters / what it blocks | How to settle | Owner | Source |
|---|---|---|---|---|
| Do frozen skills act deterministically or by sampling? | Deterministic play loops in partially observed rooms; under sampling every frozen node scored 150/150 in the audit (`solve_doorkey` 149/150), but `open_door` rose only 0.74→0.90 in E.2. Blocks `pickup_box`/UnlockPickup and the train/eval rules. | Audit exists (evaluation only); still to run: training + final evaluation under each mode on the same seeds | Emir | [`plans/2026-10-02_execution_mode_audit.md`](../plans/2026-10-02_execution_mode_audit.md) |
| How is UnlockPickup finished? | `pickup_box` failed STABLE twice (0.64–0.70). Step 3 of the 4-task curriculum. | Retrain `pickup_box` under the chosen execution mode | Emir | [`plans/2026-10-02_H7a_closeout.md`](../plans/2026-10-02_H7a_closeout.md); [`plans/2026-10-02_H7_summary.md`](../plans/2026-10-02_H7_summary.md) §2 (source of 0.64–0.70, spec-2 held-out) |
| Phase B growth (composite-first heuristic) | Needed for "compose while learning" (0006); Phase A is temporary (0005). | G-1…G-3, v2 §8 | Emir | v2 §8, §12 |
| Lateral connections: forward transfer at all? | Decides whether PN is part of the claim (0003, 0006). | C-2 on primitives first | Emir | [`decisions/0003`](decisions/0003-progressive-networks-off-critical-path.md) |
| Fresh test seed set for final evaluation | Only item of the evaluation-rigour list without a PR. | Define once, never used in training or earlier evaluation | Mehmet P. | team chat, 2026-11 |
| When does the GRU's task id select the root node? | v2 §6.2 designs it; code ignores the prediction. Elif's PRs cover training/eval only. | After R-1 accuracy is known, integrate and measure end-to-end | Elif + Emir | PR #18, v2 §6.2 |
| Does the flat baseline need memory? | A memoryless PPO may be an unfair baseline in partially observed tasks. | Run a recurrent PPO baseline on the same tasks | Emir | [`decisions/0006`](decisions/0006-claim-compose-and-reuse.md) |
| Fix or replace the choice task (`unlock_matching`) | A wrong key pick is recoverable (43/50 swaps), so the task cannot test conditioned choice. | Redesign so a wrong pick is unrecoverable; rerun | Emir | [`plans/2026-10-02_H6_summary.md`](../plans/2026-10-02_H6_summary.md) §4 |
| Retrain the PN-4 column on normalised observations (~40k steps)? | Needed before C-2 if the old column is the lateral source. | Part of C-2 setup | Emir | SUPERVISOR_QUESTIONS Q9 |
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

Moved unchanged from [`ENS491_Project_State.md`](../ENS491_Project_State.md). Section numbers refer to v2.

| Topic | Alternatives | Criterion | v2 |
|---|---|---|---|
| AE architecture | Undercomplete → Sparse AE → VQ-VAE | Separation of known vs novel tasks (ROC-AUC); ref. Meyer et al. 2024 | §6.1 |
| AE scope | Single global AE → one AE per task (base paper) | Novelty accuracy after each new task | §6.1 |
| AE threshold | Fixed per task (mean + kσ) → statistical test (KS / Wasserstein, ref. SWOKS) | False positive / negative rate | §6.1 |
| GRU sequence length | N = 5 → 20 → 50 | Accuracy vs switching latency | §6.2 |
| GRU continual training | Retrain on accumulated sequences → EWC on the GRU | Old-task accuracy after each new task | §6.3 |
| Lateral sources | Most recent primitive → all earlier primitives → nearest by AE latent | Steps to STABLE (forward transfer) | §7 |
| Execution mode of frozen nodes | Deterministic → stochastic | Success rate, loop frequency | §5.4 |
| SMDP discounting | Per-decision γ → exact γᵏ | Composite learning speed | §5.2 |
| Stabilization thresholds | SR 0.90 × 2 evals (default) → stricter / looser | Downstream reuse quality vs training cost | §5.3 |
| Growth order | Composite-first → primitive-first | Unnecessary nodes created, total steps | §8 |
| Primitive training distribution | Single layout → mix of all layouts where the goal is defined | Success inside parent tasks | §11 |
| Composite depth | T9a (level 2, primitives) → T9b (level 3, reuses `unlock_pickup`) | Steps to STABLE | §11 |
