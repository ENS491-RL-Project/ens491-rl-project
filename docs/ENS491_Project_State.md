# ENS 491 — Project State & Roadmap

> Technical memory of the project: current status, open empirical questions, and where each roadmap task stands.
> **Design decisions live in `ENS491_Target_Architecture_v2.md`**, not here. This file only tracks state.
>
> Last updated: 2026-10-02 (H-6.9 diagnosis; H-7 started and stopped at H-7.3 on plan assumption A2).

---

## Story Points Reference

| SP | Meaning |
|----|---------|
| 1 | Trivial. Already know exactly how to do it. |
| 2 | Straightforward. Minor unknowns. |
| 3 | Moderate. A few unknowns. |
| 5 | Significant. Design decision + implementation together. |
| 8 | Complex. Multiple unknowns, requires iteration. |
| 13 | Very complex. Should probably be broken into subtasks. |

---

## Locked Decisions

See `ENS491_Target_Architecture_v2.md`. Summary:

- Skill nodes (primitive / composite) with initiation, goal, timeout and policy; levels derived from children.
- DAG with immutable child sets; a composite's policy is its meta-controller.
- Active execution stack; options persist until goal, timeout or ancestor termination.
- Composites trained as SMDPs (`OptionEnv` + MaskablePPO).
- Stabilization = success rate ≥ 0.90 in two consecutive evaluations.
- Forgetting of skills prevented by freezing; Progressive Network laterals measured as forward transfer.
- AE (novelty) and GRU (task id → root node) live in the lifecycle layer; AE and reward signals never merged.
- Four independent tracks; only the hierarchy track is on the critical path.

The v1 decisions this replaced are listed in v2 §13.

---

## Open Empirical Questions

No predetermined correct answer. Implement each alternative and compare. Section numbers refer to v2.

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

---

### Answered by H-6 (2026-10-02)

| Question | Answer |
|---|---|
| **Q8 — how to frame the masking contribution** | On the §11 curriculum, success rate **cannot** separate composition from masking at any depth: every composite's subgoal order is strict, the initiation mask encodes exactly that order, and a random policy under it ties the learned composite (0.98 at depth 1 in H-5, 1.00 at depth 2 in H-6.3). Stronger than "ties": under the strict mask the *randomly initialised* selector already scores 1.00, so the mask does not accelerate learning, it removes the need for it. The report states that as a property of the curriculum and quantifies the learned policy's contribution two other ways: the **divergence count** (0 of 20 withheld choices taken, so the ordering was discovered rather than given) and **efficiency** (14.2 env steps vs 113–143 for random-over-available, 8×). A dedicated choice task was built to escape this; see H-6 summary §4 for its result and §9 for why the first attempt at it did not count |
| **Primitive training distribution** | The mix matters, measured both ways. The three H-4 primitives, trained on two layouts with randomized poses, transfer into DoorKey at 1.00 from real handoff states. `{0,2} open_door`, trained on a narrower mix, scores **0.83** in the two-key variant (re-measured in H-6.8 under the corrected hook) and had to be replaced by `{0,8} open_door_tk`, trained on the parent's own distribution, which scores 0.98 there |
| **Composite depth** | First data point, re-measured in H-6.8 over distinct-policy evaluations. `{2,1} solve_doorkey` (depth 2) is STABLE in **3** policy updates / 48,887–55,659 own env steps; `{1,B} flat_doorkey_options` (depth 1, same three skills) in 3–4 updates / 64,000–80,951. Depth wins by ~1.3–1.4× on own env steps and loses with children counted (661–667k for `solve_doorkey` against 544–561k for the option set), both at held-out 1.00 on 3 seeds. The decisive comparison is still T9a vs T9b in H-7 |
| **Does reuse pay?** | **Not answered, and H-6 cannot answer it.** One task cannot distinguish amortisation from arithmetic. H-7 |
| **Can a composite condition its child choice on the observation?** | **Not shown; no run learned it within 100 updates, and the cause is not established.** `{1,2} unlock_matching`, 3 seeds × 2 mask conditions × 100 policy updates: every goal-only run converges to "always pick yellow" (50/50 episodes). Under the strict mask seed 42 picks blue first in 49/50, but seeds 43 and 44 (blue first in 29 and 32 of 50) were leaving the one-key plateau when the budget ran out — stopping success near 20/50 until about update 50, then 25–30/50; seed 43 picks the right key in 36/50 held-out episodes. A5 fails on all three seeds. H-6.9 (H-6 summary §10): the door is in view and the strict mask is correct in 400/400 resets; the selector's own network, trained supervised, reads the colour at 0.940 from 1,000 examples and 0.997 from 4,000 (logistic regression 0.91). Between the pre-set thresholds, so no verdict: not a representation limit, not cheap either. The task's own premise is also broken — a wrong pick is recoverable, because `get_key_<colour>` drops and swaps 43/50 times when handed a full hand. Fixable; see H-6 summary §4 |
| **Stabilization thresholds** | One further constraint, added in H-6.8: `eval_every` must be at least one policy update (`n_steps × n_envs`), or "two consecutive evaluations" scores the same network twice. Composites now use `n_steps = 64`. v2 §5.3 |

---

## Current Status (2026-10-01)

### ✅ Done (v1 phase, Feb–May 2026)

| Item | Notes |
|---|---|
| Literature review, proposal, Progress Report I | Submitted. Progress I framed results as architectural feasibility |
| PPO baseline on Empty-8x8 | `ImgObsWrapper` + `FlattenObservation` + MlpPolicy, reward ~0.04 → ~0.91 |
| `ColumnPolicy` / `LateralMlpExtractor` | SB3 `ActorCriticPolicy` subclass with additive laterals; reused in v2 for primitives |
| PN-4 run | `{0,1}` Empty-8x8: stable at 40k steps, mean reward 0.958, frozen (`runs/pn4/col01`). `{0,2}` FourRooms with lateral: 2M steps, peak ~0.49, final ~0.24, not stable (`runs/pn4/col02`). Note: `train_pn4.py` used a relaxed 0.75 threshold for `{0,2}`, not 0.85 |
| v1 module skeletons | AE, GRU, Option, ε-greedy MetaController, TaskLifecycleManager, Streamlit GUI; 58/58 tests passing |
| Repo moved | Now `memirkaraaslan/hierarchical-rl-agent`; `fix/declare-missing-deps` adds matplotlib and pandas to `requirements.txt` |
| Target Architecture v2 | Locked 2026-10-01 |

### ⚠️ Known limitations of the v1 code (to be replaced, see v2 §10)

- AE and GRU have never been trained on real data; AE input is not normalized and its 0.05 threshold is arbitrary.
- The GRU task id is computed but never used.
- The MetaController never learns (ε-greedy; greedy choice is always option 0).
- Options stop after a fixed 200 steps; no initiation or goal conditions.
- Single-level execution only; `Column.sub_layer` is never used; new columns are always opened at n=0 and nothing trains them.
- ~~`Column.forward` uses `deterministic=not self.frozen`, so frozen columns act stochastically.~~ Fixed in H-0 (2026-10-01).
- `scripts/demo_run.py` and the GUI run on mock policies and random observations; real checkpoints are never loaded by the pipeline.

---

## Roadmap Status

Task definitions are in v2 §12. ⬜ not started · 🔄 in progress · ✅ done

### Track H — Hierarchy (critical path)

| # | Task | SP | Status |
|---|---|---|---|
| H-0 | Fix deterministic flag; document 0.75 / 0.85 mismatch | 1 | ✅ 2026-10-01 — 59/59 tests pass |
| H-0.5 | Apply Part A amendments to v2 / CLAUDE.md / supervisor questions; pin deps, add sb3-contrib | 2 | ✅ 2026-10-01 — reqs resolve, MaskablePPO imports, 59/59 |
| H-1 | `SkillNode`, `Frame`, `Executor` + unit tests (depth-5 synthetic graph) | 8 | ✅ 2026-10-01 — 108/108 tests (49 new) |
| H-2 | Predicates + `SubgoalEnv` for T1, T2, T4; verify env IDs | 5 | ✅ 2026-10-01 — 150/150 tests; env IDs + budgets verified |
| H-3 | `EvaluationMonitor`; re-evaluate legacy `col01` | 3 | ✅ 2026-10-01 — 171/171; col01 50/50 (but Empty-8x8 is layout-deterministic) |
| H-4 | Train and freeze T1 `get_key`, T2 `open_door`, T4 `go_to_goal` | 5 | ✅ 2026-10-01 — all three FROZEN; held-out 1.00 / 1.00 / 0.92; F29 gate passed after retraining go_to_goal |
| H-5 | `OptionEnv` + MaskablePPO; T3 `unlock_door` | 8 | ✅ 2026-10-01 — `{1,1} unlock_door` FROZEN, held-out 0.98 in 6k decisions; both C1 controls run |
| H-6 | T5 `solve_doorkey` + C1 baselines (first hierarchical result) | 8 | ✅ 2026-10-02 — `{2,1} solve_doorkey` FROZEN, held-out **1.00** on 3 seeds, depth-3 stack confirmed. Flat PPO 300–450k env steps vs the composite's **~53k own / ~664k with children** (H-6.8 figures). The genuine-choice task gives a **negative** result after 100 policy updates (§4 of the summary) |
| H-7 | T6–T9b reuse and depth experiment | 8 | ⛔ **Stopped at H-7.3** (2026-10-02). H-7.0–H-7.2 done: v2 amended (Part A), pre-flight passed (V1–V7 reproduce; 0/500 unsolvable layouts; A1 `unlock_door` transfers to UnlockPickup at **0.98**), nine flat baselines launched. `{0,4} pickup_box` **not stable in 600,000 env steps on any seed** (held-out 0.28 / 0.40 / 0.24, still rising) — plan A2 is a stop. Flat PPO: Unlock STABLE at 350–450k; UnlockPickup and BlockedUnlockPickup **> 3,000,000, best SR 0.00** on all 3 seeds (censored). See `plans/2026-10-02_H7_summary.md` |
| H-6.9 | Why the choice selector did not learn | 1 | ✅ 2026-10-02 — task built as designed (400/400); the selector's network reads the door colour supervised at 0.940 from 1,000 examples and 0.997 from 4,000; between the pre-set thresholds, so no verdict. H-6 summary §10 |
| H-6.8 | Advisor corrections to H-6: fair choice run, distinct-policy evaluations, stale numbers | 3 | ✅ 2026-10-02 — v2 §5.3 gets an `eval_every` floor; composites move to `n_steps = 64`; choice task rerun at 100 updates (A5 fails, reported); `{0,2} open_door` re-measured at 0.83; regression 8 of 8. H-6 summary §9 |

### Track C — Continual learning / PN

| # | Task | SP | Status |
|---|---|---|---|
| C-1 | FourRooms diagnosis with plain PPO | 2 | ⬜ |
| C-2 | Forward transfer: laterals vs none | 5 | ⬜ |
| C-3 | Retention check after each new task | 2 | ⬜ |

### Track R — Recognition

| # | Task | SP | Status |
|---|---|---|---|
| R-1 | AE normalization, calibration, ROC-AUC; single vs per-task AE | 5 | ⬜ |
| R-2 | GRU dataset, accuracy vs N, old-task accuracy | 5 | ⬜ |
| R-3 | Task registry: GRU task id → root node | 3 | ⬜ |

### Track G — Growth (after H-6)

| # | Task | SP | Status |
|---|---|---|---|
| G-1 | Lifecycle API: declare → train → stabilize → freeze → register | 5 | ⬜ |
| G-2 | Composite-first heuristic + ablation | 8 | ⬜ |
| G-3 | AE-triggered task declaration | 5 | ⬜ |

### Track UI

| # | Task | SP | Status |
|---|---|---|---|
| UI-1 | GUI on real runs: DAG, live stack, traces, node stats | 8 | ⬜ |
| UI-2 | Import external policy as a frozen primitive | 3 | ⬜ |

### Stretch

| # | Task | SP | Status |
|---|---|---|---|
| P-1 | Planning over initiation / goal predicates | 13 | ⬜ |

---

## Critical Path

```
H-0 → H-1 → H-2 → H-3 → H-4 → H-5 → H-6 → H-7
```

In parallel, never blocking the critical path: `C-1, C-2, C-3`, `R-1, R-2, R-3`, `UI-1` (after H-1). Growth (`G-*`) starts after H-6.

---

## What Happened to the v1 Roadmap

| v1 task | Fate |
|---|---|
| PN-1, PN-2, PN-3 | Superseded by the custom PN already in `column_policy.py`; a Doric comparison is no longer needed |
| PN-4 | Done, with `{0,2}` unstable. Diagnosis moves to C-1 |
| PN-5 (forgetting test) | Becomes C-3 (retention is guaranteed by freezing; the check guards against bugs) |
| PN-6 (EWC baseline) | Optional baseline under C4 |
| PN-7 (third column, DoorKey) | Replaced by the compositional curriculum (H-4 … H-7) |
| AE-1 … AE-4 | R-1 |
| GRU-1 … GRU-3 | R-2, R-3 |
| INT-1 | R-1 + R-3 |
| OPT-1 (frozen PPO + MC PoC) | Subsumed by H-4 + H-5 |
| OPT-2 … OPT-5 | H-5 (SMDP training, masking, goal-based termination); growth questions → G-2 |
| INT-2 … INT-6 | H-6, H-7, C-2 ablations, G-2 |
| PLN-1 … PLN-3 | PLN-3 (multi-level) is now the core of H-6/H-7; planning → P-1 |
| EVAL-1 … EVAL-5 | UI-1, C-3, evaluation plan in v2 §11 |
