# ENS 491 — Project State & Roadmap

> Technical memory of the project. Reference point for team meetings, supervisor sessions, and implementation decisions.
>
> **Fixed:** Locked architectural decisions — not open for debate.  
> **Empirical:** Design alternatives — implement all, compare results, pick the winner.  
> **Status:** Current state and next steps.

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

## Locked Architectural Decisions

Not open for debate. If a change is needed, get supervisor approval first, then update this document.

### Continual Learning Backbone: Progressive Networks

- Separate column per task (`{n, m}` notation: `n` = layer, `m` = column count)
- Column weights are **frozen** after training — never modified again
- Lateral connections defined **between children of the same meta-controller**: `{n, m-1} → {n, m}`
- Column count and layer count **grow at runtime** — architecture is recursive/unbounded from the start
- Implementation path: Doric → custom PN → architecture-specific custom PN (this order cannot be skipped)

### Hierarchy Structure

- Hierarchy is implemented as a **tree** — each primitive column belongs to exactly one meta-controller
- Each layer's meta-controller **only sees its own layer**
- Hierarchy **depth is not fixed** — new layers can open at runtime
- `{n, m}` notation: higher `n` = higher abstraction, `m` = column count at that layer

### Task Detection

- **Reconstruction error** → observation-level novelty signal (is there a new task?)
- **Sub-policy reward** → behavioral insufficiency signal (is the current policy sufficient?)
- These two signals are kept **independent** — never merged
- AE and GRU are sequential in the pipeline: AE answers "is it new?", GRU answers "which one?"

### Options Framework

- Every trained column is **wrapped as an option** (Sutton et al. 1999)
- Options are added to MetaController only **after stabilization** — no half-trained options
- MetaController is trained with **PPO** (SB3)
- When a new option is added, MetaController is not retrained from scratch — exploration bonus is used instead

### Stack

- Python 3.10, PyTorch + CUDA, MiniGrid (Gymnasium), Stable-Baselines3
- `FlatObsWrapper + MlpPolicy` — CnnPolicy does not work on MiniGrid (7×7 obs vs 8×8 kernel)
- Primary test environment: MiniGrid (Empty → FourRooms → DoorKey → KeyCorridor)
- Experiment tracking: Weights & Biases

---

## Open Empirical Questions

No predetermined correct answer. Implement each alternative, compare.

### AE Architecture
`Undercomplete AE` → `Sparse AE` → `VQ-VAE`  
Criterion: how sharply does reconstruction error spike at task boundaries?  
Reference: Meyer et al. (2024) — replicate their comparison on MiniGrid.

### AE Threshold
`Fixed global` → `Per-task adaptive` → `Statistical test (KS / Wasserstein)`  
Reference: Dick et al. 2024 (SWOKS) — does this on MiniGrid + PPO.

### AE Update Protocol
`Single global AE, fine-tune` → `Per-task separate AE` → `Frozen shared encoder + task-specific head`  
Also run: AE never updated — how much does performance degrade?

### GRU Sequence Length
`N=5` → `N=20` → `N=50`  
Criterion: classification accuracy vs task-switching latency trade-off.

### GRU Training Paradigm
`Supervised (labeled)` → `Contrastive (SimCLR-style)` → `Clustering (DBSCAN)`  
Supervised is the mandatory starting point. Label-free transition is a research question.

### Meta-Controller Sparse Reward
`Sub-goal completion bonus` → `Potential-based shaping` → `Intrinsic curiosity`  
Criterion: convergence speed and final performance.

### Termination Condition
`Fixed step limit` → `Learned termination (Option-Critic style)` → `GRU task-change signal`

### New Column Trigger
`AE threshold only` → `Reward plateau only` → `AND` → `OR`  
Run each combination on the same task sequence, measure unnecessary column openings.

### Task Sequence
`Easy→Hard` → `Hard→Easy` → `Similar first` → `Isolated first`  
Criterion: BWT, FWT, Average Performance. Sequence effect is itself a finding.

---

## Current Status

### ✅ Done

| Task | Notes |
|---|---|
| PPO baseline — Empty-8x8 | FlatObsWrapper + MlpPolicy, reward ~0.04 → ~0.91 |
| Catastrophic forgetting demonstration | Fine-tune → return to Empty, performance drop measured |
| Development environment | Windows + RTX 3070 Ti, CUDA 12.1, Miniconda ens491 env, SB3, MiniGrid |
| GitHub sandbox repo | `ens491-sandbox` active |
| `doric_test.py` | Doric library PN explored, lateral connection mechanism studied |
| Proposal + literature review | 120+ papers, 7 categories, Zotero integration |
| Team onboarding document | Alignment doc complete |

### 🔄 In Progress

| Task | Status |
|---|---|
| Progressive Networks implementation | At Doric stage. `custom_pn_test.py` not yet written |
| Recursive hierarchy design | Paper decisions made, not yet reflected in code |

---

## Roadmap

### Phase 1 — Continual Learning Core

**Goal:** Does PN work in this stack? Do AE and GRU work independently? Is forgetting prevented?  
Explicit task ID present. If something doesn't work here, moving to Phase 2 is pointless.

#### Progressive Networks

| # | Task | SP | Status |
|---|---|---|---|
| PN-1 | `custom_pn_test.py` — pure PyTorch, two columns, one lateral connection, on Empty | 5 | ⬜ Next |
| PN-2 | Doric vs custom PN comparison — which fits the recursive design more naturally? | 3 | ⬜ |
| PN-3 | Architecture-specific custom PN — `{n,m}` notation, runtime column growth, PPO + SB3 integration | 13 | ⬜ |
| PN-4 | Second column + FourRooms, visualize lateral connection activations | 5 | ⬜ |
| PN-5 | Forgetting test: return to first task after N tasks, measure retained performance | 3 | ⬜ |
| PN-6 | EWC baseline — same task sequence, lower bound for paper | 5 | ⬜ |
| PN-7 | Third column with DoorKey — does it scale to N tasks? | 3 | ⬜ |

> **Before PN-3:** Finalize recursive hierarchy design on paper before writing code. DAG structure, MC-to-MC option calling, termination propagation. This is a milestone — skipping it means a rewrite later.

#### Autoencoder

| # | Task | SP | Status |
|---|---|---|---|
| AE-1 | Convolutional AE baseline — MiniGrid obs (7×7×3), encode/decode pipeline | 3 | ⬜ |
| AE-2 | Threshold analysis — familiar vs novel task error distributions, measure overlap | 5 | ⬜ |
| AE-3 | Architecture comparison: Undercomplete → Sparse AE → VQ-VAE (empirical) | 8 | ⬜ |
| AE-4 | Update protocol comparison: global fine-tune / per-task / frozen encoder (empirical) | 8 | ⬜ |

#### GRU

| # | Task | SP | Status |
|---|---|---|---|
| GRU-1 | Data collection pipeline — labeled episode observation sequences across tasks | 2 | ⬜ |
| GRU-2 | Supervised baseline — compare N=5 vs N=20 vs N=50 (empirical) | 5 | ⬜ |
| GRU-3 | Label-free transition attempt: contrastive / clustering (research question, negative result is also valuable) | 8 | ⬜ |

#### Phase 1 Integration

| # | Task | SP | Status |
|---|---|---|---|
| INT-1 | AE + GRU handoff — when AE says "novel", does GRU step in correctly? Measure false positive/negative | 5 | ⬜ |

---

### Phase 2 — Hierarchical Control

**Goal:** Can the system understand tasks without explicit task IDs? Can it use learned skills as macro-actions?  
This phase is the paper's core contribution. A completed Phase 2 system is publishable on its own.

> Before Phase 2: recursive hierarchy paper design must be confirmed, and Phase 1 PN/AE/GRU must be working.

#### Dynamic Options & Meta-Controller

| # | Task | SP | Status |
|---|---|---|---|
| OPT-1 | PoC: 2 frozen PPO + meta-controller without PN — does the options framework work? | 8 | ⬜ |
| OPT-2 | Wrap PN columns as options — frozen column = option, test termination | 5 | ⬜ |
| OPT-3 | Sparse reward strategy comparison: sub-goal bonus / potential-based / curiosity (empirical) | 8 | ⬜ |
| OPT-4 | New option discovery: exploration bonus vs optimistic initialization (empirical) | 5 | ⬜ |
| OPT-5 | Termination condition comparison: fixed limit / learned / GRU signal (empirical) | 8 | ⬜ |

#### Phase 2 Integration & Ablation

| # | Task | SP | Status |
|---|---|---|---|
| INT-2 | AE + GRU + PN + MetaController end-to-end, explicit task ID removed | 13 | ⬜ |
| INT-3 | Ablation: AE disabled | 3 | ⬜ |
| INT-4 | Ablation: GRU disabled | 3 | ⬜ |
| INT-5 | Ablation: lateral connections disabled | 3 | ⬜ |
| INT-6 | Task sequence comparison: 4 different orderings, same system (empirical) | 8 | ⬜ |

---

### Phase 3 — Planning (Stretch Goal)

**Goal:** Can the MetaController plan a sequence over the option graph rather than making reactive per-step selections?  
Do not start until Phase 2 is stable. Paper is written from Phase 2 results; Phase 3 is additional contribution.

| # | Task | SP | Status |
|---|---|---|---|
| PLN-1 | Build option transition graph — which options succeed after which? | 8 | ⬜ |
| PLN-2 | Add lookahead — compare reactive vs deliberative | 13 | ⬜ |
| PLN-3 | Multi-level hierarchy end-to-end: Level-2 MetaController, does two levels work? | 13 | ⬜ |

---

### Phase 4 — Evaluation & Paper (Parallel with Phase 2)

| # | Task | SP | Status |
|---|---|---|---|
| EVAL-1 | W&B dashboard: reconstruction error, option frequency, per-task reward live | 3 | ⬜ |
| EVAL-2 | BWT / FWT / Average Performance utility — logs to W&B | 3 | ⬜ |
| EVAL-3 | Minimal Streamlit GUI: active column, option history, policy visualization | 8 | ⬜ |
| EVAL-4 | Full evaluation: 3+ task sequence, all metrics, baseline comparison | 8 | ⬜ |
| EVAL-5 | Paper draft — from Phase 2 results | 13 | ⬜ |

---

## Critical Path

If this sequence is blocked, the system cannot be assembled:

```
PN-1 → PN-2 → PN-3* → PN-4 → OPT-1 → OPT-2 → INT-2
                ↑
        Recursive hierarchy design milestone (on paper)
```

Can be started in parallel (does not block critical path):  
`AE-1, AE-2, GRU-1, GRU-2, EVAL-1, EVAL-2`
