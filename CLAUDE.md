# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## What This Project Is

A hierarchical continual reinforcement learning agent that learns multiple tasks sequentially without catastrophic forgetting, reuses learned skills as macro-actions (options) in higher-level policies, and detects novel tasks autonomously via reconstruction error. The hierarchy grows unboundedly at runtime.

Evaluation environment: **MiniGrid** (Empty-8x8 → FourRooms → DoorKey → KeyCorridor).

---

## Commands

All commands assume the `ens491` conda environment is active:

```bash
conda activate ens491
```

Run all tests:
```bash
pytest tests/ -v
```

Run a single test file:
```bash
pytest tests/test_placeholder.py -v
```

Run the Streamlit GUI:
```bash
streamlit run src/gui/app.py
```

---

## Read These Docs First

Before writing any code, read the relevant doc:

- `docs/ENS491_Module_Interfaces.md` — input/output contracts for every module. **Do not deviate from these interfaces.**
- `docs/ENS491_Recursive_Hierarchy_Design.md` — formal hierarchy design, {n,m} notation, recursive structure.
- `docs/ENS491_Project_State.md` — locked decisions, open empirical questions, roadmap, and current status.

---

## Stack

- Python 3.10, PyTorch + CUDA (RTX 3070 Ti, CUDA 12.1)
- MiniGrid (`pip install minigrid`), Stable-Baselines3, Gymnasium
- Experiment tracking: Weights & Biases
- GUI: Streamlit
- Do not install packages outside the `ens491` conda environment

---

## Repo Structure

```
src/
  continual_learning/   # Progressive Networks columns (PN) — build first
  task_detection/       # Autoencoder (AE) + GRU task identifier — build second
  options/              # Option wrapper + MetaController — build third
  gui/                  # Streamlit visualization — build last
tests/
docs/                   # Architecture and design documents (read these first)
sandbox/                # Reference only — do not modify
```

All four `src/` modules are implemented (Phase 1). The MetaController uses epsilon-greedy selection in Phase 1 — PPO-based MC is deferred to Phase 2 due to dynamic action space complexity.

---

## Architecture: Signal Flow

Every environment step runs this pipeline:

```
obs_t → [AE] → is_novel?
                  ├─ True  → on_novel_task_detected() → open new Column → train → stabilize → add_option()
                  └─ False → [GRU] → task_id → [TaskLifecycleManager]
                                                       └─ recursive_step(obs, root_MetaController)
                                                              └─ MC.select_option() → Option.step()
                                                                     ├─ sub_layer=None (LEAF): Column.forward() → action → env
                                                                     └─ sub_layer=MC (NON-LEAF): recurse deeper
```

Key architectural property: `TaskLifecycleManager` only calls the root `MetaController`. It does not know or care how many levels exist below. Adding a new layer only requires setting a `Column.sub_layer = MetaController(...)` — nothing at the root changes.

---

## Critical Path

The following sequence is blocking — if any step is skipped, the full system cannot be assembled:

```
PN-1 → PN-2 → PN-3* → PN-4 → OPT-1 → OPT-2 → INT-2
                ↑
        Recursive hierarchy design must be finalized on paper before PN-3
```

Can be started in parallel (does not block critical path): `AE-1, AE-2, GRU-1, GRU-2, EVAL-1, EVAL-2`

---

## Module Interfaces (Quick Reference)

```python
# AE
AEOutput(reconstruction_error: float, is_novel: bool, latent_z: Tensor)

# GRU
GRUOutput(task_id: int, confidence: float, all_probs: Tensor)  # task_id=-1 means uncertain

# Column
Column(n, m, frozen, lateral_source, sub_layer)
column.forward(obs) -> ColumnOutput(action, value, activations)

# Option
option.step(obs) -> OptionStepOutput(action: int, terminated: bool, info: dict)

# MetaController
mc.select_option(obs) -> MetaControllerOutput(selected_option_id, option)
mc.add_option(option) -> None  # does NOT retrain from scratch; uses exploration bonus

# TaskLifecycleManager
tlm.step(obs) -> Action  # the single entry point for every env step
```

Full interface spec: `docs/ENS491_Module_Interfaces.md`

---

## Network Architectures

**Column policy network:** MLP `147 → 64 → 64 → 7` (input obs → hidden → hidden → actions). Use for all columns unless explicitly told otherwise.

**Autoencoder:** MLP `147 → 64 → 32 → 64 → 147`. Input is always `(147,)` because `FlatObsWrapper` is always applied. The convolutional AE mentioned in `ENS491_Project_State.md` is outdated — ignore it.

**GRU task identifier:** hidden size 64, sequence length N=20, supervised training.

---

## SB3 Integration Pattern

Subclass SB3's `ActorCriticPolicy` and override the forward pass to inject lateral connections. Keep SB3's training loop intact — do not reimplement PPO.

**If any of the following occur, stop and report before attempting a workaround:**
- Lateral activations are not accessible inside the custom policy's forward pass
- Gradient flow through lateral connections is broken
- VecEnv is incompatible with the lateral activation sharing mechanism

---

## Lateral Connection Mechanism

Column `{n, m}` reads the 64-dim hidden activations from `{n, m-1}`, passes them through `nn.Linear(64, 64)`, and **adds** the result element-wise to its own 64-dim hidden. No concatenation.

---

## Stabilization Criterion

A column is stabilized when, over the **last 50 episodes**:
- Mean episode reward > 0.85
- Standard deviation of episode reward < 0.05

Only after both conditions are met is the column frozen and wrapped as an option. This is the only stabilization criterion — do not invent others.

---

## Hard Rules — Never Break These

**Environment:**
- Always use `FlatObsWrapper` + `MlpPolicy`. **Never use CnnPolicy** — MiniGrid obs is 7×7, smaller than the default 8×8 CNN kernel.
- Observation shape after FlatObsWrapper: `(147,)` float32.

**Progressive Networks:**
- `frozen=True` means **no gradient**, but **forward pass still runs**. Lateral connections need activations from frozen columns.
- Never modify a frozen column's weights for any reason.
- `sub_layer` field must exist on `Column` even if `None` in Phase 1. Do not remove it.
- Lateral connections are defined only within a layer, left to right: `{n, m-1} → {n, m}`.

**Signals:**
- `reconstruction_error` (AE) and `sub_policy_reward` are **separate signals**. Never merge into a single threshold.
- Meta-controller reward and sub-policy reward travel on **separate channels**.

**Options:**
- Never add an option to MetaController before its column is stabilized (training complete, reward plateau reached).

**Hierarchy:**
- The system is unbounded by design. Never hardcode number of layers or columns.
- New layer = set `column.sub_layer = MetaController(...)`. Nothing else changes at root level.
- Hierarchy is a **tree** (not DAG) in Phase 1-2 — each column belongs to exactly one MetaController.

---

## What NOT To Do

- Do not use CnnPolicy anywhere
- Do not hardcode layer/column counts
- Do not merge AE and reward signals
- Do not add unstabilized options to MetaController
- Do not remove `sub_layer` field from Column
- Do not use `localStorage` or any browser storage in GUI artifacts
- Do not reference or use the sandbox repo — build any needed baselines from scratch here
- Do not change module interfaces without updating `docs/ENS491_Module_Interfaces.md` first
