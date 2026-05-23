# CLAUDE.md

This file is the briefing for Claude Code. Read this before touching any file.

---

## What This Project Is

A hierarchical continual reinforcement learning agent that:
- Learns multiple tasks sequentially **without catastrophic forgetting**
- Reuses learned skills as macro-actions (options) in higher-level policies
- Detects novel tasks autonomously via reconstruction error
- Grows its hierarchy unboundedly at runtime

Evaluation environment: **MiniGrid** (Empty-8x8 → FourRooms → DoorKey → KeyCorridor).

---

## Repo Structure

```
src/
  continual_learning/   # Progressive Networks columns (PN)
  task_detection/       # Autoencoder (AE) + GRU task identifier
  options/              # Option wrapper + MetaController
  gui/                  # Streamlit visualization
tests/
docs/                   # Architecture and design documents (read these)
```

---

## Read These Docs First

Before writing any code, read the relevant doc:

- `docs/ENS491_Module_Interfaces.md` — input/output contracts for every module. **Do not deviate from these interfaces.**
- `docs/ENS491_Recursive_Hierarchy_Design.md` — formal hierarchy design, {n,m} notation, recursive structure.

---

## Stack

- Python 3.10, PyTorch + CUDA
- MiniGrid (`pip install minigrid`), Stable-Baselines3, Gymnasium
- Experiment tracking: Weights & Biases
- GUI: Streamlit

---

## Hard Rules — Never Break These

**Environment:**
- Always use `FlatObsWrapper` + `MlpPolicy`. **Never use CnnPolicy** — MiniGrid obs is 7×7, smaller than default 8×8 CNN kernel.
- Observation shape after FlatObsWrapper: `(147,)` float32.

**Progressive Networks:**
- `frozen=True` means **no gradient**, but **forward pass still runs**. Lateral connections need activations from frozen columns.
- Never modify a frozen column's weights for any reason.
- `sub_layer` field must exist on `Column` even if `None` in Phase 1. Do not remove it.

**Signals:**
- `reconstruction_error` (AE) and `sub_policy_reward` are **separate signals**. Never merge into a single threshold.
- Meta-controller reward and sub-policy reward travel on **separate channels**.

**Options:**
- Never add an option to MetaController before its column is stabilized (training complete, reward plateau reached).

**Hierarchy:**
- The system is unbounded by design. Never hardcode number of layers or columns.
- New layer = set `column.sub_layer = MetaController(...)`. Nothing else changes at root level.

---

## What To Build

Four modules, all empty right now. Fill them in this order:

1. `src/continual_learning/` — Progressive Networks (2 columns, lateral connections, PPO training)
2. `src/task_detection/` — Autoencoder (MLP, 147→64→32→64→147) + GRU task identifier (hidden=64, N=20, supervised)
3. `src/options/` — Option wrapper + MetaController (SB3 PPO, Discrete action space)
4. Integration script — AE → GRU → MetaController → Option → env pipeline
5. `src/gui/` — Streamlit: reconstruction error curve, active column, option selection history

Each module's exact interface is in `docs/ENS491_Module_Interfaces.md`. Build to that spec.

---

## Module Interfaces (Quick Reference)

```python
# AE output
AEOutput(reconstruction_error: float, is_novel: bool, latent_z: Tensor)

# GRU output  
GRUOutput(task_id: int, confidence: float, all_probs: Tensor)

# Column
Column(n, m, frozen, lateral_source, sub_layer)
column.forward(obs) -> ColumnOutput(action, value, activations)

# Option
option.step(obs) -> OptionStepOutput(action: int, terminated: bool, info: dict)

# MetaController
mc.select_option(obs) -> MetaControllerOutput(selected_option_id, option)
mc.add_option(option) -> None
```

Full interface spec: `docs/ENS491_Module_Interfaces.md`

---

## What NOT To Do

- Do not use CnnPolicy anywhere
- Do not hardcode layer/column counts
- Do not merge AE and reward signals
- Do not add unstabilized options to MetaController
- Do not remove `sub_layer` field from Column
- Do not use `localStorage` or any browser storage in GUI artifacts
- Do not install packages outside the `ens491` conda environment
- Do not modify `sandbox/` — it is reference only
