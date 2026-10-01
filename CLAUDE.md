# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## What This Project Is

A hierarchical continual reinforcement learning agent. Every learned task becomes a frozen, reusable **skill node**. More complex skills are **composite nodes** whose policy selects among earlier frozen skills as options, to arbitrary depth. Skills are never forgotten because they are frozen; task recognition (autoencoder + GRU) decides when something is new and which known task is running.

Evaluation environment: **MiniGrid**, with a compositional curriculum built from Unlock, DoorKey, UnlockPickup and BlockedUnlockPickup (see `docs/ENS491_Target_Architecture_v2.md` §11).

---

## Read These Docs First

1. `docs/ENS491_Target_Architecture_v2.md`: **the reference design.** Skill nodes, DAG, execution stack, SMDP training, stabilization, roadmap. It overrides every other doc where they disagree.
2. `docs/ENS491_Project_State.md`: current status of each roadmap task and the open empirical questions.
3. `docs/ENS491_Module_Interfaces.md`: v1 interface contracts. Only the AE and GRU sections still apply; the Column / Option / MetaController / TaskLifecycleManager sections are superseded by v2 §2–§5 and §10.

`docs/ENS491_Recursive_Hierarchy_Design.md` is the superseded v1 design. Read it only for history.

---

## Migration Status (read before writing code)

The code in `src/` is still **v1**: layer-0 columns, one ε-greedy `MetaController` at n=1, a single global active option, fixed 200-step termination, reward mean/std stabilization. It does not implement v2.

- Implement v2 following the roadmap in v2 §12, starting with **H-0** and **H-1**. Status is tracked in `docs/ENS491_Project_State.md`.
- Do not extend v1-only modules (`meta_controller.py`, `option.py`, `task_lifecycle.py`, `stabilization.py`). Replace them as listed in v2 §10.
- Reuse what still fits: `ColumnPolicy` / `LateralMlpExtractor` (primitive policy network), `AutoencoderModule`, `GRUTaskIdentifier`, the seeding and checkpoint code in `pn_trainer.py`.
- Keep the existing tests green until the module they cover is replaced. Each replacement lands with its own tests.

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
pytest tests/test_pn_basic.py -v
```

Run the Streamlit GUI:
```bash
streamlit run src/gui/app.py
```

---

## Stack

- Python 3.10, PyTorch. **No CUDA on the current machine** (Intel Arc integrated only); the RTX 3070 Ti
  of the v1 runs was a different machine, and a MacBook Pro (M5 Pro) is a second target. Device
  resolution is `cuda → mps → cpu` with an override, but **`cpu` is the default**: a 147→64→64 MLP is
  too small to benefit, and SB3 recommends CPU for `MlpPolicy`. Keep `requirements.txt` free of
  CUDA-pinned wheels so it installs on macOS arm64.
- MiniGrid, Gymnasium, Stable-Baselines3, **sb3-contrib** (MaskablePPO for composite nodes). Versions
  are pinned in `requirements.txt`; the dev env is minigrid 3.1.0 / gymnasium 1.3.0 / SB3 2.9.0.
- Experiment tracking: Weights & Biases
- GUI: Streamlit
- Do not install packages outside the `ens491` conda environment

---

## Core Model (v2, summary)

```
SkillNode {n,m}
  children     ()            → primitive: policy outputs a MiniGrid action (0–6), n = 0
  children     (c1, c2, …)   → composite: policy outputs a child index, n = 1 + max(child.n)
  initiation   I(s)          predicate on env state (default: always True)
  goal         β(s)          predicate on env state = terminal condition
  timeout      int           exceeding it = failure
  status       DECLARED → TRAINING → STABLE → FROZEN
```

- **Graph:** a DAG. A composite may use any earlier FROZEN node, at any level, as a child. A node may have several parents.
- **Execution:** one `Executor` keeps an active stack `[root, …, primitive]`. A selected child keeps control until its own termination (goal reached, timeout, or an ancestor terminated). Termination is checked top-down at every env step; selection descends until a primitive is on top.
- **Training:** primitives use PPO on `SubgoalEnv` (reset hook = initial conditions, reward on goal). Composites use MaskablePPO on `OptionEnv`, where one step = one complete child execution (SMDP).
- **Reward:** every node's reward comes from **its own `goal` predicate**, never from the env's reward —
  `1 − 0.9·steps/timeout` on success, 0 otherwise, at every level.
- **Masking is goal-only:** `available(s) = not goal(s) and (primitive or any child available)`.
  `initiation` is task metadata and drives reset hooks; it does **not** mask.
- **Roots:** the GRU's task id selects which root node the executor runs, through a task registry.

Full definitions and pseudo-code: v2 §2–§6.

---

## Network Architectures

**Primitive policy network:** `ColumnPolicy` with `LateralMlpExtractor`, MLP `147 → 64 → 64`, then SB3's policy (7 actions) and value heads. Optional laterals: `nn.Linear(64, 64)` from each hidden layer of a source column, **added** to the receiving hidden layer (after its ReLU, as currently implemented). No concatenation.

**Composite policy network:** standard MaskablePPO MLP policy on the same 147-dim observation, action space `Discrete(len(children))`.

**Autoencoder:** MLP `147 → 64 → 32 → 64 → 147`, on **per-channel normalized** input (object index /10, color /5, state /2).

**GRU task identifier:** hidden size 64, sequence length N = 20 by default, supervised training.

---

## SB3 Integration Pattern

Subclass SB3's `ActorCriticPolicy` for primitives (as `ColumnPolicy` does) and keep SB3's training loop intact. Do not reimplement PPO. Composite nodes get their SMDP from the `OptionEnv` wrapper, so MaskablePPO runs unmodified.

**If any of the following occur, stop and report before attempting a workaround:**
- Lateral activations are not accessible inside the custom policy's forward pass
- Gradient flow through lateral connections is broken
- VecEnv is incompatible with the lateral activation sharing mechanism
- `OptionEnv` cannot be vectorized because frozen child policies hold per-env state

---

## Stabilization Criterion

Every `eval_every` training steps, run 50 evaluation episodes in execution mode (deterministic) on the node's training env. **Success rate** = fraction of episodes in which the node's `goal` is reached before its `timeout`.

A node is **STABLE** when, in two consecutive evaluations, its success rate is **≥ 0.90** *and* its
**Wilson 95% lower bound is ≥ 0.80** (at N=50 this binds at 46/50; plain "≥0.90 twice" stops on a lucky
pair, since σ≈0.042). The 50 eval seeds are **fixed and identical at every evaluation**. At freeze time
the node is evaluated once on a **second seed set never used for stopping**, and that is the number
quoted as its success rate. Then it is frozen. This applies to primitive and composite nodes alike.
Thresholds are experiment parameters: record them in each run's `meta.json`.

The v1 rule (mean reward > 0.85 and std < 0.05 over 50 episodes) is retired.

---

## Hard Rules — Never Break These

**Environment:**
- Observation pipeline: `ImgObsWrapper` → `FlattenObservation` → `ChannelNormalize` → shape `(147,)`,
  float32. (`FlatObsWrapper` adds the mission string and does not give 147 dims; do not use it.)
- `ChannelNormalize` divides by `[10, 5, 2]` per channel and is **on by default for every new node**.
  The same utility normalizes the AE input. `SubgoalEnv`, `OptionEnv` and evaluation must all use the
  same setting, and the setting is part of the node's `spec_hash`. The legacy `runs/pn4/col01` was
  trained on raw integers and cannot be a lateral source for a normalized node without retraining.
- **Never use `CnnPolicy`**: the 7×7 observation is smaller than SB3's default 8×8 kernel.
- Goal and initiation predicates read the environment state, through an immutable `StateView`
  snapshot (see Hierarchy below), never `env.unwrapped` directly. Policies only see the 147-dim
  observation. Masking is privileged in this sense and policies are not; that asymmetry is deliberate.

**Skill nodes:**
- A node's `children` tuple is **immutable** after creation. Never grow a node's action space; create a new node instead.
- Only FROZEN nodes may be children or registered roots. Never reuse a node before it is stabilized.
- `frozen` means no gradients, but the forward pass still runs (laterals and parents need it). Never modify a frozen node's weights for any reason.
- Frozen nodes act deterministically by default.
- A node's level `n` is derived from its children. Never assign it by hand.
- `timeout ≥ max(child.timeout)`, and a root's `timeout ≤ env.max_steps`. Validate at creation.
- Every node carries a `spec_hash` over env, predicates, timeout, obs pipeline and children's hashes.
  Loading a node whose hash does not match the requested spec is an error, never a warning.

**Hierarchy:**
- Unbounded by design: never hardcode the number of levels or nodes, and never branch on `n` (`if level == 1: …`). The executor and trainers must work for any depth.
- There is no per-layer MetaController object. A composite node's policy is its meta-controller.
- Children are masked unless available: `not goal(s)` and, for composites, at least one available
  child — evaluated recursively and **memoized per env step** (it is exponential on a DAG otherwise).
  `initiation` does not appear here. A mask handed to a policy always has at least one True;
  when nothing is available the episode ends instead (MaskablePPO cannot take an empty mask).
- `max_decisions` is mandatory on every composite, not optional: under goal-only masking a parent can
  otherwise re-select a timed-out child forever.
- Predicates read an immutable `StateView` snapshot built once per env step, never `env.unwrapped`
  directly.

**Signals:**
- AE reconstruction error and reward-based signals are separate. Never merge them into one threshold.
- A parent's SMDP reward and a child's subgoal reward travel on separate channels.

---

## What NOT To Do

- Do not use `CnnPolicy` anywhere
- Do not hardcode layer or node counts, or special-case any level
- Do not add children to an existing node
- Do not reuse a node that has not stabilized
- Do not merge AE and reward signals
- Do not extend v1-only modules; replace them per v2 §10
- Do not put FourRooms (or any single benchmark) on the hierarchy's critical path
- Do not use `localStorage` or any browser storage in GUI artifacts
- Do not let a composite's reward come from the environment's reward
- Do not use `SubprocVecEnv` for `OptionEnv`; `DummyVecEnv` only
- Do not delete `stabilization.py` while `ColumnTrainer` still imports it
- Do not reference or use the sandbox repo; build any needed baselines from scratch here
- Do not change the design in code without updating `docs/ENS491_Target_Architecture_v2.md` first
