# ENS 491 — Recursive Hierarchy Design

> This document is the paper design required before starting PN-3 (architecture-specific custom PN).  
> **🔒 Locked:** Decided and agreed upon — not open for debate.  
> **⚠️ Open:** Pending supervisor confirmation or empirical resolution.  
> **❓ Unclear:** Not yet discussed, to be addressed in a later phase.

---

## 1. Notation

**`{n, m}`** — identifies any column in the system.

- `n` → layer index. `n=0` is the lowest level (primitive skills). Higher `n` = higher abstraction.
- `m` → column position within the layer. Starts at `m=1`, grows left to right.

Both dimensions **grow at runtime** — no fixed depth or width is defined at startup.

```
Layer n=1:   {1,1}         {1,2}         {1,3}  ...
               ↑              ↑              ↑
Layer n=0:  {0,1}  {0,2}  {0,3}  {0,4}  {0,5}  ...
```

---

## 2. The Dual Role of a Column

Every `{n, m}` column is **simultaneously** two things:

1. **Standalone policy** — an independent PPO policy for its task at its layer.
2. **Option** — a macro-action that can be called by the meta-controller of the layer above.

This dual role is fundamental to the architecture. A column is first trained as a policy, then once stabilized it is offered as an option to the layer above.

---

## 3. Hierarchy Structure

### 3.1 Tree structure 🔒

The hierarchy is implemented as a **tree**. Each primitive column belongs to exactly one meta-controller.

```
MC(n=1)
├── {0,1}
├── {0,2}
└── {0,3}
```

**Why tree, not DAG:**
- At the scale of Phase 1-2 (2-4 tasks, single MC layer) there are no primitive columns to share — DAG provides zero benefit at this scale.
- Python object references naturally allow DAG if needed later. Implementing as tree does not prevent DAG; we are simply not enforcing it now.
- Avoids unnecessary complexity.

Connections always flow from lower `n` to higher `n` — learning is one-directional, acyclic property is structurally guaranteed.

### 3.2 Meta-controller scope 🔒

Each meta-controller **only sees its own layer**:
- Current environment observation
- Status of options currently available at its layer

It has no direct access to the internal state of layers above or below.

---

## 4. Lateral Connections

### 4.1 Definition 🔒

Lateral connections are defined **within a layer, left to right**:

```
{n, m-1}  →  {n, m}
```

While `{n, m}` is being trained, it can access the intermediate layer activations of `{n, m-1}`. This is the transfer mechanism — the new column learns from the old one.

### 4.2 Scope constraint 🔒

Lateral connections only exist **between columns managed by the same meta-controller**. There are no lateral connections between columns belonging to different meta-controllers — cross-MC knowledge transfer happens at the hierarchy level, via the option mechanism.

### 4.3 Lateral connections during inference ⚠️

When a frozen `{n, m-1}` column is called as an option by an upper layer, do its lateral connections remain active?

**Current design position:** Yes — the frozen column runs a forward pass and its activations are read by `{n, m}`. Freezing only blocks gradients, not the forward pass.

> **⚠️ Pending supervisor confirmation.**

---

## 5. Option Wrapping

### 5.1 When does a column become an option? 🔒

After training and stabilization. "Stabilized" means:

- Reward curve has reached a plateau
- Performance is consistently above a defined threshold

An unstabilized column is never added to a MetaController as an option.

### 5.2 Option components (Sutton et al. 1999)

| Component | Content |
|---|---|
| **Initiation set** | States from which the option can be started. Currently the full state space (callable from anywhere). ❓ Should this be restricted? |
| **Policy** | The column's trained PPO policy. Frozen. |
| **Termination condition** | When to stop. ⚠️ See Section 6. |

---

## 6. Termination and Recursive Calling

### 6.1 Single-level call

```
MetaController (n=1)  →  calls option {0, m}
{0, m} runs
{0, m} terminates
MetaController (n=1) regains control
```

### 6.2 Recursive call ⚠️

```
MetaController (n=2)  →  calls option {1, k}
{1, k} acts as a meta-controller internally
{1, k}  →  calls option {0, m}
{0, m} terminates → signals {1, k}
{1, k} terminates → signals MC(n=2)
MC(n=2) regains control
```

Termination **propagates bottom-up.** Each level makes its own termination decision and signals the level above.

**Termination strategies (to be compared empirically):**

| Strategy | Mechanism | Risk |
|---|---|---|
| Fixed step limit | Option runs for at most K steps | K needs to be tuned per task |
| Learned termination | Option-Critic style — termination network decides | Training instability |
| GRU task-change signal | Terminate current option when GRU detects task switch | Increases inter-module coupling |

> **⚠️ Exact mechanics of recursive termination propagation to be confirmed with supervisor.**

---

## 7. Runtime Growth: New Column vs New Layer

### 7.1 Opening a new column (expanding within current layer)

Trigger: AE reconstruction error exceeds threshold **and/or** current policy reward has plateaued.

```
{n, m} → system detects new task → {n, m+1} opens → trains → stabilizes → added as option
```

### 7.2 Opening a new layer (hierarchy deepens)

Trigger: Current task requires a **combination** of primitive options from the current layer.

**Current position:** This is the "category detection" problem — determining which signal should open a new layer is an open research question. New layer decisions will be **manual/explicit** throughout Phase 1-2. Addressed in Phase 3.

> **⚠️ To be discussed with supervisor.**

---

## 8. End-to-End Inference Flow

```
1. Receive observation: obs_t

2. AE: reconstruction_error(obs_t)
   - High → new task → start new column process
   - Low  → proceed to step 3

3. GRU: task_id = classify(obs_{t-N:t})
   - Which task are we in?

4. MetaController (topmost active layer):
   - Input: obs_t + task context
   - Output: which option to call

5. Selected option runs:
   - Can recursively run the same flow internally
   - Stops when termination condition is met

6. Reward returned to MetaController
   - Sub-layer reward signal and MetaController reward signal kept SEPARATE
   - Never merged
```

---

## 9. Implementation Requirements (for PN-3)

The `Column` class must carry the following fields to implement this design:

```python
class Column:
    n: int                              # layer index
    m: int                              # column position
    policy: PPOPolicy                   # trained policy
    frozen: bool                        # True = no gradient
    lateral_source: Column | None       # ref to {n, m-1}
    is_option: bool                     # has been added to MetaController?
    sub_layer: MetaController | None    # None = leaf (n=0)
```

`sub_layer = None` → primitive column (leaf node)  
`sub_layer = MetaController(...)` → this column is both a policy and a hierarchy manager

This is a recursive definition — a column can contain a sub-MetaController, which manages more columns. If this is not designed in from the start, PN-3 will require a rewrite later.

---

## 10. Open Questions — Supervisor Meeting Agenda

| # | Question | Why it matters |
|---|---|---|
| H-2 | How exactly does termination propagation work in recursive MC→MC option calls? | PN-3 implementation depends on this |
| H-3 | Do lateral connections remain active when a frozen column is used as an option during inference? | Changes forward pass behavior |
| H-4 | What signal triggers a new layer? Manual or automatic? | Determines Phase 2 scope |
| H-5 | Should the option initiation set be restricted, or is everywhere-initiation acceptable? | Affects MetaController exploration |
