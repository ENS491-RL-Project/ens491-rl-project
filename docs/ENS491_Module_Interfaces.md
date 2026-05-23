# ENS 491 — Module Interface Specification

> Input/output contracts for every module. This is the reference during parallel development.  
> Module internals can change — but these contracts do not.  
> If a contract needs to change, update this document first, then the code.

---

## Common Types

```python
import torch
from dataclasses import dataclass
from typing import Optional

# MiniGrid observation: 7×7×3 uint8 (after FlatObsWrapper: 147-dim float vector)
Observation = torch.Tensor       # shape: (147,) — flattened
ObsSequence  = torch.Tensor       # shape: (N, 147) — last N observations
Action       = int                # discrete: 0-6 (MiniGrid action space)
TaskID       = int                # 0-indexed, -1 = unknown
LayerIdx     = int                # n — hierarchy layer
ColumnIdx    = int                # m — column position within layer
```

---

## Module 1: Autoencoder (AE)

**Responsibility:** "Have I seen this observation before?"

```python
@dataclass
class AEOutput:
    reconstruction_error: float       # MSE — low = familiar, high = novel
    is_novel: bool                    # True if error > threshold
    latent_z: torch.Tensor            # shape: (latent_dim,) — for downstream use

class AutoencoderModule:
    def encode(self, obs: Observation) -> torch.Tensor:
        """Compress observation into latent space."""
        ...

    def decode(self, z: torch.Tensor) -> Observation:
        """Reconstruct observation from latent vector."""
        ...

    def forward(self, obs: Observation) -> AEOutput:
        """Main interface. Called every step."""
        ...

    def update_threshold(self, recent_errors: list[float]) -> None:
        """Update detection threshold."""
        ...
```

**Notes:**
- `is_novel = True` → skip GRU, trigger new column process
- `is_novel = False` → pass to GRU
- `latent_z` may be unused initially but kept for downstream use (GRU input, task representation)
- Threshold strategy (fixed / adaptive / statistical) is encapsulated inside AE — callers only see `is_novel`

---

## Module 2: GRU Task Identifier

**Responsibility:** "Which task is this?"

```python
@dataclass
class GRUOutput:
    task_id: TaskID                   # predicted task ID (-1 = uncertain)
    confidence: float                 # [0.0, 1.0]
    all_probs: torch.Tensor           # shape: (num_known_tasks,) — softmax output

class GRUTaskIdentifier:
    def forward(self, obs_sequence: ObsSequence) -> GRUOutput:
        """Take last N observations, predict task."""
        ...

    def register_new_task(self, task_id: TaskID) -> None:
        """Expand output dimension when a new task is learned."""
        ...
```

**Notes:**
- `obs_sequence` is the last N steps — N=20
- `task_id = -1` → GRU uncertain, system behavior TBD
- GRU is called only when AE returns `is_novel = False`
- Trained with supervised labels in Phase 1

---

## Module 3: Progressive Networks Column

**Responsibility:** "How do I solve this task?" + provide lateral transfer

```python
@dataclass
class ColumnOutput:
    action: Action                              # selected action
    value: float                                # PPO value estimate
    activations: dict[int, torch.Tensor]        # layer_idx → activation (for lateral)

class Column:
    n: LayerIdx
    m: ColumnIdx
    frozen: bool
    lateral_source: Optional['Column']          # ref to {n, m-1}, None if first column
    sub_layer: Optional['MetaController']       # None = leaf (primitive), else recursive

    def forward(self, obs: Observation) -> ColumnOutput:
        """
        Two behaviors depending on sub_layer:

        Leaf (sub_layer=None):
            - Process observation, produce action directly.
            - If lateral_source exists, pull its activations and integrate.
            - If frozen=True, no gradient computed.

        Non-leaf (sub_layer=MetaController):
            - This Column does NOT produce an action directly.
            - Calls sub_layer.select_option(obs).
            - Selected option runs its own forward() (recursive).
            - Action from the deepest leaf propagates upward.
            - This Column's policy weights are used for meta-controller training,
              not for direct env actions.
        """
        ...

    def freeze(self) -> None:
        """Freeze weights. Called when training is complete."""
        ...

    def get_activations(self) -> dict[int, torch.Tensor]:
        """Return intermediate activations from last forward pass. Used by lateral connections."""
        ...

    def as_option(self) -> 'Option':
        """Wrap this column as an Option."""
        ...
```

**Notes:**
- `frozen=True` → no gradient, but forward pass still runs (required for lateral connections) 🔒
- `activations` dict is read via `get_activations()` by the next column
- `sub_layer=None` → leaf node (n=0, primitive skill)
- `sub_layer=MetaController(...)` → this column is both a policy and a hierarchy manager
- `sub_layer` field must exist even if `None` in Phase 1 — do not remove it

---

## Module 4: Option Wrapper

**Responsibility:** Wrap a Column into the Sutton et al. (1999) option format

```python
@dataclass
class OptionStepOutput:
    action: Action
    terminated: bool        # did this option terminate on this step?
    info: dict              # debug — step count, internal reward, etc.

class Option:
    column: Column
    option_id: int
    step_count: int         # steps elapsed since this option was invoked

    def step(self, obs: Observation) -> OptionStepOutput:
        """Run one step, check termination."""
        ...

    def reset(self) -> None:
        """Reset state at the start of a new invocation."""
        ...

    def can_initiate(self, obs: Observation) -> bool:
        """
        Initiation set check.
        Currently always True — option can start from any state.
        """
        return True
```

**Notes:**
- `terminated=True` → MetaController regains control
- Termination strategy (fixed step limit / learned / GRU signal) is encapsulated inside Option
- `step_count` is used by the fixed step limit strategy

---

## Module 5: Meta-Controller

**Responsibility:** "Which option should I call, and when?"

```python
@dataclass
class MetaControllerOutput:
    selected_option_id: int
    option: Option

class MetaController:
    n: LayerIdx                       # which layer this MC manages
    available_options: list[Option]

    def select_option(self, obs: Observation) -> MetaControllerOutput:
        """Select an option using PPO policy."""
        ...

    def add_option(self, option: Option) -> None:
        """
        Called when a new column stabilizes.
        Uses exploration bonus to encourage discovery of the new option.
        Does NOT retrain from scratch.
        """
        ...

    def step(self, obs: Observation, reward: float, done: bool) -> None:
        """PPO update step."""
        ...
```

**Notes:**
- MetaController reward is kept **separate** from sub-policy rewards — never merged 🔒
- `add_option()` does not trigger retraining — adds exploration bonus on top of existing policy
- A MetaController can itself be wrapped as an Option (enables recursive hierarchy)

---

## Module 6: Task Lifecycle Manager

**Responsibility:** Orchestration — controls which module is active and when

```python
@dataclass
class SystemState:
    current_task_id: TaskID
    active_column: Optional[Column]
    active_option: Optional[Option]
    is_training: bool

class TaskLifecycleManager:
    ae: AutoencoderModule
    gru: GRUTaskIdentifier
    columns: dict[tuple[LayerIdx, ColumnIdx], Column]
    meta_controllers: dict[LayerIdx, MetaController]

    def step(self, obs: Observation) -> Action:
        """
        Called every env step.
        Runs the AE → GRU → MetaController → Option → action pipeline.
        """
        ...

    def on_novel_task_detected(self) -> None:
        """Called when AE returns is_novel=True. Starts new column process."""
        ...

    def on_column_stabilized(self, column: Column) -> None:
        """Called when column training completes. Wraps as option, adds to MetaController."""
        ...

    def get_state(self) -> SystemState:
        """Returns current system state for GUI and debugging."""
        ...
```

---

## Signal Flow

### Entry point (every step)

```
obs_t
  │
  ▼
[AE]──────────────────────────────────────────┐
  │ is_novel=False                             │ is_novel=True
  ▼                                            ▼
[GRU]                                  on_novel_task_detected()
  │ task_id                                    │
  ▼                                            ▼
[TaskLifecycleManager]                 new Column opened
  │                                    trained → stabilized → wrapped as option
  ▼
[recursive_step(obs, root_mc)]  ← see below
```

### Recursive execution (unbounded)

```
recursive_step(obs, MetaController):
  │
  ▼
MetaController.select_option(obs)
  │ → Option selected (wraps Column {n, m})
  ▼
Option.step(obs):
  │
  ├─ Column.sub_layer == None  (LEAF)
  │     │
  │     └─ Column.forward(obs) → action
  │           └─ env.step(action) → reward, next_obs, done
  │                 └─ MetaController.step(reward)
  │
  └─ Column.sub_layer == MetaController  (NON-LEAF)
        │
        └─ recursive_step(obs, Column.sub_layer)
              │  (same flow runs one level deeper)
              ▼
             ... (unbounded depth)
              │
              └─ deepest leaf produces action
                    └─ action propagates upward
                          └─ each level's MC receives its own reward
```

**Key point:** `TaskLifecycleManager` only calls the root `MetaController` — it does not know or care how many levels exist below. Each level manages its own sub-levels. Adding a new layer only requires setting a Column's `sub_layer` field — nothing at the root changes.

---

## Hard Constraints

| Rule | Description |
|---|---|
| AE and reward signals are separate | `reconstruction_error` and `sub_policy_reward` are never merged into a single threshold |
| Frozen = forward pass still active | `frozen=True` only blocks gradients — activations are still produced |
| Stabilization required before adding option | A column that has not stabilized must not be added to MetaController |
| Reward channels are separate | Sub-policy reward and MetaController reward travel on separate channels |
| Recursive design from the start | `sub_layer` field must exist on Column even if `None` in Phase 1 |
