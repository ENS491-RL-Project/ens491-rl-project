# ENS 491 — Target Architecture v2

> **Status:** Locked on 2026-10-01. This is the reference design for all further work.
> **Supersedes:** `ENS491_Recursive_Hierarchy_Design.md` (v1) entirely, and every conflicting rule in
> `ENS491_Project_State.md`, `ENS491_Module_Interfaces.md` and `CLAUDE.md`. Where an older doc disagrees with this one, this one wins.
>
> 🔒 = locked decision · ⚠️ Qn = pending supervisor confirmation (see `SUPERVISOR_QUESTIONS.md`) · 🧪 = empirical, implement alternatives and compare

---

## 0. Why v2 exists

The project goal never changed: a **hierarchical, unbounded, option-based continual learning agent** in which every learned task becomes a reusable skill, and more complex skills are built by composing earlier ones, to arbitrary depth.

v1 had the right goal but three problems that stopped the implementation at "layer-0 columns + one ε-greedy MetaController":

1. It mixed two models of a higher level: a free-standing MetaController per layer (`meta_controllers[n]`) and a column that owns a sub-MetaController (`Column.sub_layer`). Neither was fully specified, so neither was built.
2. It tied four independent concerns into one roadmap chain: the hierarchy waited for Progressive Networks, which waited for FourRooms to stabilize. When PN-4 failed on FourRooms, everything behind it stopped.
3. Several locked rules blocked the design they were meant to protect: tree-only structure, options added to a growing MetaController (which made a PPO MetaController impossible), and a stabilization criterion that some MiniGrid environments cannot satisfy (see §5.3).

v2 fixes the model (§2–§4), separates the concerns (§1), and puts the hierarchy on the critical path (§12).

---

## 1. Separation of concerns 🔒

| Concern | Question it answers | Mechanism in v2 |
|---|---|---|
| **Architecture** | How are skills represented, composed and executed? | Skill nodes, DAG, active execution stack, options, SMDP |
| **Learning mechanism** | How is one skill's policy trained? | PPO (SB3 / sb3-contrib), optional Progressive Network laterals |
| **Task recognition** | Is this situation new? Which known task is it? | Autoencoder (novelty), GRU (identification) |
| **Growth strategy** | When is a new node created, and of which kind? | Manual curriculum first, then a heuristic (§8) |

Each concern can be changed without changing the others. The roadmap has one track per concern (§12), and only the architecture track is on the critical path.

---

## 2. Skill Node 🔒

Everything the agent knows is a **skill node**. There are no other policy-holding objects.

```python
class SkillNode:
    id: str                       # "{n,m}" plus a readable name, e.g. "{1,1} unlock_door"
    n: int                        # level, DERIVED (see below), never chosen by hand
    m: int                        # creation index within level n (1-indexed)
    children: tuple[SkillNode]    # () for primitive nodes; immutable after creation
    policy: Policy                # primitive: obs -> env action; composite: obs -> child index
    initiation: Predicate         # I(s): may this node start in state s?  (default: always True, ⚠️ Q5)
    goal: Predicate               # β(s): has this node succeeded?  (the terminal condition)
    timeout: int                  # max env steps per invocation; exceeding it = failure
    status: NodeStatus            # DECLARED -> TRAINING -> STABLE -> FROZEN  (or ABANDONED)
    lateral_sources: tuple[SkillNode]  # PN laterals, primitive nodes only (§7); may be empty
    meta: dict                    # env spec, training/eval stats, creation time, ...
```

**Two kinds, one class:**

| | Primitive node | Composite node |
|---|---|---|
| `children` | empty | non-empty, all FROZEN at creation time |
| policy output | MiniGrid action (0–6) | index into `children` |
| level `n` | 0 | `1 + max(child.n for child in children)` |
| trained as | MDP (PPO on the base env) | SMDP (PPO on `OptionEnv`, §5.2) |

**Task representation.** The supervisor's requirement "initial conditions + terminal conditions + a policy between them" maps one-to-one onto `initiation`, `goal` (+ `timeout`) and `policy`. This holds at every level.

**What "MetaController" means now.** There is no MetaController object per layer. The option-selection policy of a composite node *is* that node's meta-controller. The name `MetaController` survives only as the class of composite-node policies.

**Notation `{n,m}`.** `n` is the derived level (abstraction height); `m` is the order of creation within that level. A node's position is a consequence of what it is built from, not a slot someone assigns.

---

## 3. Graph structure: a DAG 🔒

- A composite node may use **any earlier FROZEN node** as a child, at any level. Example: `{2,1} solve_doorkey` uses `{1,1} unlock_door` (level 1) and `{0,3} go_to_goal` (level 0). No dummy wrappers are needed to "lift" a primitive to the right layer.
- A node may have **several parents**. `{0,1} get_key` can be a child of `unlock_door` and of any later node that needs a key. This is safe because children are frozen: parents reference them and never change their weights.
- **Acyclic by construction:** a parent can only be created after all its children are FROZEN, so creation order is a topological order.
- **Children are immutable.** A node's child set is fixed when the node is created. If a better combination becomes possible later, a *new* node is created. Consequence: every composite policy has a fixed `Discrete(len(children))` action space for its whole life, which is what makes PPO training of meta-controllers straightforward (the reason it was deferred in v1 disappears).
- **Unbounded** means: nothing in the code branches on `n`, and nothing caps depth or width. Depth grows only because tasks are composed from earlier skills. The claim is demonstrated by (a) a unit test over a synthetic depth-5 graph and (b) real stacks of depth ≥ 3 in the experiments (§11).

---

## 4. Execution semantics 🔒

### 4.1 Options persist until they terminate

A selected child keeps control until **its own** termination. The hierarchy is not re-decided from the root on every env step (that would remove temporal abstraction). The runtime keeps an **active stack** `[root, child, grandchild, …, primitive]`.

### 4.2 Termination

A node in the stack terminates when, checked at the start of each env step:

- `goal(state)` is true → outcome `SUCCESS`
- its own step counter reaches `timeout` → outcome `TIMEOUT` (failure; the parent's SMDP return reflects the wasted steps)
- an **ancestor** terminates → outcome `INTERRUPTED` (the whole sub-stack under a finished node is closed)

Termination is checked **top-down**: the highest terminated node is closed together with everything below it, and control returns to its parent, which selects again. If the root terminates, the episode for that task is over and the lifecycle manager decides what happens next.

`goal` and `initiation` are predicates over the **environment state** (`env.unwrapped`: carried object, door state, agent position), not over the policy's observation. They are task definitions supplied by the environment designer (⚠️ Q1). Policies only ever see the 147-dim observation.

### 4.3 Availability

A node is **available** in state `s` iff

```
initiation(s) and not goal(s) and (node.is_primitive or any(c.available(s) for c in node.children))
```

Unavailable children are masked at selection time. The second term prevents the classic zero-length-option loop (selecting a child whose goal already holds would terminate instantly, forever). The third term, evaluated recursively, prevents selecting a composite that could not select anything itself. If a composite already on the stack finds no available child (the state changed under it), it terminates with `TIMEOUT` and its parent selects again; the recursive rule stops the parent from re-selecting it in the same state.

### 4.4 The executor (depth-agnostic)

```python
@dataclass
class Frame:
    node: SkillNode
    steps: int = 0        # env steps since this node was pushed
    ret: float = 0.0      # discounted return since push (logged; used by OptionEnv)

class Executor:
    """One executor serves runtime, evaluation and OptionEnv. No code path depends on node.n."""

    def __init__(self, root: SkillNode):
        self.stack = [Frame(root)]

    def act(self, obs, state) -> int | None:
        # 1. Termination, top-down: close the highest finished node and everything under it.
        for k, frame in enumerate(self.stack):
            outcome = frame.node.check_termination(state, frame.steps)  # SUCCESS / TIMEOUT / None
            if outcome is not None:
                self._close_from(k, outcome)       # logs outcomes; descendants get INTERRUPTED
                break
        if not self.stack:
            return None                            # root finished

        # 2. Selection, top-down: descend until a primitive is on top.
        while not self.stack[-1].node.is_primitive:
            parent = self.stack[-1].node
            mask = [c.available(state) for c in parent.children]
            if not any(mask):
                self._close_from(len(self.stack) - 1, TIMEOUT)
                return self.act(obs, state)
            idx = parent.policy.select(obs, mask)
            self.stack.append(Frame(parent.children[idx]))

        # 3. Act.
        return self.stack[-1].node.policy.act(obs)

    def observe_reward(self, r: float) -> None:
        for frame in self.stack:
            frame.ret += GAMMA ** frame.steps * r
            frame.steps += 1
```

The `_active_option` TODO from v1 (`task_lifecycle.py`) is exactly this stack.

---

## 5. Learning semantics

### 5.1 Primitive nodes 🔒

Trained with SB3 PPO on a `SubgoalEnv` wrapper around a MiniGrid env:

- `reset()` applies the node's **initial-condition hook** (e.g. `open_door` starts with the key already in hand).
- reward = `1 − 0.9 · (steps / timeout)` when `goal(state)` becomes true, else 0 (same shape as MiniGrid's own reward).
- `terminated` when `goal(state)`, `truncated` at `timeout`.

Observation pipeline: `ImgObsWrapper` → `FlattenObservation` → `(147,)`. Never `CnnPolicy` (7×7 obs is smaller than SB3's default 8×8 kernel). (Older docs said `FlatObsWrapper`; that wrapper adds the mission string and does **not** produce 147 dims. The code has always used the pipeline above.)

The existing `ColumnPolicy` / `LateralMlpExtractor` (147 → 64 → 64, optional laterals) remains the primitive policy network.

### 5.2 Composite nodes: SMDP via `OptionEnv` 🔒

A composite node in TRAINING is trained with an unmodified PPO implementation on a wrapper whose actions are children:

```python
class OptionEnv(gym.Wrapper):
    """The SMDP seen by a composite node: one step = one complete child execution."""
    action_space = Discrete(len(node.children))

    def step(self, idx):
        child = node.children[idx]
        ex = Executor(child)                       # same executor, child as root
        R, k = 0.0, 0
        while True:
            a = ex.act(self.obs, self.state)
            if a is None:                          # child finished: SUCCESS or TIMEOUT
                break
            self.obs, r, term, trunc, info = self.env.step(a)
            ex.observe_reward(r); R += GAMMA ** k * r; k += 1
            if term or trunc or node.goal(self.state):   # env end or the parent's own goal interrupts the child
                break
        return self.obs, R, node.goal(self.state) or term, trunc or self._decision_cap(), {"k": k, ...}

    def action_masks(self):                        # consumed by MaskablePPO
        return [c.available(self.state) for c in node.children]
```

- **Algorithm:** `MaskablePPO` (sb3-contrib) with `action_masks()` from §4.3. New dependency: `sb3-contrib`.
- **Discounting:** within an option, rewards are discounted per env step (`R` above). Between decisions, PPO's ordinary per-step γ is used instead of the exact γᵏ. This is the standard practical approximation. Exact γᵏ discounting is an optional refinement 🧪.
- **Decision cap:** at most `max_decisions` child calls per episode (truncation), as a guard against degenerate policies.
- Children are frozen, so the SMDP's dynamics are stationary while the parent trains.

### 5.3 Stabilization criterion (replaces mean > 0.85, std < 0.05) 🔒

> Every `eval_every` training steps, run `N_eval = 50` evaluation episodes in execution mode (§5.4) on the node's training env.
> **Success rate** = fraction of episodes in which `goal` is reached before `timeout`.
> A node is **STABLE** when the success rate is **≥ 0.90 in two consecutive evaluations**.
> Mean episode length is reported alongside but is not part of the criterion.

Why the old rule had to go: MiniGrid's reward depends on path length. In environments with random start and goal positions (FourRooms), a policy that succeeds 100 % of the time typically still has reward std above 0.05, because path lengths differ from episode to episode, so the old rule can be impossible to satisfy regardless of learning quality. The PN-4 FourRooms run also used a relaxed 0.75 threshold in `train_pn4.py` while docs said 0.85; v2 retires both numbers.

The thresholds (0.90, two evaluations, N = 50) are defaults, fixed per experiment and reported 🧪. The same rule applies to primitive and composite nodes.

### 5.4 Freezing and execution mode 🔒

- STABLE → FROZEN: `requires_grad_(False)` on all policy parameters. The forward pass keeps running (needed by laterals and by parents).
- A frozen node is never modified, for any reason.
- **Execution mode:** frozen nodes act **deterministically** by default. Stabilization is evaluated in the same mode. (Fixes v1's `deterministic=not self.frozen`, which made frozen columns sample stochastically.) Deterministic MiniGrid policies can loop under partial observability; `timeout` bounds the damage, and stochastic execution is a 🧪 alternative.

---

## 6. Task recognition (AE + GRU)

AE and GRU live in the **lifecycle layer**. They never select children inside the skill graph. 🔒

### 6.1 Autoencoder: "is this situation new?"

- Output: reconstruction error → novelty flag. It is one of the two growth signals (§8); it is **never merged** with the reward signal 🔒.
- Input is **normalized per channel** (MiniGrid image channels are object index 0–10, color 0–5, state 0–2) before the MLP AE (147 → 64 → 32 → 64 → 147). In v1 raw integers went in, which made the fixed 0.05 threshold meaningless.
- Threshold: calibrated per known task on familiar data (mean + k·σ), then frozen during inference.
- 🧪 Single global AE vs **one AE per task** (as in the base paper, Erden et al. 2025). A single AE fine-tuned on new tasks forgets old ones, which makes old tasks look novel; per-task AEs do not.

### 6.2 GRU: "which known task is this?"

- Classifies the last N observations into a known task id; `-1` when confidence < threshold.
- **Root selection:** the lifecycle manager keeps a registry `task_id → root SkillNode`. The GRU's task id chooses which root the executor runs. (In v1 the task id was computed and then ignored.)
- 🧪 N ∈ {5, 20, 50}: accuracy vs switching latency.

### 6.3 Where continual-learning forgetting actually happens

Skill nodes cannot forget: they are frozen. The **recognition layer can**: a GRU head retrained on each new task, or a single shared AE, may degrade on earlier tasks. Backward-transfer measurements are therefore meaningful for task-identification accuracy and novelty detection, and are reported there (§11, R-2). GRU retraining protocol is ⚠️ Q7.

---

## 7. Progressive Networks: forward transfer, not forgetting 🔒

- **Forgetting of skills is prevented by freezing**, independent of PN. A library of frozen PPO policies (the base paper's design) already has zero skill forgetting.
- **PN's contribution is forward transfer:** a new primitive node may read hidden activations of earlier frozen primitives through lateral connections (`nn.Linear(64, 64)` added to its own hidden layer, as implemented today). The claim is measured as steps-to-stabilization with vs without laterals (C-2).
- **Lateral sources** are chosen when a primitive node is created, from FROZEN primitive nodes. Default: the most recently created primitive (equivalent to v1's `{0,m-1} → {0,m}`). 🧪 Alternatives: all earlier primitives; the primitive whose AE latent is closest.
- Laterals between **composite** nodes are out of scope for v2.
- PN is **not** on the critical path. The hierarchy works with plain PPO primitives.

---

## 8. Growth strategy (a heuristic, not an architectural law)

The architecture allows creating a primitive or a composite node at any time. *When* and *which* is a separate, replaceable strategy.

**Phase A: manual curriculum (now).** The experimenter declares each task: env, initial-condition hook, goal predicate, timeout, node kind, and children (for composites). Everything after declaration (train → evaluate → stabilize → freeze → register) runs automatically through one lifecycle API (G-1).

**Phase B: composite-first heuristic (to evaluate, ⚠️ Q3).** When a new task arrives (declared, or flagged by the AE):

1. Create a composite candidate whose children are the currently FROZEN nodes. Train with budget `B_c`.
2. If it stabilizes, register it. Otherwise create a primitive candidate (optionally with laterals) and train with budget `B_p`.
3. Optionally retry step 1 once the new primitive is frozen.

"The composite failed, so a primitive is missing" is **not** a valid inference on its own: failure can also come from exploration, hyperparameters, timeouts, budget, the child subset, or partial observability. Phase B is reported as a heuristic with an ablation (composite-first vs primitive-first), never as a solved problem.

**Known caveat:** with "all frozen nodes" as children, a new composite's structural level is `1 + max level in the library` even if it only uses low-level children. Phase B therefore also reports an **effective level** computed from the children actually selected during evaluation. In Phase A, children are chosen deliberately, so structural and effective levels coincide.

---

## 9. Node lifecycle and interfaces

```
DECLARED ──train──▶ TRAINING ──eval: SR ≥ 0.90 twice──▶ STABLE ──freeze──▶ FROZEN ──▶ usable as a child / as a root
                       │
                       └── budget exhausted ──▶ ABANDONED (kept in the log, never used)
```

- Only FROZEN nodes may be children or registered roots 🔒 (v1's "never add an unstabilized option" rule, kept).
- **Importing external policies** (supervisor requirement for the GUI): an externally trained policy is imported as a FROZEN primitive node by supplying its goal predicate, initiation predicate and timeout. It then behaves like any other node.
- **Planning hook (Phase 3 stretch):** every node declares `initiation` and `goal`, so a planner can chain nodes whose goal satisfies the next node's initiation. No v2 component depends on this.

---

## 10. Code migration map (current repo → v2)

| Current (v1) | v2 | Action |
|---|---|---|
| `Column(n, m, lateral_source, sub_layer)` | `SkillNode` (primitive) wrapping `ColumnPolicy` | Replace `sub_layer` with `children`; `n` derived; `lateral_source` → `lateral_sources` |
| `Column.forward(... deterministic=not self.frozen)` | deterministic when frozen | **Bug fix**, do first |
| `Option` (`MAX_STEPS=200`, `can_initiate → True`) | node's `goal` / `initiation` / `timeout` + `Frame` | Fold into `SkillNode` + `Executor` |
| `MetaController` (per-layer object, ε-greedy, greedy = option 0) | policy class of composite nodes, trained with MaskablePPO on `OptionEnv` | Rewrite; delete `meta_controllers[n]` dict |
| `TaskLifecycleManager` (single level, global `_active_option`, `n=0` hardcoded) | executor stack + task registry + lifecycle API | Rewrite |
| `StabilizationMonitor` (mean / std of reward) | `EvaluationMonitor` (success rate, §5.3) | Replace |
| `ColumnTrainer` / `pn_trainer.py` | generic node trainer: `SubgoalEnv` (primitive) or `OptionEnv` (composite) | Generalize |
| `AutoencoderModule` | + per-channel input normalization, + calibration step | Extend |
| `GRUTaskIdentifier` | + task registry wiring | Extend |
| `scripts/demo_run.py` (mock policies, `randn` obs) | real experiment scripts | Retire once H-6 produces real runs |
| GUI (reads `demo_state.pkl`) | reads real run directories: graph, active stack, traces | Rewrite panels (UI-1) |
| `runs/pn4/col01` (Empty-8x8, stable) | legacy primitive; lateral source for `go_to_goal` in C-2 | Keep |

Tests: existing v1 tests stay green until the module they cover is replaced; each replacement lands with its own tests.

---

## 11. Evaluation plan: claims → experiments

All env-step counts include steps executed inside frozen children. Pretraining cost of reused children is reported separately (it is amortized across every task that reuses them).

| Claim | Experiment | Metric | Baselines |
|---|---|---|---|
| **C1 Composition helps.** A composite over frozen skills learns a task faster than learning it flat. | `{2,1} solve_doorkey` on DoorKey | success rate vs env steps; steps to STABLE | flat PPO on DoorKey; flat composite over the three primitives |
| **C2 Reuse across tasks and depth (unbounded).** A skill learned once is reused by later tasks, and deeper composition reuses intermediate skills. | UnlockPickup and BlockedUnlockPickup reusing `unlock_door`; BlockedUnlockPickup variant A (level 2) vs variant B (level 3) | same as C1; stack depth reached | composite from primitives only; flat PPO |
| **C3 Task recognition.** The AE separates known from novel tasks; the GRU identifies known tasks. | observations from all curriculum envs | AE ROC-AUC per task pair; GRU accuracy and latency vs N; recognition accuracy on old tasks after each new task | single vs per-task AE |
| **C4 Continual learning.** Skills are retained; laterals give forward transfer. | re-evaluate every FROZEN node after each new task; primitives with vs without laterals | retention (expected unchanged, by construction); steps to STABLE | no laterals; (optional) EWC fine-tuning |
| **C5 Introspection.** Internal state is inspectable. | GUI on real runs | qualitative: graph, active stack, traces, per-node stats | — |

### Candidate curriculum (⚠️ Q6; env IDs to be verified against the installed `minigrid` version in H-2)

| # | Node | Kind | Children | Env | Goal predicate |
|---|---|---|---|---|---|
| T1 | `{0,1} get_key` | primitive | — | Unlock (+ DoorKey layouts) | agent carries the key |
| T2 | `{0,2} open_door` | primitive (I: carrying key) | — | Unlock, reset with key in hand | locked door is open |
| T3 | `{1,1} unlock_door` | composite | T1, T2 | `MiniGrid-Unlock-v0` | door open (env success) |
| T4 | `{0,3} go_to_goal` | primitive | — | DoorKey, reset with door open | agent on goal tile |
| T5 | `{2,1} solve_doorkey` | composite | T3, T4 | `MiniGrid-DoorKey-6x6-v0` | env success |
| T6 | `{0,4} pickup_box` | primitive | — | UnlockPickup, reset with door open | agent carries the box |
| T7 | `{2,2} unlock_pickup` | composite | T3, T6 | `MiniGrid-UnlockPickup-v0` | env success |
| T8 | `{0,5} move_blocker` | primitive | — | BlockedUnlockPickup | cell in front of the door is free |
| T9a | `{2,3}` | composite | T8, T3, T6 | `MiniGrid-BlockedUnlockPickup-v0` | env success |
| T9b | `{3,1}` | composite | T8, T7 | same | env success |

T5 already exercises a mixed-level DAG and a depth-3 stack (`solve_doorkey → unlock_door → get_key`). T9a vs T9b is the depth-reuse comparison. 🧪 Primitive generality: train primitives on a mix of the layouts where their goal predicate is defined, so they work inside every parent task.

FourRooms is no longer part of the curriculum. It survives only as a diagnostic (C-1).

---

## 12. Roadmap: four tracks

Story points as in `ENS491_Project_State.md`. Status lives there; this table defines the work.

**Track H: Hierarchy (critical path)**

| # | Task | SP |
|---|---|---|
| H-0 | Fix `Column.forward` deterministic flag; document the 0.75 / 0.85 mismatch in the run log | 1 |
| H-1 | `SkillNode`, `Frame`, `Executor` + unit tests with scripted policies on a toy env: depth-5 synthetic graph, interrupt semantics, availability masking, timeout | 8 |
| H-2 | Predicates + `SubgoalEnv` (initial-condition reset hooks) for T1, T2, T4; verify MiniGrid env IDs | 5 |
| H-3 | `EvaluationMonitor` (§5.3); re-evaluate legacy `col01` under the new criterion | 3 |
| H-4 | Train and freeze T1, T2, T4 | 5 |
| H-5 | `OptionEnv` + MaskablePPO; train and freeze T3 `unlock_door` | 8 |
| H-6 | T5 `solve_doorkey` + C1 baselines → **first hierarchical result** | 8 |
| H-7 | T6–T9b: reuse and depth experiment (C2) | 8 |

**Track C: Continual learning / PN (parallel)**

| # | Task | SP |
|---|---|---|
| C-1 | FourRooms diagnosis: plain PPO, no laterals, same budget. Classifies the PN-4 failure for the report | 2 |
| C-2 | Forward transfer: `go_to_goal` with vs without lateral from legacy `col01`; repeat on T1/T2 | 5 |
| C-3 | Retention check: re-evaluate all FROZEN nodes after each new task | 2 |

**Track R: Recognition (parallel)**

| # | Task | SP |
|---|---|---|
| R-1 | AE normalization + calibration; error distributions and ROC-AUC per curriculum env; single vs per-task AE | 5 |
| R-2 | GRU dataset from curriculum envs; accuracy vs N; old-task accuracy after each new task | 5 |
| R-3 | Task registry: GRU task id → root node in the lifecycle manager | 3 |

**Track G: Growth (after H-6)**

| # | Task | SP |
|---|---|---|
| G-1 | Lifecycle API: declare → train → stabilize → freeze → register, with no hand-wiring | 5 |
| G-2 | Composite-first heuristic + ablation (§8) | 8 |
| G-3 | AE-triggered task declaration | 5 |

**Track UI (after H-1)**

| # | Task | SP |
|---|---|---|
| UI-1 | GUI on real runs: skill DAG, live active stack, execution trace, per-node stats | 8 |
| UI-2 | Import an external policy as a FROZEN primitive node | 3 |

**Stretch:** P-1 planning over `initiation` / `goal` (§9), 13 SP.

```
Critical path:  H-0 → H-1 → H-2 → H-3 → H-4 → H-5 → H-6 → H-7
Parallel:       C-1, C-2, C-3   ·   R-1, R-2, R-3   ·   UI-1 (after H-1)
After H-6:      G-1 → G-2 → G-3
```

---

## 13. Changelog v1 → v2

| v1 rule | v2 rule | Reason |
|---|---|---|
| Hierarchy is a tree; each column belongs to one MetaController | DAG; a frozen node may have many parents | Reuse is the point of the project; frozen children make sharing free |
| One MetaController object per layer (`meta_controllers[n]`) | The policy of each composite node is its meta-controller | Removes the two-model ambiguity |
| Recursion via `Column.sub_layer` | Recursion via immutable `SkillNode.children` | Explicit, depth-agnostic |
| Layer chosen at creation; MC sees only its own layer | Level derived: `1 + max(child levels)`; children from any level | No dummy wrappers |
| New option added to an existing MC with an exploration bonus | New composite node with a fixed child set | Fixed action space → PPO works |
| MetaController PPO deferred (dynamic action space) | MaskablePPO on `OptionEnv` | Enabled by immutable children |
| Lateral connections between children of the same MC | Laterals among primitive nodes, sources chosen at creation | "Same MC" is undefined in a DAG |
| Stabilization: mean > 0.85, std < 0.05 (code used 0.75) | Success rate ≥ 0.90 in two consecutive evaluations | Reward std depends on path length |
| Option termination: fixed 200 steps | Goal predicate; timeout as failure; ancestor interrupt | Temporal abstraction with real subgoals |
| Initiation set: everywhere | Predicate (default everywhere) + availability masking | Prevents zero-length option loops |
| New layer vs new column: manual, Phase 3 | Growth track: manual curriculum now, heuristic later | Separate concern |
| PN on the critical path | PN is an independent track (forward transfer) | Hierarchy must not wait on FourRooms |
| Multi-level hierarchy = Phase 3 stretch | Multi-level hierarchy = core result (H-6, H-7) | It is the project's main claim |
| `FlatObsWrapper` | `ImgObsWrapper` + `FlattenObservation` → (147,) | Matches the actual code |

**Unchanged and still locked:** AE and reward signals are never merged; frozen means no gradients but a running forward pass; only stabilized nodes are reused; Python 3.10, PyTorch, MiniGrid, Gymnasium, SB3, W&B, Streamlit; never `CnnPolicy`; nothing hardcodes depth or width.
