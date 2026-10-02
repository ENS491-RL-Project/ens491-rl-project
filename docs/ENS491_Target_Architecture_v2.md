# ENS 491 — Target Architecture v2

> **Status:** Locked on 2026-10-01; **amended 2026-10-01 (H-0.5)** with the corrections in
> `plans/2026-10-01_v2_H0-H5_plan.md` Part A (items F1-F34). Amended passages are marked **[A: Fnn]**.
> This is the reference design for all further work.
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
    lateral_sources: tuple[SkillNode]  # PN laterals, primitive nodes only (§7); len <= 1 for now [A: F20]
    spec_hash: str                # identity: hash of the spec this node was trained under [A: F18]
    meta: dict                    # env spec, training/eval stats, creation time, ...
```

**Node identity [A: F18].** A frozen node is reused across experiments, so a checkpoint must carry the
spec it was trained under. `spec_hash` is a hash over: env id(s), the source of the `goal` and
`initiation` predicates, `timeout`, the observation pipeline (including the normalization flag of
§5.1), and the `spec_hash` of every child. The node registry refuses to load a node whose stored hash
does not match the spec being requested. Without this, changing a predicate silently invalidates every
frozen node built on it — the v1 checkpoints in `runs/pn4/` already show the failure mode: their
`meta.json` records neither seed nor env id.

**Policy adapters [A: F19].** `policy` is one field but the two kinds have different call signatures, and
SB3's `ActorCriticPolicy` and sb3-contrib's `MaskableActorCriticPolicy` differ again. Two thin protocols
keep the executor free of `isinstance` branching:

```python
class PrimitivePolicy(Protocol):
    def act(self, obs, deterministic: bool = True) -> int: ...        # -> MiniGrid action 0-6

class CompositePolicy(Protocol):
    def select(self, obs, mask: Sequence[bool], deterministic: bool = True) -> int: ...  # -> child index
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
- **Acyclic by construction:** a parent can only be created after all its children are FROZEN, so creation order is a topological order. This holds only while nodes are created in order, which a graph loaded from disk need not respect, so `SkillNode.__init__` additionally runs a DFS over `children` and raises if the node is reachable from itself **[A: F21]**.
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

`goal` and `initiation` are predicates over the **environment state**, not over the policy's observation. They are task definitions supplied by the environment designer (⚠️ Q1). Policies only ever see the 147-dim observation.

**`StateView` [A: F5].** Predicates do not read `env.unwrapped` directly. `env.unwrapped` is a live mutable
object, so a predicate, a mask and the policy's observation could otherwise be evaluated at different
instants, and an execution trace written for the GUI would hold references to a mutating object rather
than a record. Instead one immutable snapshot is built per env step and passed everywhere:

```python
@dataclass(frozen=True)
class StateView:
    carrying_type: str | None; carrying_color: str | None
    agent_pos: tuple[int, int]; agent_dir: int
    doors: tuple[DoorView, ...]        # position, color, is_open, is_locked
    objects: tuple[ObjView, ...]       # position, type, color
    step_count: int
    env_terminated: bool
    version: int                       # monotonic; the cache key of §4.3
```

`version` increments once per env step, which gives the availability cache of §4.3 its key for free and
makes the trace serializable. One builder per env family (MiniGrid's `RoomGrid` envs expose
`env.unwrapped.door` and `.obj`; `DoorKey` does not, so its builder scans the grid).

**Goal must not hold at reset [A: F7].** If `goal(s₀)` is already true, the root terminates at step 0 with
`SUCCESS` and the episode silently inflates the measured success rate. `SubgoalEnv.reset` and
`OptionEnv.reset` assert `not goal(s₀)` and resample the seed if it holds.

### 4.3 Availability

A node is **available** in state `s` iff

```
not goal(s) and (node.is_primitive or any(c.available(s) for c in node.children))
```

Unavailable children are masked at selection time. The first term prevents the classic
zero-length-option loop (selecting a child whose goal already holds would terminate instantly,
forever). The second, evaluated recursively, prevents selecting a composite that could not select
anything itself. If a composite already on the stack finds no available child (the state changed under
it), it terminates with `TIMEOUT` and its parent selects again.

**`initiation` does not mask [A: F1].** In v2 as first written, availability began with `initiation(s)`.
Applied to the curriculum of §11 that makes the composite's job vacuous: in `{1,1} unlock_door`,
`get_key`'s goal (carrying the key) becomes true exactly when `open_door`'s initiation (carrying the
key) does, so there is **exactly one legal child at every decision point** and the policy has nothing
to learn. C1 would then be measuring the mask, not composition. So:

- `initiation` stays on every node — it is half of the supervisor's "initial conditions + terminal
  conditions + a policy" (⚠️ Q1, Q5) — and it drives `SubgoalEnv`'s reset hooks (§5.1) and the planning
  hook (§9). It does **not** enter the availability predicate.
- A composite may therefore select a child outside its initiation set, and that child will usually fail
  and time out. Learning that it fails is the signal the parent needs.
- C1 reports two controls alongside the composite: a policy choosing **uniformly at random over the
  available children**, and an ablation with **initiation masking switched back on**. Together they
  separate what composition contributes from what masking contributes.

**Re-selection is legal but bounded [A: F2].** A primitive that times out still has `goal(s)` false, so it
is immediately available again in an unchanged state and the parent may select it again, forever. Two
bounds, both mandatory and both recorded in `meta.json`: the parent's own `timeout`, and
`max_decisions`, a cap on child invocations per episode (no longer "a guard against degenerate
policies" — it is load-bearing). The executor counts re-selections per episode and logs the count.

**Availability is memoized [A: F4].** `available()` recurses over children, so in a DAG a node reachable
by several paths is re-evaluated once per path — exponential in depth (branching 5 at depth 5 is ~5⁵
predicate evaluations per decision). The executor memoizes on `(node, state_view.version)` and clears
the cache each env step.

**Masking is privileged, policies are not [A: F9].** Availability is computed from the `StateView`, which
sees the whole environment; the policy sees only the 147-dim partial observation, in which a door's
state is invisible unless the door is in view. This asymmetry is deliberate and legal — but it is the
reason the masking ablation above exists, and it is stated here so a reader does not have to infer it.

### 4.4 The executor (depth-agnostic)

```python
@dataclass
class Frame:
    node: SkillNode
    steps: int = 0        # env steps since this node was pushed
    ret: float = 0.0      # discounted return since push (logged; used by OptionEnv)

class Executor:
    """One executor serves runtime, evaluation and OptionEnv. No code path depends on node.n."""

    MAX_ITERS = 256       # [A: F6] bound; exceeding it is a bug, not a slow episode

    def __init__(self, root, gamma=0.99, execution_mode="deterministic", max_decisions=64):
        self.stack = [Frame(root)]
        self.gamma = gamma                       # [A: F8] no module-level GAMMA
        self.execution_mode = execution_mode     # [A: F17] deterministic | stochastic | epsilon
        self.max_decisions = max_decisions       # [A: F2] mandatory
        self.decisions = 0
        self.reselections = 0
        self.trace = []                          # [A: F28] (depth, node_id, event, outcome, steps)
        self._avail = {}                         # [A: F4] (node_id, sv.version) -> bool

    def act(self, obs, sv) -> int | None:
        """Bounded loop, not recursion [A: F6]. Returns None when the root has finished."""
        for _ in range(self.MAX_ITERS):
            # 1. Termination, top-down: close the highest finished node and everything under it.
            for k, frame in enumerate(self.stack):
                outcome = frame.node.check_termination(sv, frame.steps)   # SUCCESS / TIMEOUT / None
                if outcome is not None:
                    self._close_from(k, outcome)        # descendants get INTERRUPTED
                    break
            if not self.stack:
                return None                             # root finished

            # 2. Selection, top-down: descend until a primitive is on top.
            top = self.stack[-1].node
            if top.is_primitive:
                return top.policy.act(obs, deterministic=(self.execution_mode == "deterministic"))

            if self.decisions >= self.max_decisions:    # [A: F2]
                self._close_from(0, TIMEOUT)
                return None

            mask = [self._available(c, sv) for c in top.children]
            if not any(mask):                           # [A: F3] never hand an empty mask to a policy
                self._close_from(len(self.stack) - 1, TIMEOUT)
                continue                                # re-enter the loop; do not recurse
            idx = top.policy.select(obs, mask,
                                    deterministic=(self.execution_mode == "deterministic"))
            child = top.children[idx]
            if self._just_timed_out(child, sv):
                self.reselections += 1                  # [A: F2] legal, counted, bounded by the cap
            self.decisions += 1
            self.stack.append(Frame(child))
            self.trace.append((len(self.stack), child.id, "SELECT", None, 0))
        raise RuntimeError("Executor.act exceeded MAX_ITERS -- cyclic termination/selection")

    def _available(self, node, sv) -> bool:             # [A: F4] memoized per env step
        key = (id(node), sv.version)
        if key not in self._avail:
            self._avail[key] = node.available(sv, self._avail)
        return self._avail[key]

    def observe_reward(self, r: float) -> None:
        """[A: F8] A frame pushed at step t has steps == 0 while acting at t;
        the discount is applied before the counter advances."""
        for frame in self.stack:
            frame.ret += self.gamma ** frame.steps * r
            frame.steps += 1
```

The `_active_option` TODO from v1 (`task_lifecycle.py`) is exactly this stack.

---

## 5. Learning semantics

### 5.1 Primitive nodes 🔒

Trained with SB3 PPO on a `SubgoalEnv` wrapper around a MiniGrid env:

- `reset()` applies the node's **initial-condition hook**, derived from `initiation` (e.g. `open_door`
  starts with the key already in hand), and asserts `not goal(s0)` **[A: F7]**.
- reward = `1 − 0.9 · (steps / timeout)` when `goal(state)` becomes true, else 0 (same shape as MiniGrid's own reward).
- `terminated` when `goal(state)`, `truncated` at `timeout`.

**Reward is defined by the node's own goal — never by the env's reward [A: F10].** This rule is shared
with §5.2 and holds at every level. It is what keeps the hard rule "a parent's SMDP reward and a
child's subgoal reward travel on separate channels" true by construction.

**The reward shape is a training signal, not a metric [A: F12].** Its denominator is the *node's*
timeout, so rewards are not comparable between nodes with different timeouts, nor to a flat-PPO
baseline whose MiniGrid reward is scaled by the env's `max_steps`. **Success rate and episode length
are the reported metrics** (§5.3, §11); reward curves are diagnostic only.

Observation pipeline: `ImgObsWrapper` → `FlattenObservation` → `(147,)`. Never `CnnPolicy` (7×7 obs is smaller than SB3's default 8×8 kernel). (Older docs said `FlatObsWrapper`; that wrapper adds the mission string and does **not** produce 147 dims. The code has always used the pipeline above.)

**Per-channel normalization, for the policy too [A: F16].** The flattened observation is
`Box(0, 255, (147,), uint8)` holding values 0–10, and SB3's `MlpPolicy` does not normalize a non-image
Box — v1 fed raw integers to the MLP. A `ChannelNormalize` wrapper (object/10, color/5, state/2, the
same utility the AE uses in §6.1) is applied after `FlattenObservation` and is **on by default** for
every new node; `SubgoalEnv`, `OptionEnv` and evaluation must all use the same setting, and the
setting is part of `spec_hash` (§2). Shape stays `(147,)`; dtype becomes float32. Consequence: the
legacy `runs/pn4/col01` checkpoint was trained on raw integers and cannot serve as a lateral source
for a normalized node without being retrained (§7, C-2).

The existing `ColumnPolicy` / `LateralMlpExtractor` (147 → 64 → 64, optional laterals) remains the primitive policy network.

### 5.2 Composite nodes: SMDP via `OptionEnv` 🔒

A composite node in TRAINING is trained with an unmodified PPO implementation on a wrapper whose actions are children:

```python
class OptionEnv(gym.Wrapper):
    """The SMDP seen by a composite node: one step = one complete child execution."""
    action_space = Discrete(len(node.children))

    def step(self, idx):
        child = node.children[idx]
        ex = Executor(child, gamma=self.gamma, execution_mode=self.execution_mode)
        k = 0
        while True:
            a = ex.act(self.obs, self.sv)
            if a is None:                          # child finished: SUCCESS or TIMEOUT
                break
            self.obs, _, term, trunc, _ = self.env.step(a)   # env reward discarded [A: F10]
            self.sv = build_state_view(self.env)             # one snapshot per step [A: F5]
            ex.observe_reward(0.0); k += 1
            if term or trunc or node.goal(self.sv):          # env end or the parent's goal interrupts
                break

        self.steps += k
        self.decisions += 1
        reached = node.goal(self.sv)
        R = (1 - 0.9 * self.steps / node.timeout) if reached else 0.0   # own goal only [A: F10]
        truncated = (trunc or self.steps >= node.timeout
                     or self.decisions >= node.max_decisions            # mandatory [A: F2]
                     or not any(self.action_masks()))                   # [A: F3]
        info = {"k": k, "child_id": child.id, "child_outcome": ex.last_outcome,
                "reselections": ex.reselections}                        # [A: F14]
        return self.obs, R, reached, truncated and not reached, info

    def action_masks(self):                        # consumed by MaskablePPO
        return [c.available(self.sv) for c in node.children]
```

- **Algorithm:** `MaskablePPO` (sb3-contrib) with `action_masks()` from §4.3. New dependency: `sb3-contrib`.
- **Reward [A: F10]:** from the composite's **own** `goal`, with the same `1 − 0.9 · steps/timeout` shape
  as §5.1. The base env's reward is *not* passed through. As first written, §5.2 returned the discounted
  env reward, which is identically zero for any composite whose goal is not the env's success
  condition — such a node could never learn. It happened to work for `{1,1} unlock_door` only because
  "door open" *is* `MiniGrid-Unlock-v0`'s success condition.
- **Never an empty mask [A: F3]:** `MaskablePPO` requires at least one valid action. When no child is
  available the episode ends instead (`terminated` if the node's goal holds, else `truncated`); the
  executor closes the frame before any policy call. A mask reaching a policy is asserted to have ≥1 True.
- **Discounting:** between decisions, PPO's ordinary per-step γ is used instead of the exact γᵏ. This
  is the standard practical approximation; exact γᵏ is a 🧪 refinement. `info["k"]` is logged on
  **every** decision from the first run **[A: F14]** so that comparison never needs a re-run.
- **Timeout invariants [A: F13]:** validated at node creation — `node.timeout ≥ max(child.timeout)`, and
  for a node registered as a root, `node.timeout ≤ env.max_steps`. Without these a parent can be unable
  to succeed for reasons that look exactly like a learning failure. The budgets are tight:
  Unlock 288, DoorKey-6x6 360, UnlockPickup 288, BlockedUnlockPickup 576. All timeouts go in `meta.json`.
- **Vectorization: `DummyVecEnv` only [A: F15].** Frozen children are shared `nn.Module`s across env
  copies, and `LateralMlpExtractor` stored activations in a single `self._acts` dict — last writer
  wins. `_extract` is therefore refactored to **return** activations rather than store them
  (`get_activations()` is kept for the GUI). `SubprocVecEnv` is not used: it would need every frozen
  child loaded per subprocess. A test asserts `DummyVecEnv(n=4)` ≡ `n=1` for a scripted policy. If this
  cannot be made to hold, that is the CLAUDE.md stop-and-report condition "OptionEnv cannot be vectorized".
- **Decision cap [A: F2]:** `max_decisions` child calls per episode. Mandatory, not a guard — see §4.3.
- Children are frozen, so the SMDP's dynamics are stationary while the parent trains.
- **Update size (amended 2026-10-02, H-6.8; restated H-7 A.7):** composites use `n_steps = 64`, because
  one composite step is a whole child execution and 2,048 of them per update left H-6's composites with
  one to ten updates in total. Every composite, baseline option sets included, uses `eval_every = 512`
  (one update at 8 envs), the floor §5.3 requires. H-6's composite held-out scores stand; their
  original `decisions_to_stable` are one update. Reasoning in §5.3.

### 5.3 Stabilization criterion (replaces mean > 0.85, std < 0.05) 🔒

> Every `eval_every` training steps, run `N_eval = 50` evaluation episodes in execution mode (§5.4) on
> the node's training env, **from a fixed seed set that is identical at every evaluation**.
> **Success rate** = fraction of episodes in which `goal` is reached before `timeout`.
> A node is **STABLE** when, in **two consecutive evaluations**, the success rate is **≥ 0.90** *and* its
> **Wilson 95 % lower bound is ≥ 0.80**.
> The two evaluations must be **at least one policy update apart**: `eval_every ≥ n_steps × n_envs`.
> Mean episode length is reported alongside but is not part of the criterion.

**Why "consecutive" needs a floor (amended 2026-10-02, H-6.8).** "Two consecutive evaluations" was
read as two evaluations, and with `eval_every` smaller than one PPO rollout it is two evaluations *of
the same network*. On a fixed seed set in deterministic execution mode the second is then a bit-exact
copy of the first, so the rule degenerates into "one evaluation, counted twice" — it removes the
binomial-noise protection the Wilson bound was added for and replaces a convergence test with a single
measurement. H-6 ran composites at `n_steps = 256 × 8 envs = 2,048` decisions per update with
`eval_every = 1,000`, and every history in `runs/v2/` came out in identical pairs
(`solve_doorkey` seed 42: 20, 20, 50, 50). That node was declared STABLE after **one** gradient update,
on one policy scored twice. The amendment makes the floor explicit and checkable, and
`EvaluationCallback` enforces it rather than trusting the caller.

**Why the extra bound [A: F11].** At p = 0.90 and N = 50 the estimate has σ ≈ 0.042. "≥ 0.90 twice in a
row" with freshly sampled eval seeds is selection on noise: it stops on a lucky pair. Fixing the seed
set removes the resampling, and the Wilson bound makes the stopping rule robust to the remaining
binomial noise. At N = 50 the bound binds at **46/50** (LB 0.812); 45/50 gives 0.786. A stricter
LB ≥ 0.85 would silently demand 48/50, which is not what "0.90" was meant to say. Fallback if 50
episodes still prove noisy: N = 100 with LB ≥ 0.85 (binds at 92/100, LB 0.850).

**A second seed set, used once [A: F11].** The fixed set above decides *when to stop*, so a score on it
is optimistically biased. At freeze time the node is evaluated once more on a seed set never used for
stopping, and both numbers go in `meta.json`. Only the second is quoted as the node's success rate.

Why the old rule had to go: MiniGrid's reward depends on path length. In environments with random start and goal positions (FourRooms), a policy that succeeds 100 % of the time typically still has reward std above 0.05, because path lengths differ from episode to episode, so the old rule can be impossible to satisfy regardless of learning quality. The PN-4 FourRooms run also used a relaxed 0.75 threshold in `train_pn4.py` while docs said 0.85; v2 retires both numbers.

The thresholds (0.90, two evaluations, N = 50) are defaults, fixed per experiment and reported 🧪. The same rule applies to primitive and composite nodes.

**Composites train at `n_steps = 64` (amended 2026-10-02, H-6.8).** A composite's unit of experience is
an SMDP decision, and a whole episode is only a handful of them — `{2,1} solve_doorkey` averages 2
decisions per episode — so `n_steps = 256 × 8 envs` buys one gradient update per 2,048 decisions and a
20,000-decision budget is ten updates in total. At `n_steps = 64` one update costs 512 decisions, which
makes a comparable budget ~100 updates and `eval_every = 512` the smallest value §5.3's floor permits.
Primitives keep `n_steps = 256`: their unit is an env step and the same budget already buys hundreds of
updates. The numbers are experiment parameters and go in each run's `meta.json` as usual.

### 5.4 Freezing and execution mode 🔒

- STABLE → FROZEN: `requires_grad_(False)` on all policy parameters. The forward pass keeps running (needed by laterals and by parents).
- A frozen node is never modified, for any reason.
- **Execution mode:** frozen nodes act **deterministically** by default. Stabilization is evaluated in the same mode. (Fixes v1's `deterministic=not self.frozen`, which made frozen columns sample stochastically — done in H-0.) Deterministic MiniGrid policies can loop under partial observability; `timeout` bounds the damage, and stochastic execution is a 🧪 alternative.
- **The mode is a constructor argument, not a constant [A: F17].** `Executor(execution_mode=...)` takes
  `deterministic` (default), `stochastic` or `epsilon`, from the first version rather than as a later
  retrofit. Under goal-only masking (§4.3) a looping child costs its parent a full child timeout per
  re-selection, so the executor logs per-child timeout and loop rates and the 🧪 comparison comes free.

---

## 6. Task recognition (AE + GRU)

AE and GRU live in the **lifecycle layer**. They never select children inside the skill graph. 🔒

### 6.1 Autoencoder: "is this situation new?"

- Output: reconstruction error → novelty flag. It is one of the two growth signals (§8); it is **never merged** with the reward signal 🔒.
- Input is **normalized per channel** (MiniGrid image channels are object index 0–10, color 0–5, state 0–2) before the MLP AE (147 → 64 → 32 → 64 → 147). In v1 raw integers went in, which made the fixed 0.05 threshold meaningless.
- **One shared utility [A: F24].** The AE and the policy pipeline (§5.1) use the *same* normalization
  function. The flattened vector is C-order over (7, 7, 3), so channels are the fastest-varying axis:
  `obs.reshape(7, 7, 3) / [10, 5, 2]` is correct and several near-miss variants half-work silently. A
  unit test asserts the per-channel maxima.
- Threshold: calibrated per known task on familiar data (mean + k·σ), then frozen during inference.
- 🧪 Single global AE vs **one AE per task** (as in the base paper, Erden et al. 2025). A single AE fine-tuned on new tasks forgets old ones, which makes old tasks look novel; per-task AEs do not.

### 6.2 GRU: "which known task is this?"

- Classifies the last N observations into a known task id; `-1` when confidence < threshold.
- **Root selection:** the lifecycle manager keeps a registry `task_id → root SkillNode`. The GRU's task id chooses which root the executor runs. (In v1 the task id was computed and then ignored.)
- 🧪 N ∈ {5, 20, 50}: accuracy vs switching latency.

### 6.3 Where continual-learning forgetting actually happens

**C-3 is a regression test, not an experiment [A: F26].** Because retention of skills is guaranteed by
freezing, re-evaluating frozen nodes after each new task guards against bugs (a mutated checkpoint, a
changed predicate) rather than evidencing continual learning. The continual-learning claim rests on
the recognition layer (C3) and on forward transfer (C-2), and §11 says so.

Skill nodes cannot forget: they are frozen. The **recognition layer can**: a GRU head retrained on each new task, or a single shared AE, may degrade on earlier tasks. Backward-transfer measurements are therefore meaningful for task-identification accuracy and novelty detection, and are reported there (§11, R-2). GRU retraining protocol is ⚠️ Q7.

---

## 7. Progressive Networks: forward transfer, not forgetting 🔒

- **Forgetting of skills is prevented by freezing**, independent of PN. A library of frozen PPO policies (the base paper's design) already has zero skill forgetting.
- **PN's contribution is forward transfer:** a new primitive node may read hidden activations of earlier frozen primitives through lateral connections (`nn.Linear(64, 64)` added to its own hidden layer, as implemented today). The claim is measured as steps-to-stabilization with vs without laterals (C-2).
- **Lateral sources** are chosen when a primitive node is created, from FROZEN primitive nodes. Default: the most recently created primitive (equivalent to v1's `{0,m-1} → {0,m}`). 🧪 Alternatives: all earlier primitives; the primitive whose AE latent is closest.
- **Arity [A: F20].** `lateral_sources` is plural in §2, but `LateralMlpExtractor` implements exactly one
  source and concatenation is forbidden by the hard rules. Until C-2, `len(lateral_sources) ≤ 1` and the
  constructor asserts it. If several sources are later adopted they are **summed**
  (`h += Σᵢ latᵢ(srcᵢ)`), never concatenated.
- **Cost [A: F22].** A lateral re-runs its frozen source on the same features, recursively through all
  ancestors: O(chain length) forward passes per env step, per env. The chain length is capped by config
  (default 1) and the cost is reported in C-2's wall-clock numbers.
- **C-2's source must match the pipeline [A: F23].** The legacy `runs/pn4/col01` was trained on Empty-8x8
  with raw-integer observations and is incompatible with the normalization of §5.1. C-2 retrains its own
  lateral source under the current pipeline rather than reusing it. Not on the critical path.
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

**Known caveat:** with "all frozen nodes" as children, a new composite's structural level is `1 + max level in the library` even if it only uses low-level children. Phase B therefore also reports an **effective level** computed from the children actually selected during evaluation. In Phase A, children are chosen deliberately, so structural and effective levels coincide. The executor emits the selection trace this needs from H-1 onward **[A: F28]**, so G-2 and UI-1 get it for free.

**Budgets are not comparable across candidates [A: F27].** "All currently FROZEN nodes" means the
candidate's action space grows with every task learned, so a fixed step budget `B_c` buys less
exploration per action later in the curriculum. G-2 either fixes the candidate child-set size or
reports budget per action; it does not compare raw `B_c` across tasks.

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
| `StabilizationMonitor` (mean / std of reward) | `EvaluationMonitor` (success rate, §5.3) | Replace. **Do not delete the module while `ColumnTrainer` lives** — `pn_trainer.py` imports it and `src/continual_learning/__init__.py` re-exports it; H-3 itself needs `ColumnTrainer.load_policy_into` to re-evaluate `col01`. Mark deprecated in H-3, delete with `ColumnTrainer`. |
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

Every composite reports **three** env-step numbers, so the two honest framings of its cost cannot drift apart (amended 2026-10-02, H-6 Part A.4):

- `env_steps_own` — env steps consumed while training this node, including steps executed inside its frozen children
- `env_steps_children` — the sum of `env_steps_own` over the transitive closure of its children
- `env_steps_total` — the two added

`env_steps_own` is the cost of acquiring *this* task once a skill library exists; `env_steps_total` is the honest total for a learner starting from nothing. Both are reported beside every flat baseline and neither is presented as "the" number.

**Reported metrics are success rate and episode length [A: F12].** Reward is a training signal whose
scale depends on each node's timeout, so it is never compared across nodes or against a flat baseline.
Every success rate quoted is the freeze-time score on the held-out seed set of §5.3.

| Claim | Experiment | Metric | Baselines |
|---|---|---|---|
| **C1 Composition helps.** A composite over frozen skills learns a task faster than learning it flat, and on a task with a real choice it learns something no mask encodes. | `{2,1} solve_doorkey` on DoorKey; **and `{1,2} unlock_matching` on the two-key DoorKey variant** | success rate vs env steps; steps to STABLE; first-decision choice accuracy conditioned on door colour | flat PPO on DoorKey (primary); flat composite over the three primitives; **plus two mandatory controls [A: F1]:** (a) uniform random choice over the available children, (b) the same run with initiation masking switched on |
| **C2 Reuse across tasks and depth (unbounded).** A skill learned once is reused by later tasks, and deeper composition reuses intermediate skills. | UnlockPickup and BlockedUnlockPickup reusing `unlock_door`; BlockedUnlockPickup variant A (level 2) vs variant B (level 3) | R1–R4 below; stack depth reached | composite from primitives only; flat PPO on each task |
| **C3 Task recognition.** The AE separates known from novel tasks; the GRU identifies known tasks. | observations from all curriculum envs | AE ROC-AUC per task pair; GRU accuracy and latency vs N; recognition accuracy on old tasks after each new task | single vs per-task AE |
| **C4 Continual learning.** Skills are retained; laterals give forward transfer. | re-evaluate every FROZEN node after each new task (**a regression test, not evidence — [A: F26]**); primitives with vs without laterals | retention (unchanged by construction; a change means a bug); steps to STABLE | no laterals; (optional) EWC fine-tuning |
| **C5 Introspection.** Internal state is inspectable. | GUI on real runs | qualitative: graph, active stack, traces, per-node stats | — |

**What C2 is evidenced by (amended 2026-10-02, H-7 A.5).** `env_steps_own < env_steps_total` holds by
construction; C2 is evidenced by transfer without retraining (R1), marginal cost against a flat
baseline (R2), and the cumulative cost over the curriculum (R3), not by that inequality.

| | Question | Measured by |
|---|---|---|
| **R1 Transfer** | Does a frozen skill work, untouched, inside a task it was never trained in? | F29 rows: `{1,1} unlock_door` inside UnlockPickup, and inside BlockedUnlockPickup after the blocker is moved |
| **R2 Marginal cost** | Given the library, is a new task cheaper than learning it flat? | `env_steps_own` of the nodes created for the task (a primitive's `steps_trained`) vs flat PPO's steps to STABLE on that task |
| **R3 Amortisation** | Over the whole curriculum, does the hierarchy's cumulative cost fall below flat PPO's, and at which task? | the cumulative table over Unlock → DoorKey → UnlockPickup → BlockedUnlockPickup |
| **R4 Depth reuse** | Does reusing an intermediate composite help? | T9b (level 3, reuses `unlock_pickup`) vs T9a (level 2, same skills, no intermediate node); a depth-4 stack |

**Two fields added to every composite's report (amended 2026-10-02, H-7 A.6).**
- `chain_ceiling`: the product of the children's success rates along the successful path, taken from
  their F29 handoff rows. A composite cannot exceed it; reporting it separates "the selector chose
  wrongly" from "a child failed".
- `weak_heldout: true` in `meta.json` for any node whose held-out Wilson lower bound is below 0.80. It
  does not block freezing (§5.3 is unchanged). It is a flag for the reader.

### Candidate curriculum (⚠️ Q6)

**Env IDs verified against the installed `minigrid==3.1.0` (2026-10-01).** All four exist, with
`max_steps` and grid size as follows — these are the budgets the timeout invariant of §5.2 is checked
against: `MiniGrid-Unlock-v0` 11×6 / 288 · `MiniGrid-DoorKey-6x6-v0` 6×6 / 360 ·
`MiniGrid-UnlockPickup-v0` 11×6 / 288 · `MiniGrid-BlockedUnlockPickup-v0` 11×6 / 576.
Two facts that shape the predicates: the `RoomGrid` envs expose `env.unwrapped.door` and `.obj` while
`DoorKey` does not (its builder scans the grid, §4.2), and MiniGrid *does* encode the carried object in
the observation (`gen_obs_grid` writes `self.carrying` into the agent's cell), so "am I holding the
key" is visible to a policy.

**Primitive generality is on the critical path, not optional [A: F29].** A primitive trained on
`SubgoalEnv` only ever sees states its reset hook produces, but a parent hands it whatever state it
reached. Unlock is an 11×6 two-room layout and DoorKey-6x6 is 6×6, so `get_key` trained only on Unlock
is unlikely to work inside `{2,1} solve_doorkey`. T1/T2 are therefore trained on **both** layouts with
randomized agent start pose, and H-4's exit gate evaluates each primitive from parent-reachable states,
not only from hook states.

| # | Node | Kind | Children | Env | Goal predicate |
|---|---|---|---|---|---|
| T1 | `{0,1} get_key` | primitive | — | Unlock (+ DoorKey layouts) | agent carries the key |
| T2 | `{0,2} open_door` | primitive (I: carrying key) | — | Unlock, reset with key in hand | locked door is open |
| T3 | `{1,1} unlock_door` | composite | T1, T2 | `MiniGrid-Unlock-v0` | door open (env success) |
| T4 | `{0,3} go_to_goal` | primitive | — | DoorKey, reset with door open | agent on goal tile |
| T5 | `{2,1} solve_doorkey` | composite | T3, T4 | `MiniGrid-DoorKey-6x6-v0` | env success |
| T6 | `{0,4} pickup_box` | primitive (I: door open and carrying nothing) | — | UnlockPickup, reset with the door open, the hand empty and no key on the door's approach cell | agent carries the box |
| T6b | `{0,9} drop_held` | primitive (I: carrying anything) | — | UnlockPickup, reset with the door open and the key in hand | agent carries nothing |
| T7 | `{2,2} unlock_pickup` | composite (I: door locked) | T3, T6b, T6 | `MiniGrid-UnlockPickup-v0` | agent carries the box (env success) |
| T8 | `{0,5} move_blocker` | primitive (I: a ball is adjacent to the door) | — | BlockedUnlockPickup | **no ball adjacent to the door and the agent carries nothing** |
| T9a | `{2,3} blocked_pickup_a` | composite (I: door locked and a ball adjacent to it) | T8, T3, T6 | `MiniGrid-BlockedUnlockPickup-v0` | agent carries the box (env success) — **deferred to H-7b** |
| T9b | `{3,1} blocked_pickup_b` | composite (I: as T9a) | T8, T7 | same | agent carries the box (env success) — **deferred to H-7b** |

**H-7 node specifications (amended 2026-10-02, H-7 A.1–A.3; T6, T6b and T7 re-specified by H-7
amendment 1).** Rationale in `plans/2026-10-02_H7_plan.md` and `plans/2026-10-02_H7_amendment_1.md`.

| # | timeout | `max_decisions` | reset hook |
|---|---|---|---|
| T6 `{0,4} pickup_box` | 120 | — | `open_door`, `key_off_door`, `randomize_agent_pose(any)` |
| T6b `{0,9} drop_held` | 30 | — | `open_door`, `give_key`, `randomize_agent_pose(any)` |
| T7 `{2,2} unlock_pickup` | 288 | 8 | none |
| T8 `{0,5} move_blocker` | 120 | — | `randomize_carrying(0.5, key)`, `randomize_agent_pose(any)` — H-7b |
| T9a `{2,3} blocked_pickup_a` | 576 | 12 | none — H-7b |
| T9b `{3,1} blocked_pickup_b` | 576 | 8 | none — H-7b |

- **T6 split into T6b + T6 (H-7 amendment 1).** MiniGrid `pickup` needs an empty hand and
  `Door.toggle` does not consume the key, so at the `unlock_door` handoff the agent is holding the
  key. A.1 answered this by training one `pickup_box` on UnlockPickup and BlockedUnlockPickup with the
  key in hand with probability 0.5. That node was **not stable within 600,000 env steps on any of
  three seeds** (held-out 0.28 / 0.40 / 0.24), and the seed-42 checkpoint scored **0.10 at the real
  handoff** (`runs/v2/pickup_box_diagnosis.json`): one flat policy under one sparse reward was asked
  to cross the door, drop the key and pick up the box. A.1 is withdrawn on that measurement. The skill
  is decomposed instead: `{0,9} drop_held` empties the hand, and `{0,4} pickup_box` is trained
  empty-handed on UnlockPickup only, and the parent T7 sequences them. Under the goal-only mask
  `drop_held` is unavailable while the hand is empty (its goal holds) and `pickup_box` stays
  selectable with a full hand, where it times out; T7 has to learn unlock → drop → pick up from that.
- **T6b `{0,9} drop_held`.** Goal `carrying:none`, initiation `carrying:any`, timeout 30, trained on
  UnlockPickup from the door-open, key-in-hand state at any reachable pose.
- **One re-specification only.** If T6b or the new T6 does not stabilise within its budget
  (100,000 and 1,000,000 env steps), that is the finding and H-7a stops. The first `pickup_box`
  attempt's cost (3 × 600,064 env steps) is reported, not counted (caveat 4 of the amortisation
  table).
- **T8, goal (A.2).** "The cell in front of the door is free" is read from the grid, and a carried
  object is not on the grid, so it becomes true the moment the ball is picked up, while the agent
  still holds it. `get_key` would then be asked to act with a full hand. The goal requires the ball to
  be put down away from the door and the hand empty. `randomize_carrying` is in T8's hook because under
  goal-only masking a parent may pick `unlock_door` first; `get_key` succeeds, `open_door` cannot, and
  control returns with the key in hand.
- **T7's timeout is constrained (A.3).** By F13 its window is 240 (T3's timeout) to 288 (the env's
  `max_steps`), so one fully wasted `unlock_door` cannot be recovered inside UnlockPickup.
- Two library functions enter `spec_hash` through their spec strings: the predicate
  `door_approach_blocked` (`"door:approach_blocked"`: a ball orthogonally adjacent to the door; false
  when there is no door) and the hook `relocate_blocker` (`"relocate_blocker"`: moves the ball to a
  reachable free cell not adjacent to the door, drawn with the env's own RNG; a no-op when there is no
  ball; raises `UnsuitableLayout` when no such cell exists). `relocate_blocker` reproduces the state T8
  leaves, which deleting the ball would not.
- Two more (H-7 amendment 1): the predicate `carrying_anything` (`"carrying:any"`: the agent holds
  some object) and the hook `key_off_door` (`"key_off_door"`: raises `UnsuitableLayout` when a key
  lies on a cell orthogonally adjacent to the door). With the door open and the hand empty, a key on
  the door's approach cell can only be passed by picking it up, which fills the hand again; at the
  real handoff it cannot happen, because the agent stands on that cell when it drops the key.

**H-7 baselines (amended 2026-10-02, H-7 A.4).** Baseline only, registered under `runs/v2/baselines/`,
never usable as a child. Each matches the node it is compared with in goal, timeout, pipeline, PPO
config, envs and seed sets.

| node | env | goal | timeout | budget | `eval_every` |
|---|---|---|---|---|---|
| `{0,B} flat_unlock_ppo` | `MiniGrid-Unlock-v0` | door open | 240 | 1,500,000 env steps | 50,000 |
| `{0,B} flat_unlockpickup_ppo` | `MiniGrid-UnlockPickup-v0` | agent carries the box | 288 | 3,000,000 | 100,000 |
| `{0,B} flat_blocked_ppo` | `MiniGrid-BlockedUnlockPickup-v0` | agent carries the box | 576 | 3,000,000 | 100,000 |
| `{1,B} flat_unlockpickup_options` | `MiniGrid-UnlockPickup-v0`, children T1, T2, T6b, T6 (amendment 1) | agent carries the box | 288, `max_decisions` 12 | 40,960 decisions | 512 |

Budgets are fixed in advance and never extended; a run not stable at its budget is reported as
`> budget (not stable)` with the best success rate it reached.

`initiation` is stated for composites too, not only primitives (amended 2026-10-02, H-6 Part A.3): T3 `{1,1} unlock_door` and T5 `{2,1} solve_doorkey` carry `I: door locked`. §2 requires it of every node, and H-5 measured what leaving it at the `ALWAYS` default costs — the strict mask then differs from the goal-only mask in 0 of 304 decisions and the C1 masking ablation is vacuous. Declaring it changes T3's `spec_hash`, so its H-5 checkpoint is correctly refused by the registry (F18) and T3 is retrained in H-6.2.

**H-6 additions (2026-10-02).** Three experiment nodes and two baseline nodes, kept below the curriculum so T1–T9b stays readable. T10a/T10b/T11 are the genuine-choice task required by the C1 row; the two `{·,B}` nodes are baselines only, are registered under `runs/v2/baselines/`, and may never be used as a child of a curriculum node.

| # | Node | Kind | Children | Env | Goal predicate |
|---|---|---|---|---|---|
| T10a | `{0,6} get_key_yellow` | primitive (I: carrying nothing) | — | DoorKey-6x6 + two-key hook | agent carries a **yellow** key |
| T10b | `{0,7} get_key_blue` | primitive (I: carrying nothing) | — | DoorKey-6x6 + two-key hook | agent carries a **blue** key |
| T11 | `{1,2} unlock_matching` | composite (I: door locked, carrying nothing) | T10a, T10b, **T12** | DoorKey-6x6 + two-key hook | door open |
| T12 | `{0,8} open_door_tk` | primitive (I: carrying key) | — | DoorKey-6x6 + two-key hook + a key in hand | door open |
| — | `{0,B} flat_doorkey_ppo` | primitive, **baseline only** | — | `MiniGrid-DoorKey-6x6-v0` | agent on goal tile |
| — | `{1,B} flat_doorkey_options` | composite, **baseline only** | T1, T2, T4 | `MiniGrid-DoorKey-6x6-v0` | agent on goal tile |

**T12 exists because T2 does not transfer here (amended 2026-10-02, H-6.8).** `{0,2} open_door` scores
1.00 on its own hook distribution and **0.83** in the two-key env from the state the key primitives
actually leave — a key in hand, a *second* key still on the floor, and a blue door half the time. It
had never seen a spare key lying around. Per F29 and the H-6 plan's A4 branch the response is a new
node trained on the parent's own distribution, never a modification of T2: `{0,8} open_door_tk` scores
**0.98** from the same handoff. T2 stays frozen and untouched, and the pair is reported as the transfer
measurement it is.

**A reset hook's spec string must change whenever its behaviour does (amended 2026-10-02, H-6.8).**
`spec_hash` covers a hook's *name*, not its implementation [A: F18], so tightening a hook's rules
leaves every existing checkpoint loadable while the distribution it was trained on has silently
changed. Nothing detects this automatically. The convention is a suffix naming the rule, as in
`place_two_keys:yellow|blue:door_clear`, and it is the author's responsibility on every hook edit.

**A hook may reject a layout, and the env resamples (amended 2026-10-02, H-6.8).** A hook defines a
node's initial-condition distribution and some randomly generated layouts cannot satisfy it. Raising
`UnsuitableLayout` makes `SubgoalEnv.reset` and `ResetHookEnv.reset` draw the next layout, up to
`max_reset_tries`; the retry stream is seeded by the episode's seed, so episodes stay reproducible.
This is distinct from a `RuntimeError`, which means the hook itself is wrong. It is needed because
DoorKey-6x6 draws its splitting wall at x = 2 or x = 3, so the left room is sometimes a single column
of four cells in which two keys cannot both be independently reachable.

**Episodes no policy can solve must be excluded before any plateau is interpreted
(amended 2026-10-02, H-6.8).** Two bugs in the two-key hook produced unsolvable episodes and both were
first misread as learning limits, at a cost of roughly 1.3 M env steps. MiniGrid `toggle` acts on
`front_pos` and DoorKey's door sits in a vertical wall, so exactly one cell on the agent's side can
open it: a key *on* that cell made 59 of 400 layouts (14.7 %) unopenable, and a key *between the agent
and* that cell made 3 of the 50 stopping seeds and 4 of the 50 held-out seeds unopenable — which is
exactly the gap between `{0,8} open_door_tk`'s 0.94 stopping score and the 0.84 it generalized to.
Before concluding that a node has hit a ceiling, audit both fixed seed sets for solvability.

The two-key hook recolours the DoorKey door and its key to one of two colours drawn per episode, adds a decoy key of the other colour, randomizes the agent's position, and faces the agent so the door is inside its 7×7 view. It is built from reset hooks only — no new env class — so it inherits `spec_hash` coverage and seed reproducibility. `Door.toggle` opens a locked door only for a key of the matching colour, and MiniGrid `pickup` needs an empty hand, so a wrong first pick is unrecoverable within the episode: the choice separates success rate, not merely episode length. Facing is pinned because the door is inside the agent's view in only 163/300 resets with a random facing and 300/300 facing the door (measured on minigrid 3.1.0) — a feedforward policy cannot condition on a signal it cannot see, and without this the experiment would measure partial observability instead of composition.

T5 already exercises a mixed-level DAG and a depth-3 stack (`solve_doorkey → unlock_door → get_key`). T9a vs T9b is the depth-reuse comparison. 🧪 Primitive generality: train primitives on a mix of the layouts where their goal predicate is defined, so they work inside every parent task.

**Why C1 needs the choice task (amended 2026-10-02, H-6 Part A.2).** On Unlock and DoorKey the subgoal order is strict, so the initiation mask *is* the task's ordering constraint: H-5 measured a random policy under that mask attaining `{1,1} unlock_door`'s score exactly (0.98, 19.2 steps, byte-identical to the learned composite). Success rate on a strict-order task therefore cannot separate composition from masking, however deep the hierarchy. What the learned composite demonstrably adds there is that it *discovers* the ordering under the goal-only mask — measured as the divergence count, 146 of 146 decisions where the strict mask would have withheld a child and the policy had already declined it — and executes 4.4× more efficiently than random choice. `{1,2} unlock_matching` is the complement: two children stay valid under the strict mask and the correct one is fixed by the door's colour in the observation, so no state-independent mask can substitute and a masked random policy is capped near 0.50 by construction.

FourRooms is no longer part of the curriculum. It survives only as a diagnostic (C-1).

---

## 12. Roadmap: four tracks

Story points as in `ENS491_Project_State.md`. Status lives there; this table defines the work.

**Track H: Hierarchy (critical path)**

| # | Task | SP |
|---|---|---|
| H-0 | Fix `Column.forward` deterministic flag; document the 0.75 / 0.85 mismatch in the run log | 1 |
| H-0.5 | Apply the Part A amendments to this document, `CLAUDE.md` and `SUPERVISOR_QUESTIONS.md`; pin dependencies and add `sb3-contrib` | 2 |
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
| R-3 | Task registry: GRU task id → root node in the lifecycle manager (**after H-1** — it needs the node registry **[A: F25]**) | 3 |

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
Critical path:  H-0 → H-0.5 → H-1 → H-2 → H-3 → H-4 → H-5 → H-6 → H-7
Parallel:       C-1, C-2, C-3   ·   R-1, R-2   ·   UI-1 and R-3 (both after H-1)
After H-6:      G-1 → G-2 → G-3
```

**Compute [A: F31].** The development machine has no discrete GPU (Intel Arc integrated only; the
RTX 3070 Ti of the v1 runs belongs to a different machine), and a MacBook Pro (M5 Pro) is a second
target. Device resolution is `cuda → mps → cpu` with an explicit override, but the **default is `cpu`**:
a 147→64→64 MLP is too small to benefit from a GPU, and SB3's own guidance is to use CPU for
`MlpPolicy`. `requirements.txt` therefore carries no CUDA-pinned wheels, so it installs on macOS
arm64. Every training script measures and prints steps/s before a budget is committed.

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

### Amendments applied 2026-10-01 (H-0.5)

Each was found by reading this document against the code and the installed MiniGrid before any v2 code
was written; the full rationale is in `plans/2026-10-01_v2_H0-H5_plan.md` Part A.

| v2 as locked | v2 as amended | Reason |
|---|---|---|
| `available = initiation(s) and not goal(s) and …` | `available = not goal(s) and …`; `initiation` is metadata and reset-hook input, not a mask | With initiation masking the curriculum's composites have one legal child per decision and learn nothing (F1) |
| Composite reward = discounted env reward | Every node's reward comes from its **own** goal, same shape at every level | A composite whose goal is not the env's success condition got identically zero (F10) |
| `max_decisions` "as a guard" | Mandatory and recorded | Goal-only masking makes re-selection of a timed-out child possible forever (F2) |
| SR ≥ 0.90 twice | SR ≥ 0.90 **and** Wilson LB ≥ 0.80 twice, on a fixed seed set; held-out set at freeze | σ ≈ 0.042 at N=50 — the old rule stops on a lucky pair (F11) |
| Policy sees raw uint8 obs | `ChannelNormalize` on by default, shared with the AE | SB3 does not normalize a non-image Box; v1 fed 0–10 integers to the MLP (F16) |
| `state` = `env.unwrapped` | immutable `StateView` snapshot, one per env step | Predicates, masks and obs must share an instant; traces must be serializable (F5) |
| recursive `available()` | memoized per `(node, state version)` | Exponential on a DAG with shared children (F4) |
| `Executor.act` recursive | bounded loop with an iteration cap | A hung run costs more to diagnose than a raised exception (F6) |
| no timeout invariant | `timeout ≥ max(child.timeout)`, root `timeout ≤ env.max_steps` | Otherwise a parent cannot succeed, and it looks like a learning failure (F13) |
| no node identity | `spec_hash` over env, predicates, timeout, pipeline, children | A frozen checkpoint must know what it was trained under (F18) |
| — | acyclicity asserted at construction; empty mask impossible; `info["k"]` always logged; `DummyVecEnv` only; single lateral source | F21, F3, F14, F15, F20 |

**Unchanged and still locked:** AE and reward signals are never merged; frozen means no gradients but a running forward pass; only stabilized nodes are reused; Python 3.10, PyTorch, MiniGrid, Gymnasium, SB3, W&B, Streamlit; never `CnnPolicy`; nothing hardcodes depth or width.

### Amendments applied 2026-10-02 (H-6)

Each follows from a measurement taken after H-5, not from a preference. Full rationale in
`plans/2026-10-02_H6_plan.md` Part A and Part C.

| v2 as it stood | v2 as amended | Reason |
|---|---|---|
| §11 states `initiation` for primitives only | It is stated for composites too: T3 and T5 carry `I: door locked`, T11 `I: door locked and carrying nothing` | §2 requires it of every node, and the `ALWAYS` default made H-5's masking ablation differ from the goal-only mask in 0 of 304 decisions |
| C1 = `solve_doorkey` vs flat baselines | C1 also runs `{1,2} unlock_matching`, a task with a genuine choice | On a strict-order task the initiation mask *is* the ordering constraint, so success rate cannot separate composition from masking — measured, not conjectured |
| "env-step counts include steps inside frozen children" | Three named fields: `env_steps_own`, `env_steps_children`, `env_steps_total`, all three reported | The amortized and from-scratch framings of a composite's cost are both legitimate and must not be allowed to drift |
| §11 node table ends at T9b | Adds T10a, T10b, T11 and two explicitly non-curriculum baseline nodes | The choice task needs colour-specific key primitives; baselines need node ids but must never be reusable children |

### Amendments applied 2026-10-02 (H-6.8)

Each comes from an advisor review of the H-6 run against `runs/v2/`. Rationale in
`plans/2026-10-02_H6_summary.md` § "Corrections (H-6.8)".

| v2 as it stood | v2 as amended | Reason |
|---|---|---|
| "two consecutive evaluations" | plus `eval_every ≥ n_steps × n_envs`, enforced by `EvaluationCallback` | With a smaller `eval_every` the two evaluations are the same network on the same seeds, so the rule became "one evaluation counted twice". Every H-6 composite history is identical pairs, and `solve_doorkey` was declared STABLE after one gradient update |
| one `n_steps` for every node | composites use `n_steps = 64`, primitives keep 256 | A composite's unit is an SMDP decision and an episode is ~2 of them; 256 × 8 bought 10 updates in a 20,000-decision budget |
| §11 lists T11's children as T10a, T10b, T2 | T10a, T10b, **T12** `{0,8} open_door_tk`, with T12 added to the table | T2 scores 0.83 from the two-key handoff, below the 0.90 gate, so the A4 branch applied |
| `spec_hash` silently covers a hook by name | a hook's spec string must change whenever its behaviour does | Tightening `place_two_keys` left every checkpoint loadable while its distribution had changed |
| a hook either works or raises | a hook may raise `UnsuitableLayout` and the env resamples | Some randomly generated layouts cannot host a node's initial conditions |
| — | audit both fixed seed sets for solvability before reading a plateau as a learning limit | Two hook bugs produced unsolvable episodes and cost ~1.3 M env steps misdiagnosed |

### Amendments applied 2026-10-02 (H-7)

Each follows from a fact measured on minigrid 3.1.0 before H-7's code was written. Rationale in
`plans/2026-10-02_H7_plan.md` Part A; re-measured by `scripts/eval_h7_preflight.py`.

| v2 as it stood | v2 as amended | Reason |
|---|---|---|
| T6 resets with the door open | door open **and the key in hand with probability 0.5**, on UnlockPickup and BlockedUnlockPickup | `pickup` needs an empty hand and the key is still held at the `unlock_door` handoff (A.1) |
| T8 goal: cell in front of the door is free | no ball adjacent to the door **and the agent carries nothing**; I: a ball adjacent to the door | the grid-based predicate is true while the ball is still in the agent's hand (A.2) |
| T7, T9a, T9b unnamed, no initiation or timeout | named, with initiation, timeout and `max_decisions`; T7's F13 window 240–288 recorded | A.3 |
| — | four H-7 baseline rows, baseline only | A.4 |
| C2 metric "same as C1" | R1–R4, and C2 is not evidenced by `env_steps_own < env_steps_total` | that inequality holds by construction (A.5) |
| composite report: three env-step fields | plus `chain_ceiling`; `weak_heldout` flag in `meta.json` | separates selector failure from child failure; flags a weak held-out score without changing §5.3 (A.6) |
| §5.3 floor and `n_steps = 64` stated in §5.3 | restated in §5.2 with `eval_every = 512` for every composite | A.7; already made by H-6.8 |

### Amendments applied 2026-10-02 (H-7 amendment 1)

H-7 stopped on its assumption A2: the A.1 `pickup_box` was not stable within 600,000 env steps on any
seed. Rationale in `plans/2026-10-02_H7_amendment_1.md`; the stop report is in
`plans/2026-10-02_H7_summary.md`.

| v2 as it stood | v2 as amended | Reason |
|---|---|---|
| T6 (A.1): door open, key in hand with probability 0.5, UnlockPickup and BlockedUnlockPickup; I: door open | T6 empty-handed on UnlockPickup only, hook `open_door`, `key_off_door`, `randomize_agent_pose(any)`; I: door open and carrying nothing; new spec, new `spec_hash`. A.1 is withdrawn | not stable in 600,000 steps on 3 seeds (held-out 0.28 / 0.40 / 0.24); 0.10 at the real handoff. One flat policy was asked to cross, drop and pick up under one sparse reward |
| — | T6b `{0,9} drop_held`: goal carrying nothing, I: carrying anything, timeout 30, UnlockPickup, hook `open_door`, `give_key`, `randomize_agent_pose(any)` | the drop is a one-action skill and gets its own node; a skill too hard for one primitive is decomposed |
| T7 children T3, T6 | T3, T6b, T6 | the parent sequences unlock → drop → pick up |
| `{1,B} flat_unlockpickup_options` children T1, T2, T6 | T1, T2, T6b, T6 | same skills as T7 without the intermediate T3 |
| T8, T9a, T9b in H-7 | deferred to H-7b, planned after H-7a's result | the H-7.1 pre-flight already shows the frozen Unlock skills break on BlockedUnlockPickup (`get_key` picks up the ball in 29/50) |
| — | predicate `carrying_anything` (`"carrying:any"`), hook `key_off_door` (`"key_off_door"`) | T6b's initiation; T6's hook rejects a key on the door's approach cell, which an empty-handed policy could pass only by refilling its hand |
