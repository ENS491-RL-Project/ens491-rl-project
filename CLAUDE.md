# CLAUDE.md

<!--
Maintainer notes (HTML comments are stripped before CLAUDE.md reaches Claude's context, so they cost nothing).

Size: keep this file well under 200 lines. It loads in full into every session; longer files lose
adherence. Topic detail lives in .claude/rules/*.md, which load only when Claude touches matching
paths (`paths:` frontmatter). Design detail lives in docs/ENS491_Target_Architecture_v2.md.
Do not duplicate a rule between this file and a rules file.

This file and .claude/rules/ are shared by the whole team, on different machines and OSes. Keep them
machine-neutral: no personal paths, OS-specific services or one person's tool setup. Personal notes go in
CLAUDE.local.md (gitignored, loaded after this file) and personal hooks in .claude/settings.json (gitignored).

graphify: graphify-out/ is gitignored. The graph is AST-only (no LLM cost) and rebuilds locally in ~2 s, so
committing it would only add 2 MB single-line diffs, merge conflicts and machine-local manifest timestamps.
`graphify hook install` keeps it fresh: its post-commit, post-merge (git pull), post-checkout and post-rewrite
hooks rebuild it; they skip until a first `graphify update .` has created graphify-out/.
Revisit committing it only if the LLM semantic pass over docs/ is adopted (that output costs tokens to rebuild).
- Extra freshness (Claude Code hooks, `graphify watch .`) is each developer's own local setup.
- graph.json is rewritten only when code topology changes, so an unchanged mtime is not a failed update.
- After substantial changes under docs/, `/graphify . --update` refreshes the semantic (LLM) pass over docs.
- The `## graphify` section at the end is written by `graphify claude install`; edit it only through that command.
-->

Hierarchical continual RL agent on **MiniGrid**. Every learned task becomes a frozen, reusable **skill node**; composite nodes select among earlier frozen nodes as options, to any depth. Freezing prevents forgetting; an autoencoder (novelty) and a GRU (task id) decide when a task is new and which known task is running. Curriculum: Unlock, DoorKey, UnlockPickup, BlockedUnlockPickup (v2 §11).

## Docs

- `docs/ENS491_Target_Architecture_v2.md` is **the reference design** and overrides every other doc. Change it first before changing the design in code.
- `docs/ENS491_Project_State.md`: status of every roadmap task (H-*, C-*, R-*, G-*) and the open empirical questions. Check it before starting roadmap work; update it when a task finishes.
- `docs/plans/`: per-task plans and summaries. The newest `*_session_handoff.md` says where work stopped.
- `docs/ENS491_Module_Interfaces.md`: only its AE and GRU sections still apply. `docs/ENS491_Recursive_Hierarchy_Design.md` is the superseded v1 design (history only).

## Code layout

- v2 (extend this): `src/skills/` (`SkillNode`, `Executor`, `NodeRegistry`, predicates, `StateView`), `src/envs/` (`apply_obs_pipeline`, `SubgoalEnv`, `OptionEnv`), `src/training/` (`node_trainer`, `evaluation` = stabilization via `StabilityCriterion` / `EvaluationCallback`, `device`). Experiments are `scripts/train_*.py`, `scripts/eval_*.py`; runs go to `runs/v2/`.
- Reused from v1: `ColumnPolicy` / `LateralMlpExtractor` (primitive network), `AutoencoderModule`, `GRUTaskIdentifier`, seeding/checkpoint code in `pn_trainer.py`.
- v1-only, do not extend: `src/options/` (`meta_controller.py`, `option.py`, `task_lifecycle.py`) and `src/continual_learning/stabilization.py`. Replace per v2 §10. Do not delete `stabilization.py` while `ColumnTrainer` imports it. Keep their tests green until the module is replaced; every replacement lands with its own tests.

## Commands

All commands run inside the `ens491` conda env (`conda activate ens491`); never install packages outside it.

```bash
pytest tests/ -v                     # all tests
pytest tests/test_skill_node.py -v   # one file
streamlit run src/gui/app.py         # GUI
```

Code graph (`graphify-out/`, gitignored, built locally): set up once per clone with `uv tool install graphifyy`, `graphify hook install` and `graphify update .`; graphify's git hooks then rebuild it after commits, pulls and branch switches. If the `graphify` command is not installed, skip the `## graphify` rules below and search the code normally.

## Stack

- Python 3.10, PyTorch, MiniGrid 3.1.0, Gymnasium 1.3.0, Stable-Baselines3 2.9.0, sb3-contrib 2.9.0 (MaskablePPO), W&B, Streamlit. Versions are pinned exactly in `requirements.txt` (MiniGrid can change env semantics silently).
- The team works on different machines and OSes, with and without a GPU; code must run on all of them. Never hardcode a device, OS-specific path or shell. Device default is `cpu` (`resolve_device`; `auto` = cuda → mps → cpu): the 147→64→64 MLP is too small to benefit. Keep `requirements.txt` free of CUDA-pinned wheels so it installs everywhere, macOS arm64 included.

## Core model (v2 §2–§6)

```
SkillNode {n,m}
  children     ()            → primitive: policy outputs a MiniGrid action (0–6), n = 0
  children     (c1, c2, …)   → composite: policy outputs a child index, n = 1 + max(child.n)
  initiation   I(s)          predicate on env state (default: always True)
  goal         β(s)          predicate on env state = terminal condition
  timeout      int           exceeding it = failure
  status       DECLARED → TRAINING → STABLE → FROZEN
```

- **Graph:** a DAG. A composite may use any earlier FROZEN node, at any level, as a child; a node may have several parents.
- **Execution:** one `Executor` keeps an active stack `[root, …, primitive]`. A selected child keeps control until its own termination (goal, timeout, or an ancestor terminated). Termination is checked top-down every env step; selection descends until a primitive is on top.
- **Training:** primitives use PPO on `SubgoalEnv`; composites use MaskablePPO on `OptionEnv`, where one step = one complete child execution (SMDP).
- **Reward:** from the node's **own `goal` predicate**, never the env reward, at every level: `1 − 0.9·steps/timeout` on success, else 0.
- **Masking is goal-only:** `available(s) = not goal(s) and (primitive or any child available)`. `initiation` is task metadata for reset hooks and does **not** mask.
- **Roots:** the GRU's task id selects the root node through a task registry.
- **Stabilization:** STABLE = success rate ≥ 0.90 *and* Wilson 95% lower bound ≥ 0.80 in two consecutive deterministic 50-episode evaluations on fixed seeds; the quoted success rate comes from a second, held-out seed set. Same rule for primitives and composites. (The v1 reward mean/std rule is retired.)

## Hard rules

**Observations**
- Pipeline: `ImgObsWrapper` → `FlattenObservation` → `ChannelNormalize` (÷`[10, 5, 2]` per channel) → `(147,)` float32, via `apply_obs_pipeline`. Never `FlatObsWrapper` (adds the mission string, not 147 dims). Never `CnnPolicy` (7×7 input is smaller than SB3's 8×8 kernel).
- Normalization is on for every new node and is part of its `spec_hash`; `SubgoalEnv`, `OptionEnv` and evaluation must use the same setting. Legacy `runs/pn4/col01` is raw-integer and cannot be a lateral source for a normalized node without retraining.
- Predicates (goal, initiation, masking) read an immutable `StateView` built once per env step, never `env.unwrapped`. Policies see only the 147-dim observation; this asymmetry is deliberate.

**Skill nodes**
- `children` is immutable after creation. Never grow a node's action space; create a new node.
- Only FROZEN nodes may be children or registered roots.
- Frozen = no gradients, forward pass still runs (laterals and parents need it). Never modify a frozen node's weights. Frozen nodes act deterministically by default.
- Level `n` is derived from children, never assigned by hand.
- Validate at creation: `timeout ≥ max(child.timeout)`, root `timeout ≤ env.max_steps`.
- `spec_hash` covers env, predicates, timeout, obs pipeline and children's hashes. A hash mismatch on load is an error, never a warning.

**Hierarchy**
- Unbounded depth: never hardcode the number of levels or nodes, never branch on `n` (`if level == 1:`). Executor and trainers must work at any depth.
- No per-layer MetaController: a composite's policy is its meta-controller.
- Availability is evaluated recursively and **memoized per env step** (exponential on a DAG otherwise). A mask handed to a policy always has at least one True; when nothing is available the episode ends (MaskablePPO cannot take an empty mask).
- `max_decisions` is mandatory on every composite: under goal-only masking a parent could otherwise re-select a timed-out child forever.
- `OptionEnv` uses `DummyVecEnv` only, never `SubprocVecEnv`.

**Signals**
- AE reconstruction error and reward-based signals stay separate; never merge them into one threshold.
- A parent's SMDP reward and a child's subgoal reward travel on separate channels.

**Scope**
- Do not put FourRooms (or any single benchmark) on the hierarchy's critical path.
- Do not reference or use the sandbox repo; build baselines from scratch here.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
