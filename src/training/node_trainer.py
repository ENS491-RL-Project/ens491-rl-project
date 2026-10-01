"""
Node trainer: one entry point for primitive and composite nodes.

Target Architecture v2 §5.1/§5.2 and §10 ("generic node trainer"). A primitive
trains with PPO on `SubgoalEnv`; a composite trains with MaskablePPO on
`OptionEnv`. Both stop on the §5.3 criterion, evaluated by `EvaluationCallback`.

Seeding and the wrapper chain follow `ColumnTrainer`, which this generalizes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.continual_learning.column_policy import ColumnPolicy
from src.envs.obs import apply_obs_pipeline, obs_pipeline_name
from src.envs.subgoal_env import NO_HOOK, ResetHook, SubgoalEnv
from src.skills.node import SkillNode
from src.training.device import resolve_device
from src.training.evaluation import (
    DEFAULT_N_EVAL,
    EvalResult,
    EvaluationCallback,
    StabilityCriterion,
    evaluate_node,
    heldout_seeds,
    stopping_seeds,
)
from src.types import NodeStatus


class PrimitivePolicyAdapter:
    """Wraps an SB3 policy as a v2 PrimitivePolicy [A: F19]."""

    def __init__(self, sb3_policy) -> None:
        self.sb3_policy = sb3_policy

    def act(self, obs, deterministic: bool = True) -> int:
        action, _ = self.sb3_policy.predict(obs, deterministic=deterministic)
        return int(action)

    # So NodeRegistry can persist it and freeze() can clear grads.
    def state_dict(self):
        return self.sb3_policy.state_dict()

    def load_state_dict(self, sd):
        return self.sb3_policy.load_state_dict(sd)

    def parameters(self):
        return self.sb3_policy.parameters()


# --------------------------------------------------------------------------
# Env construction
# --------------------------------------------------------------------------

def subgoal_env_factory(
    node: SkillNode,
    env_id: str,
    reset_hook: ResetHook = NO_HOOK,
    *,
    seed: Optional[int] = None,
    normalize: bool = True,
    monitor: bool = True,
) -> Callable[[], gym.Env]:
    def _make() -> gym.Env:
        env = gym.make(env_id)
        env = SubgoalEnv(env, node, reset_hook)
        env = apply_obs_pipeline(env, normalize=normalize)
        if monitor:
            env = Monitor(env)
        if seed is not None:
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        return env

    return _make


def make_training_vec_env(
    node: SkillNode,
    env_ids: Sequence[str],
    reset_hook: ResetHook,
    *,
    n_envs: int = 8,
    seed: int = 42,
    normalize: bool = True,
) -> DummyVecEnv:
    """
    [A: F29] Layouts are mixed across the vectorized envs, not trained one after
    the other, so the policy never has a phase where it only sees one layout.

    DummyVecEnv only [A: F15]: SubprocVecEnv would need every frozen child
    reloaded per subprocess, and the shared-module hazard is only safe because
    DummyVecEnv steps sequentially.
    """
    factories = []
    for i in range(n_envs):
        env_id = env_ids[i % len(env_ids)]
        factories.append(subgoal_env_factory(
            node, env_id, reset_hook, seed=seed + i, normalize=normalize
        ))
    return DummyVecEnv(factories)


def pooled_eval_fn(
    node: SkillNode,
    env_ids: Sequence[str],
    reset_hook: ResetHook,
    *,
    n_episodes: int = DEFAULT_N_EVAL,
    normalize: bool = True,
    seed_base: Optional[int] = None,
    prepare: Optional[Callable] = None,
    env_factory_fn: Optional[Callable] = None,
    initiation_masking: bool = False,
) -> Callable[[], EvalResult]:
    """
    Evaluate on every training layout and pool the episodes into one score.

    A node trained on two layouts must clear the bar on the mixture, not on its
    easier half: pooling is what stops a primitive that works on Unlock and
    fails on DoorKey from being frozen.
    """
    base = seed_base if seed_base is not None else stopping_seeds(1)[0]
    per_env = max(1, n_episodes // len(env_ids))
    # Primitives evaluate on SubgoalEnv; composites evaluate in execution mode on
    # a plain env, with the Executor driving the whole stack (v2 §5.3).
    build = env_factory_fn or (
        lambda env_id, hook, normalize=True: subgoal_env_factory(
            node, env_id, hook, normalize=normalize, monitor=False
        )
    )

    def _eval() -> EvalResult:
        successes, lengths, outcomes, timeouts = 0, [], [], 0
        for k, env_id in enumerate(env_ids):
            seeds = [base + k * 1000 + i for i in range(per_env)]
            res = evaluate_node(
                node,
                build(env_id, reset_hook, normalize=normalize),
                seeds=seeds,
                prepare=prepare,
                initiation_masking=initiation_masking,
            )
            successes += res.successes
            lengths.extend(res.lengths)
            outcomes.extend(res.per_episode)
            timeouts += res.timeouts
        n = len(outcomes)
        return EvalResult(
            n_episodes=n,
            successes=successes,
            mean_length=(sum(lengths) / n) if n else 0.0,
            seed_base=base,
            per_episode=outcomes,
            lengths=lengths,
            timeouts=timeouts,
        )

    return _eval


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

# PPO hyperparameters for MiniGrid primitives.
#
# SB3's defaults are tuned for continuous control and are a poor fit here:
#   n_steps=2048 x 8 envs = 16k transitions per update, so a 600k-step budget
#   buys only ~36 gradient updates; and ent_coef=0.0 gives no exploration
#   pressure at all, which matters because these goals are sparse -- the reward
#   is zero until the key is picked up or the door opens.
# n_steps=256 x 8 envs = 2048 per update (~290 updates in the same budget) and
# ent_coef=0.01 is the usual MiniGrid setting.
PPO_DEFAULTS = dict(
    n_steps=256,
    batch_size=256,
    n_epochs=4,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
)


def measure_throughput(vec_env, model, n_steps: int = 2048) -> float:
    """[A: F31] Print steps/s before committing to a budget."""
    t0 = time.perf_counter()
    model.learn(total_timesteps=n_steps, reset_num_timesteps=False, progress_bar=False)
    dt = time.perf_counter() - t0
    return n_steps / dt if dt > 0 else float("nan")


def train_primitive(
    node: SkillNode,
    env_ids: Sequence[str],
    reset_hook: ResetHook = NO_HOOK,
    *,
    total_timesteps: int = 500_000,
    eval_every: int = 20_000,
    n_eval: int = DEFAULT_N_EVAL,
    n_envs: int = 8,
    seed: int = 42,
    device: str = "cpu",
    lateral_source=None,
    normalize: bool = True,
    ppo_overrides: Optional[dict] = None,
    verbose: int = 0,
) -> dict:
    """Train one primitive node to STABLE (or to the budget). Returns a report."""
    dev = resolve_device(device)
    node.status = NodeStatus.TRAINING

    vec = make_training_vec_env(
        node, env_ids, reset_hook, n_envs=n_envs, seed=seed, normalize=normalize
    )
    ppo_kwargs = dict(PPO_DEFAULTS)
    ppo_kwargs.update(ppo_overrides or {})
    model = PPO(
        ColumnPolicy,
        vec,
        policy_kwargs={"lateral_source_column": lateral_source},
        seed=seed,
        device=dev,
        verbose=0,
        **ppo_kwargs,
    )
    node.policy = PrimitivePolicyAdapter(model.policy)

    criterion = StabilityCriterion()
    callback = EvaluationCallback(
        node,
        env_factory=None,
        eval_fn=pooled_eval_fn(node, env_ids, reset_hook,
                               n_episodes=n_eval, normalize=normalize),
        eval_every=eval_every,
        criterion=criterion,
        verbose=verbose,
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    wall = time.perf_counter() - t0
    steps_done = model.num_timesteps
    vec.close()

    report = {
        "node_id": node.id,
        "env_ids": list(env_ids),
        "reset_hook": reset_hook.spec,
        "timeout": node.timeout,
        "normalize_obs": normalize,
        "obs_pipeline": obs_pipeline_name(normalize),
        "seed": seed,
        "device": dev,
        "n_envs": n_envs,
        "eval_every": eval_every,
        "ppo": ppo_kwargs,
        "budget": total_timesteps,
        "steps_trained": steps_done,
        "wall_seconds": round(wall, 1),
        "steps_per_second": round(steps_done / wall, 1) if wall > 0 else None,
        "stable": criterion.is_stable,
        "criterion": criterion.as_dict(),
    }
    if criterion.is_stable:
        node.status = NodeStatus.STABLE
    return report, model


def freeze_and_save(
    node: SkillNode,
    registry,
    report: dict,
    *,
    env_ids: Sequence[str],
    reset_hook: ResetHook,
    normalize: bool = True,
    n_eval: int = DEFAULT_N_EVAL,
    prepare: Optional[Callable] = None,
) -> dict:
    """
    Freeze, score on the held-out seed set, and persist.

    [A: F11] The stopping set decided when to stop, so it is optimistically
    biased. The number quoted for the node is the held-out one.
    """
    node.freeze()
    heldout = pooled_eval_fn(
        node, env_ids, reset_hook, n_episodes=n_eval, normalize=normalize,
        seed_base=heldout_seeds(1)[0],
    )()
    report = dict(report)
    report["heldout"] = heldout.as_dict()
    if prepare is not None:
        reach = pooled_eval_fn(
            node, env_ids, reset_hook, n_episodes=n_eval, normalize=normalize,
            seed_base=heldout_seeds(1)[0] + 500, prepare=prepare,
        )()
        report["parent_reachable"] = reach.as_dict()

    node.meta.update(report)
    registry.save(node, extra_meta=report)
    return report


# --------------------------------------------------------------------------
# Composite nodes -- v2 §5.2
# --------------------------------------------------------------------------

class CompositePolicyAdapter:
    """Wraps a MaskablePPO policy as a v2 CompositePolicy [A: F19]."""

    def __init__(self, sb3_policy) -> None:
        self.sb3_policy = sb3_policy

    def select(self, obs, mask, deterministic: bool = True) -> int:
        action, _ = self.sb3_policy.predict(
            obs, deterministic=deterministic, action_masks=np.asarray(mask, dtype=bool)
        )
        return int(action)

    def state_dict(self):
        return self.sb3_policy.state_dict()

    def load_state_dict(self, sd):
        return self.sb3_policy.load_state_dict(sd)

    def parameters(self):
        return self.sb3_policy.parameters()


class RandomOverAvailable:
    """
    C1 control (a): choose uniformly among the children the mask allows.

    [A: F1] This is the number that says how much of a composite's score is
    composition and how much is the mask doing the work. It is reported beside
    every composite, not kept as an optional extra.
    """

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def select(self, obs, mask, deterministic: bool = True) -> int:
        idx = [i for i, ok in enumerate(mask) if ok]
        return int(self.rng.choice(idx))


def option_env_factory(
    node: SkillNode,
    env_id: str,
    reset_hook: ResetHook = NO_HOOK,
    *,
    seed: Optional[int] = None,
    normalize: bool = True,
    monitor: bool = True,
    initiation_masking: bool = False,
) -> Callable[[], gym.Env]:
    from src.envs.option_env import make_option_env

    def _make() -> gym.Env:
        return make_option_env(
            node, env_id, reset_hook, seed=seed, normalize=normalize,
            monitor=monitor, initiation_masking=initiation_masking,
        )

    return _make


def execution_env_factory(
    env_id: str,
    reset_hook: ResetHook = NO_HOOK,
    *,
    normalize: bool = True,
) -> Callable[[], gym.Env]:
    """
    A plain env for evaluating a node in execution mode.

    Note this is NOT an OptionEnv: evaluation drives the whole stack through the
    Executor, exactly as the runtime would, so a composite and a primitive are
    scored by the same code (v2 §5.3).
    """
    from src.envs.option_env import ResetHookEnv

    def _make() -> gym.Env:
        env = gym.make(env_id)
        env = ResetHookEnv(env, reset_hook)
        return apply_obs_pipeline(env, normalize=normalize)

    return _make


def train_composite(
    node: SkillNode,
    env_ids: Sequence[str],
    reset_hook: ResetHook = NO_HOOK,
    *,
    total_timesteps: int = 200_000,
    eval_every: int = 5_000,
    n_eval: int = DEFAULT_N_EVAL,
    n_envs: int = 8,
    seed: int = 42,
    device: str = "cpu",
    normalize: bool = True,
    initiation_masking: bool = False,
    ppo_overrides: Optional[dict] = None,
    verbose: int = 0,
) -> tuple[dict, object]:
    """
    Train one composite with MaskablePPO on OptionEnv.

    `total_timesteps` counts SMDP decisions, not env steps -- one decision is a
    whole child execution, so the env-step cost is far higher and is reported
    separately.
    """
    from sb3_contrib import MaskablePPO

    for child in node.children:
        if child.status is not NodeStatus.FROZEN:
            raise RuntimeError(
                f"{node.id}: child {child.id} is {child.status.value}, not frozen. "
                "Only FROZEN nodes may be children (v2 §9)."
            )

    dev = resolve_device(device)
    node.status = NodeStatus.TRAINING

    # [A: F15] DummyVecEnv only: frozen children are shared modules and
    # SubprocVecEnv would need each one reloaded per subprocess.
    vec = DummyVecEnv([
        option_env_factory(node, env_ids[i % len(env_ids)], reset_hook,
                           seed=seed + i, normalize=normalize,
                           initiation_masking=initiation_masking)
        for i in range(n_envs)
    ])

    ppo_kwargs = dict(PPO_DEFAULTS)
    ppo_kwargs.update(ppo_overrides or {})
    model = MaskablePPO("MlpPolicy", vec, seed=seed, device=dev, verbose=0, **ppo_kwargs)
    node.policy = CompositePolicyAdapter(model.policy)

    criterion = StabilityCriterion()
    # The ablation must be SCORED under the mask it was TRAINED with, or the
    # two C1 arms are not comparable [A: F1].
    eval_fn = pooled_eval_fn(
        node, env_ids, reset_hook, n_episodes=n_eval, normalize=normalize,
        env_factory_fn=execution_env_factory,
        initiation_masking=initiation_masking,
    )
    callback = EvaluationCallback(
        node, env_factory=None, eval_fn=eval_fn,
        eval_every=eval_every, criterion=criterion, verbose=verbose,
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    wall = time.perf_counter() - t0
    decisions = model.num_timesteps
    # [H-6 A.4] Read the counters BEFORE closing the vec env. One SMDP decision
    # is a whole child execution, so decisions say nothing about env-step cost --
    # and env steps are the unit a flat baseline is measured in.
    env_steps_own = int(sum(vec.env_method("get_total_env_steps")))
    vec.close()

    report = {
        "node_id": node.id,
        "kind": "composite",
        "children": [c.id for c in node.children],
        "level_n": node.n,
        "env_ids": list(env_ids),
        "reset_hook": reset_hook.spec,
        "timeout": node.timeout,
        "max_decisions": node.max_decisions,
        "initiation_masking": initiation_masking,
        "normalize_obs": normalize,
        "obs_pipeline": obs_pipeline_name(normalize),
        "seed": seed,
        "device": dev,
        "n_envs": n_envs,
        "eval_every": eval_every,
        "ppo": ppo_kwargs,
        "budget_decisions": total_timesteps,
        "decisions_trained": decisions,
        # [H-6 A.4] The three fields v2 §11 now requires of every composite.
        "env_steps_own": env_steps_own,
        "env_steps_per_decision": (
            round(env_steps_own / decisions, 2) if decisions else None
        ),
        "env_steps_children": children_env_steps(node),
        "env_steps_total": env_steps_own + children_env_steps(node),
        "wall_seconds": round(wall, 1),
        "decisions_per_second": round(decisions / wall, 1) if wall > 0 else None,
        "stable": criterion.is_stable,
        # Evaluations happen every `eval_every` decisions, so the decision count
        # at the evaluation that completed the streak is simply how many ran.
        "decisions_to_stable": (
            len(criterion.history) * eval_every if criterion.is_stable else None
        ),
        "criterion": criterion.as_dict(),
    }
    if criterion.is_stable:
        node.status = NodeStatus.STABLE
    return report, model


def attach_sb3_primitive_policy(
    node: SkillNode,
    env_id: str,
    reset_hook: ResetHook = NO_HOOK,
    *,
    normalize: bool = True,
    device: str = "cpu",
    lateral_source=None,
) -> SkillNode:
    """
    Give `node` a freshly built SB3 policy of the right shape, ready for weights.

    The network shape comes from the env, not from the checkpoint, so a throwaway
    PPO is constructed exactly as `ColumnTrainer.load_policy_into` does.
    """
    env = DummyVecEnv([subgoal_env_factory(node, env_id, reset_hook,
                                           normalize=normalize, monitor=False)])
    model = PPO(
        ColumnPolicy, env,
        policy_kwargs={"lateral_source_column": lateral_source},
        device=resolve_device(device), verbose=0,
    )
    node.policy = PrimitivePolicyAdapter(model.policy)
    env.close()
    return node


def load_frozen_primitive(
    node: SkillNode,
    registry,
    env_id: str,
    reset_hook: ResetHook = NO_HOOK,
    *,
    normalize: bool = True,
    device: str = "cpu",
) -> SkillNode:
    """Build the policy, load the checkpoint, and verify the spec hash [A: F18]."""
    attach_sb3_primitive_policy(node, env_id, reset_hook,
                                normalize=normalize, device=device)
    registry.load_into(node, require_frozen=True)
    return node


def children_env_steps(node: SkillNode) -> int:
    """
    Pretraining cost of everything below `node`, in env steps [H-6 A.4].

    Walks the transitive closure (`descendants()` already de-duplicates a DAG's
    shared children, so a node reached by two parents is counted once) and reads
    each child's recorded training cost out of the `meta` the registry persisted.
    A primitive records `steps_trained`; a composite records `env_steps_own`.

    Returns 0 for a child whose meta carries neither, rather than guessing. That
    is visible in the report as an understated total, which is preferable to a
    fabricated one; `{1,1} unlock_door` was in exactly that state until H-6.2
    retrained it with the counter in place.
    """
    total = 0
    for child in node.descendants():
        meta = child.meta or {}
        if "env_steps_own" in meta:
            total += int(meta["env_steps_own"])
        elif "steps_trained" in meta:
            total += int(meta["steps_trained"])
    return total


def load_frozen_composite(
    node: SkillNode,
    registry,
    env_id: str,
    reset_hook: ResetHook = NO_HOOK,
    *,
    normalize: bool = True,
    device: str = "cpu",
    initiation_masking: bool = False,
) -> SkillNode:
    """
    The composite analogue of `load_frozen_primitive`.

    A composite's policy shape comes from `OptionEnv` -- `Discrete(len(children))`
    over the 147-dim observation -- so a throwaway `MaskablePPO` is built exactly
    as the trainer would build it, and the weights are loaded into that. The
    children must already be FROZEN and loaded: `spec_hash` recurses into their
    hashes, so a parent cannot be verified before its children are [A: F18].
    """
    from sb3_contrib import MaskablePPO

    if node.is_primitive:
        raise ValueError(f"{node.id}: use load_frozen_primitive for a primitive node")
    for child in node.children:
        if child.status is not NodeStatus.FROZEN:
            raise RuntimeError(
                f"{node.id}: child {child.id} is {child.status.value}; load the "
                "children before the parent (v2 §9)"
            )

    vec = DummyVecEnv([
        option_env_factory(node, env_id, reset_hook, normalize=normalize,
                           monitor=False, initiation_masking=initiation_masking)
    ])
    try:
        model = MaskablePPO("MlpPolicy", vec, device=resolve_device(device), verbose=0)
        node.policy = CompositePolicyAdapter(model.policy)
    finally:
        vec.close()

    registry.load_into(node, require_frozen=True)
    return node


def set_torch_threads(n: int) -> int:
    """
    Cap torch's intra-op threads so several runs can share the machine.

    A `DummyVecEnv` run is almost entirely single-threaded Python, but torch
    defaults to one thread per core (16 here), so three concurrent seeds would
    contend for every core instead of using one each. Call this before building
    any model. Returns the value actually set.
    """
    import torch

    n = max(1, int(n))
    torch.set_num_threads(n)
    return torch.get_num_threads()
