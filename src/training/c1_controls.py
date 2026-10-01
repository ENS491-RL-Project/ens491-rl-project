"""
The C1 control arms, shared by every composite experiment.

Target Architecture v2 §11 [A: F1]. H-5 made the two masking controls mandatory
beside every composite rather than optional extras, and it was right to: on
`{1,1} unlock_door` a random policy under the strict initiation mask matched the
learned composite exactly, so the headline success rate on its own said nothing
about whether composition helped.

This module exists so the arms are defined once and every experiment reports the
same five, computed the same way. H-6 runs it three times (`{2,1} solve_doorkey`,
`{1,B} flat_doorkey_options`, `{1,2} unlock_matching`), and a per-script copy
would drift.

The five arms, from weakest to strongest:

    random_over_all        ignores the mask (clipped to legal) -- what the mask
                           itself is worth
    random_over_available  uniform over the goal-only mask
    random_strict_mask     uniform over the strict (initiation) mask -- the arm
                           that tied the learned composite in H-5
    learned                the trained policy under the goal-only mask
    learned_strict_mask    the trained policy under the strict mask

Every arm is scored on the same held-out seeds, through the same `Executor`, and
each is scored under the mask it would run with -- scoring an arm under a
different mask than it acts with is the plumbing bug H-5 found and fixed.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from src.skills.executor import Executor
from src.skills.node import SkillNode
from src.skills.state import build_state_view
from src.training.evaluation import EvalResult, evaluate_node


# `RandomOverAvailable` lives in `node_trainer` (H-5) and is re-exported here so
# the arm list reads as one table. One definition, two import paths.
from src.training.node_trainer import RandomOverAvailable  # noqa: E402


class RandomOverAll:
    """
    Ignore the mask entirely, clipped to the available set.

    The clip is not a cheat: `Executor` raises if a policy returns an unavailable
    child, so an unclipped version could not run at all. This arm measures what
    the mask is worth on its own, which is the floor every other arm is read
    against.
    """

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def select(self, obs, mask, deterministic: bool = True) -> int:
        idx = int(self.rng.integers(0, len(mask)))
        if not mask[idx]:
            idx = next(i for i, ok in enumerate(mask) if ok)
        return idx


# Arm name -> (policy factory, initiation_masking). `None` means "the learned
# policy", substituted by `run_arms`.
ARMS: tuple[tuple[str, Optional[Callable[[int], object]], bool], ...] = (
    ("random_over_all", lambda seed: RandomOverAll(seed), False),
    ("random_over_available", lambda seed: RandomOverAvailable(seed), False),
    ("random_strict_mask", lambda seed: RandomOverAvailable(seed), True),
    ("learned", None, False),
    ("learned_strict_mask", None, True),
)

ARM_LABELS = {
    "random_over_all": "random over all children (mask clipped)",
    "random_over_available": "random over available children",
    "random_strict_mask": "random, initiation masking ON",
    "learned": "learned composite",
    "learned_strict_mask": "learned composite, initiation masking ON",
}


def run_arms(
    build_node: Callable[[], SkillNode],
    env_id: str,
    reset_hook,
    seeds: Sequence[int],
    *,
    learned_policy=None,
    learned_strict_policy=None,
    env_factory_fn: Optional[Callable] = None,
    rng_seed: int = 0,
    arms: Sequence[str] = tuple(a[0] for a in ARMS),
) -> dict[str, EvalResult]:
    """
    Score every requested arm on the same seeds.

    `build_node()` must return a *fresh* node with the same children and spec,
    so an arm's policy cannot leak into the next arm. `learned_strict_policy`
    defaults to `learned_policy`: when the same weights are scored under both
    masks that is the ablation, and when a separately trained strict-mask run
    exists its policy is passed instead.
    """
    from src.training.node_trainer import execution_env_factory

    build_env = env_factory_fn or execution_env_factory
    strict_policy = learned_strict_policy or learned_policy
    out: dict[str, EvalResult] = {}

    for name, factory, strict in ARMS:
        if name not in arms:
            continue
        if factory is None:
            policy = strict_policy if strict else learned_policy
            if policy is None:
                continue                      # nothing trained yet; skip, do not fake
        else:
            policy = factory(rng_seed)

        probe = build_node()
        probe.policy = policy
        out[name] = evaluate_node(
            probe,
            build_env(env_id, reset_hook),
            seeds=list(seeds),
            initiation_masking=strict,
        )
    return out


def mask_divergence(
    node: SkillNode,
    env_id: str,
    reset_hook,
    seeds: Sequence[int],
    *,
    env_factory_fn: Optional[Callable] = None,
) -> dict:
    """
    How much of the learned policy's behaviour the strict mask would have handed it.

    At every decision point along the learned policy's own trajectory, compare
    the goal-only mask with the strict one. Two counts matter:

      `masks_differ`         decisions where the strict mask would have withheld
                             at least one child the goal-only mask offered
      `would_have_blocked`   of those, decisions where it would have withheld the
                             child the learned policy actually chose

    `would_have_blocked == 0` over many differing decisions is the evidence that
    the policy *discovered* the ordering constraint instead of being handed it --
    the only claim that survives when the strict-mask control ties the learned
    arm on success rate. A non-zero count is equally informative and means the
    policy is choosing children the task representation calls inapplicable.
    """
    from src.training.node_trainer import execution_env_factory

    build_env = env_factory_fn or execution_env_factory
    env = build_env(env_id, reset_hook)()
    masks_differ = decisions = would_have_blocked = 0
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            version = 0
            sv = build_state_view(env, version)
            ex = Executor(node)
            for _ in range(node.timeout):
                # A decision cannot be detected by probing the stack before
                # `act()`: closing a finished child and selecting the next one
                # both happen INSIDE one `act()` call, so the stack is already
                # back down to the new child by the time the call returns. The
                # trace is the reliable record -- a SELECT at depth 2 is this
                # node choosing one of its own children, and every selection
                # inside one call was made at the same `sv`.
                mark = len(ex.trace)
                action = ex.act(obs, sv)
                chosen = [
                    e.node_id for e in ex.trace[mark:]
                    if e.event == "SELECT" and e.depth == 2
                ]
                if chosen:
                    goal_only = [c.available(sv, {}) for c in node.children]
                    strict = [a and bool(c.initiation(sv))
                              for a, c in zip(goal_only, node.children)]
                    decisions += 1
                    if goal_only != strict:
                        masks_differ += 1
                        picked = next(i for i, c in enumerate(node.children)
                                      if c.id == chosen[-1])
                        if not strict[picked]:
                            would_have_blocked += 1
                if action is None:
                    break
                obs, _, term, trunc, _ = env.step(action)
                version += 1
                sv = build_state_view(env, version)
                ex.observe_reward(0.0)
                if term or trunc or node.goal(sv):
                    break
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    return {
        "episodes": len(seeds),
        "decisions": decisions,
        "masks_differ": masks_differ,
        "would_have_blocked_learned_choice": would_have_blocked,
    }


def capture_trace(
    node: SkillNode,
    env_id: str,
    reset_hook,
    seed: int,
    *,
    env_factory_fn: Optional[Callable] = None,
) -> dict:
    """
    One episode's execution trace, plus the maximum stack depth it reached.

    v2 §11 claims `{2,1} solve_doorkey` "exercises a mixed-level DAG and a
    depth-3 stack". That is checked here rather than asserted: the trace records
    every SELECT and CLOSE with its depth, so the claim is either in the data or
    it is not.
    """
    from src.training.node_trainer import execution_env_factory

    build_env = env_factory_fn or execution_env_factory
    env = build_env(env_id, reset_hook)()
    try:
        obs, _ = env.reset(seed=seed)
        version = 0
        sv = build_state_view(env, version)
        ex = Executor(node)
        max_depth = ex.depth
        for _ in range(node.timeout):
            action = ex.act(obs, sv)
            max_depth = max(max_depth, ex.depth)
            if action is None:
                break
            obs, _, term, trunc, _ = env.step(action)
            version += 1
            sv = build_state_view(env, version)
            ex.observe_reward(0.0)
            max_depth = max(max_depth, ex.depth)
            if term or trunc:
                break
        return {
            "seed": seed,
            "env_id": env_id,
            "success": bool(node.goal(sv)),
            "env_steps": version,
            "max_stack_depth": max_depth,
            "levels_selected": sorted({
                c.n for c in [node, *node.descendants()]
                if c.id in set(ex.selected_children())
            }),
            "selected_children": ex.selected_children(),
            "trace": ex.trace_dicts(),
        }
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
