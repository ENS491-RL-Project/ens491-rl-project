"""
H-6.1: env-step accounting for composites, and loading a frozen composite.

Three things H-6's measurements need and that did not exist after H-5:

  1. a cumulative env-step counter on `OptionEnv`, reachable through the monitor
     wrapper and through `DummyVecEnv.env_method` [H-6 A.4];
  2. `children_env_steps`, which walks a DAG's transitive closure once per node;
  3. `load_frozen_composite`, the composite analogue of `load_frozen_primitive`,
     which must refuse a spec mismatch like every other load path [A: F18].
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from src.envs.option_env import OptionEnv, make_option_env
from src.envs.subgoal_env import NO_HOOK, give_key
from src.skills.node import ALWAYS, SkillNode
from src.skills.predicates import carrying, door_open
from src.skills.registry import NodeRegistry, SpecMismatch
from src.training.node_trainer import (
    children_env_steps,
    load_frozen_composite,
    option_env_factory,
    set_torch_threads,
)
from src.types import NodeStatus

UNLOCK = "MiniGrid-Unlock-v0"


# --------------------------------------------------------------------------
# Fixtures: a composite over two scripted primitives
# --------------------------------------------------------------------------

class TurnPolicy:
    """Always turn left. Never reaches any goal, so every child times out."""

    def act(self, obs, deterministic: bool = True) -> int:
        return 0

    def parameters(self):
        return iter(())


class FirstChild:
    def select(self, obs, mask, deterministic: bool = True) -> int:
        return int(np.argmax(np.asarray(mask, dtype=bool)))

    def parameters(self):
        return iter(())


def _primitive(node_id: str, timeout: int = 4) -> SkillNode:
    node = SkillNode(node_id, goal=carrying("box"), timeout=timeout)
    node.policy = TurnPolicy()
    node.status = NodeStatus.FROZEN
    return node


def _composite(children, timeout: int = 40, max_decisions: int = 5) -> SkillNode:
    node = SkillNode(
        "{1,t} test_composite",
        children=children,
        goal=carrying("box"),          # never true in Unlock -> no early exit
        timeout=timeout,
        max_decisions=max_decisions,
        env_ids=[UNLOCK],
    )
    node.policy = FirstChild()
    return node


# --------------------------------------------------------------------------
# 1. the cumulative counter
# --------------------------------------------------------------------------

def test_total_env_steps_accumulates_across_episodes():
    """The per-episode `steps` resets; the cumulative counter must not."""
    node = _composite([_primitive("{0,a} a"), _primitive("{0,b} b")])
    env = make_option_env(node, UNLOCK, NO_HOOK, monitor=False)
    inner = env

    expected = 0
    for episode in range(3):
        inner.reset(seed=100 + episode)
        assert inner.steps == 0, "per-episode counter must reset"
        while True:
            _, _, term, trunc, info = inner.step(0)
            expected += info["k"]
            if term or trunc:
                break
        assert inner.get_total_env_steps() == expected
    env.close()
    assert expected > 0


def test_total_env_steps_equals_sum_of_k():
    node = _composite([_primitive("{0,a} a")])
    env = make_option_env(node, UNLOCK, NO_HOOK, monitor=False)
    env.reset(seed=7)
    total_k = 0
    for _ in range(3):
        _, _, term, trunc, info = env.step(0)
        total_k += info["k"]
        if term or trunc:
            break
    assert env.get_total_env_steps() == total_k
    env.close()


def test_counter_reachable_through_monitor_and_vec_env():
    """
    The passthrough is not cosmetic: Gymnasium 1.x does not forward arbitrary
    attributes through `Wrapper.__getattr__`, so a counter living on the inner
    `OptionEnv` is invisible from the outside without it. This is the same
    failure mode `action_masks` already had.
    """
    node = _composite([_primitive("{0,a} a"), _primitive("{0,b} b")])
    vec = DummyVecEnv([
        option_env_factory(node, UNLOCK, NO_HOOK, seed=3, monitor=True),
        option_env_factory(node, UNLOCK, NO_HOOK, seed=4, monitor=True),
    ])
    vec.reset()
    for _ in range(4):
        vec.step(np.zeros(2, dtype=np.int64))

    per_env = vec.env_method("get_total_env_steps")
    assert len(per_env) == 2
    assert all(isinstance(v, int) and v > 0 for v in per_env)
    assert sum(per_env) == int(sum(per_env))
    vec.close()


def test_counter_is_monotone():
    node = _composite([_primitive("{0,a} a")])
    env = make_option_env(node, UNLOCK, NO_HOOK, monitor=False)
    env.reset(seed=11)
    seen = [env.get_total_env_steps()]
    for _ in range(5):
        _, _, term, trunc, _ = env.step(0)
        seen.append(env.get_total_env_steps())
        if term or trunc:
            env.reset(seed=12)
            seen.append(env.get_total_env_steps())
    assert seen == sorted(seen)
    env.close()


# --------------------------------------------------------------------------
# 2. children_env_steps over a DAG
# --------------------------------------------------------------------------

def test_children_env_steps_sums_primitive_and_composite_costs():
    a = _primitive("{0,a} a")
    a.meta["steps_trained"] = 1_000
    b = _primitive("{0,b} b")
    b.meta["steps_trained"] = 2_000
    mid = SkillNode("{1,m} mid", children=[a, b], goal=carrying("box"), timeout=10)
    mid.meta["env_steps_own"] = 500
    mid.status = NodeStatus.FROZEN

    top = SkillNode("{2,t} top", children=[mid], goal=carrying("box"), timeout=20)
    assert children_env_steps(top) == 3_500


def test_children_env_steps_counts_a_shared_child_once():
    """A DAG child with two parents is pretraining paid for once, not twice."""
    shared = _primitive("{0,s} shared")
    shared.meta["steps_trained"] = 7_000
    left = SkillNode("{1,l} left", children=[shared], goal=carrying("box"), timeout=10)
    left.status = NodeStatus.FROZEN
    right = SkillNode("{1,r} right", children=[shared], goal=carrying("box"), timeout=10)
    right.status = NodeStatus.FROZEN

    top = SkillNode("{2,t} top", children=[left, right], goal=carrying("box"), timeout=20)
    assert children_env_steps(top) == 7_000


def test_children_env_steps_understates_rather_than_guesses():
    """A child with no recorded cost contributes 0, not an estimate."""
    a = _primitive("{0,a} a")          # no meta at all
    top = SkillNode("{1,t} top", children=[a], goal=carrying("box"), timeout=10)
    assert children_env_steps(top) == 0


def test_children_env_steps_is_zero_for_a_primitive():
    assert children_env_steps(_primitive("{0,a} a")) == 0


# --------------------------------------------------------------------------
# 3. load_frozen_composite
# --------------------------------------------------------------------------

def _real_children():
    """Two primitives with real SB3 policies, frozen, as a parent would see them."""
    from src.training.node_trainer import attach_sb3_primitive_policy

    get_key = SkillNode("{0,x} get_key_t", goal=carrying("key"), timeout=20,
                        env_ids=[UNLOCK], train_hook="none")
    attach_sb3_primitive_policy(get_key, UNLOCK, NO_HOOK)
    get_key.freeze()

    open_d = SkillNode("{0,y} open_door_t", goal=door_open(), initiation=carrying("key"),
                       timeout=20, env_ids=[UNLOCK], train_hook="give_key")
    attach_sb3_primitive_policy(open_d, UNLOCK, give_key())
    open_d.freeze()
    return [get_key, open_d]


def _real_composite(children, timeout: int = 40) -> SkillNode:
    return SkillNode(
        "{1,z} unlock_t",
        children=children,
        goal=door_open(),
        timeout=timeout,
        max_decisions=4,
        env_ids=[UNLOCK],
    )


def test_load_frozen_composite_round_trips(tmp_path):
    from src.training.node_trainer import CompositePolicyAdapter
    from sb3_contrib import MaskablePPO

    children = _real_children()
    node = _real_composite(children)
    vec = DummyVecEnv([option_env_factory(node, UNLOCK, NO_HOOK, monitor=False)])
    model = MaskablePPO("MlpPolicy", vec, device="cpu", verbose=0, n_steps=16, batch_size=16)
    node.policy = CompositePolicyAdapter(model.policy)
    vec.close()
    node.status = NodeStatus.STABLE
    node.freeze()

    registry = NodeRegistry(tmp_path)
    registry.save(node, extra_meta={"env_steps_own": 1234})

    fresh = _real_composite(_real_children())
    load_frozen_composite(fresh, registry, UNLOCK, NO_HOOK)
    assert fresh.status is NodeStatus.FROZEN
    assert fresh.spec_hash() == node.spec_hash()
    assert fresh.meta["env_steps_own"] == 1234

    obs = np.zeros(147, dtype=np.float32)
    mask = [True, True]
    assert fresh.policy.select(obs, mask) == node.policy.select(obs, mask)


def test_load_frozen_composite_refuses_a_changed_child(tmp_path):
    """
    `spec_hash` recurses into the children's hashes, so changing a child's spec
    must invalidate the parent. This is the mechanism that forces H-6.2 to
    retrain `{1,1} unlock_door` once `initiation` is declared on it.
    """
    from src.training.node_trainer import CompositePolicyAdapter
    from sb3_contrib import MaskablePPO

    children = _real_children()
    node = _real_composite(children)
    vec = DummyVecEnv([option_env_factory(node, UNLOCK, NO_HOOK, monitor=False)])
    node.policy = CompositePolicyAdapter(
        MaskablePPO("MlpPolicy", vec, device="cpu", verbose=0,
                    n_steps=16, batch_size=16).policy
    )
    vec.close()
    node.status = NodeStatus.STABLE
    node.freeze()
    registry = NodeRegistry(tmp_path)
    registry.save(node)

    changed = _real_children()
    changed[0].timeout = 21            # a different child spec
    fresh = _real_composite(changed)
    with pytest.raises(SpecMismatch):
        load_frozen_composite(fresh, registry, UNLOCK, NO_HOOK)


def test_load_frozen_composite_rejects_an_unfrozen_child(tmp_path):
    children = _real_children()
    children[1].status = NodeStatus.STABLE
    node = _real_composite(children)
    with pytest.raises(RuntimeError, match="load the children before the parent"):
        load_frozen_composite(node, NodeRegistry(tmp_path), UNLOCK, NO_HOOK)


def test_load_frozen_composite_rejects_a_primitive(tmp_path):
    node = _primitive("{0,a} a")
    with pytest.raises(ValueError, match="load_frozen_primitive"):
        load_frozen_composite(node, NodeRegistry(tmp_path), UNLOCK, NO_HOOK)


# --------------------------------------------------------------------------
# 4. thread control
# --------------------------------------------------------------------------

def test_set_torch_threads_clamps_and_reports():
    import torch

    before = torch.get_num_threads()
    try:
        assert set_torch_threads(2) == 2
        assert set_torch_threads(0) == 1          # clamped, never 0
    finally:
        torch.set_num_threads(before)
