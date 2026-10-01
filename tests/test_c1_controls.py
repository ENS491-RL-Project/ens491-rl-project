"""
H-6.3: the shared C1 control arms.

The arms themselves are trivial; the measurement around them is not. Two things
here are easy to get subtly wrong and would corrupt every composite's reported
numbers rather than crashing:

  * **detecting a decision.** Closing a finished child and selecting the next one
    both happen inside one `Executor.act()` call, so probing the stack depth
    before each call misses every decision that follows a child's termination --
    which is most of them. `mask_divergence` reads the trace instead, and
    `test_mask_divergence_counts_a_decision_made_after_a_child_closed` is the
    regression for exactly that.
  * **policy leakage between arms.** `run_arms` must build a fresh node per arm;
    sharing one node would score the last arm's policy five times.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.envs.subgoal_env import NO_HOOK
from src.skills.node import ALWAYS, NEVER, Predicate, SkillNode
from src.training.c1_controls import (
    ARM_LABELS,
    RandomOverAll,
    RandomOverAvailable,
    capture_trace,
    mask_divergence,
    run_arms,
)

UNLOCK = "MiniGrid-Unlock-v0"


# --------------------------------------------------------------------------
# Scripted fixtures. Goals read `step_count` off the StateView, so the graph
# behaves identically in any MiniGrid env and the test measures the executor's
# bookkeeping rather than a policy's competence.
# --------------------------------------------------------------------------

def _step_at_least(k: int) -> Predicate:
    return Predicate(f"step>={k}", lambda sv: sv.step_count >= k)


def _step_odd() -> Predicate:
    return Predicate("step:odd", lambda sv: sv.step_count % 2 == 1)


class TurnLeft:
    def act(self, obs, deterministic: bool = True) -> int:
        return 0

    def parameters(self):
        return iter(())


class FirstAvailable:
    def select(self, obs, mask, deterministic: bool = True) -> int:
        return int(np.argmax(np.asarray(mask, dtype=bool)))

    def parameters(self):
        return iter(())


def _child(node_id, goal, initiation, timeout=10) -> SkillNode:
    node = SkillNode(node_id, goal=goal, initiation=initiation, timeout=timeout)
    node.policy = TurnLeft()
    return node


def _graph():
    """
    `a` closes on every odd step and is always applicable; `b` never reaches its
    goal and is never applicable. So the strict mask differs from the goal-only
    mask at every decision, and the parent is forced onto `b` the moment `a` has
    just succeeded.
    """
    a = _child("{0,a} a", _step_odd(), ALWAYS)
    b = _child("{0,b} b", _step_at_least(100), NEVER)
    parent = SkillNode(
        "{1,p} parent", children=[a, b],
        goal=_step_at_least(5), initiation=ALWAYS,
        timeout=12, max_decisions=20, env_ids=[UNLOCK],
    )
    parent.policy = FirstAvailable()
    return parent


# --------------------------------------------------------------------------
# mask_divergence
# --------------------------------------------------------------------------

def test_mask_divergence_counts_a_decision_made_after_a_child_closed():
    """
    The episode contains exactly two root decisions:

      step 0  `a` and `b` both available -> `a` chosen (strict allows it)
      step 1  `a`'s goal now holds so it closes and is unavailable; the only
              available child is `b`, which the strict mask forbids

    The second decision is made *inside* the same `act()` call that closed `a`,
    so a depth-probing implementation reports 1 decision instead of 2 and misses
    the one case where the strict mask would have changed the outcome.
    """
    node = _graph()
    div = mask_divergence(node, UNLOCK, NO_HOOK, [31])
    assert div["decisions"] == 2
    assert div["masks_differ"] == 2
    assert div["would_have_blocked_learned_choice"] == 1


def test_mask_divergence_reports_zero_when_the_masks_agree():
    a = _child("{0,a} a", _step_odd(), ALWAYS)
    b = _child("{0,b} b", _step_at_least(100), ALWAYS)   # applicable too
    node = SkillNode("{1,p} parent", children=[a, b], goal=_step_at_least(5),
                     timeout=12, max_decisions=20, env_ids=[UNLOCK])
    node.policy = FirstAvailable()
    div = mask_divergence(node, UNLOCK, NO_HOOK, [31])
    assert div["decisions"] >= 1
    assert div["masks_differ"] == 0
    assert div["would_have_blocked_learned_choice"] == 0


def test_mask_divergence_accumulates_over_episodes():
    node = _graph()
    one = mask_divergence(node, UNLOCK, NO_HOOK, [31])
    three = mask_divergence(node, UNLOCK, NO_HOOK, [31, 32, 33])
    assert three["episodes"] == 3
    assert three["decisions"] == 3 * one["decisions"]


# --------------------------------------------------------------------------
# run_arms
# --------------------------------------------------------------------------

def test_run_arms_scores_every_arm_and_does_not_leak_policies():
    learned = FirstAvailable()
    arms = run_arms(_graph, UNLOCK, NO_HOOK, [41, 42, 43], learned_policy=learned)
    assert set(arms) == set(ARM_LABELS)
    for name, res in arms.items():
        assert res.n_episodes == 3, name
    # A fresh node per arm: the factory's node must still carry its own policy,
    # not whichever arm ran last.
    assert isinstance(_graph().policy, FirstAvailable)


def test_run_arms_skips_learned_arms_when_nothing_is_trained():
    arms = run_arms(_graph, UNLOCK, NO_HOOK, [41], learned_policy=None)
    assert "learned" not in arms
    assert "random_over_available" in arms


def test_run_arms_honours_an_arm_subset():
    arms = run_arms(_graph, UNLOCK, NO_HOOK, [41],
                    learned_policy=FirstAvailable(),
                    arms=("learned", "random_strict_mask"))
    assert set(arms) == {"learned", "random_strict_mask"}


# --------------------------------------------------------------------------
# the random policies
# --------------------------------------------------------------------------

def test_random_over_available_only_returns_available_children():
    pol = RandomOverAvailable(seed=0)
    mask = [False, True, False, True]
    for _ in range(50):
        assert mask[pol.select(None, mask)]


def test_random_over_all_is_clipped_to_the_available_set():
    """
    The clip is required, not a convenience: `Executor` raises if a policy
    returns an unavailable child, so an unclipped arm could not run at all.
    """
    pol = RandomOverAll(seed=0)
    mask = [False, False, True]
    for _ in range(50):
        assert pol.select(None, mask) == 2


def test_random_policies_are_seeded():
    mask = [True, True, True]
    a = [RandomOverAvailable(seed=7).select(None, mask) for _ in range(5)]
    b = [RandomOverAvailable(seed=7).select(None, mask) for _ in range(5)]
    assert a == b


# --------------------------------------------------------------------------
# capture_trace
# --------------------------------------------------------------------------

def test_capture_trace_records_depth_and_selections():
    node = _graph()
    trace = capture_trace(node, UNLOCK, NO_HOOK, 31)
    assert trace["max_stack_depth"] == 2            # parent + one primitive
    assert trace["selected_children"]
    assert all({"depth", "node_id", "event"} <= set(e) for e in trace["trace"])


def test_capture_trace_reaches_depth_three_on_a_two_level_graph():
    """
    The check v2 §11's depth claim needs: a composite whose child is itself a
    composite must drive a 3-frame stack.
    """
    inner = _graph()
    inner.status = inner.status                      # unchanged; scripted, not frozen
    outer = SkillNode(
        "{2,p} outer", children=[inner], goal=_step_at_least(5),
        timeout=12, max_decisions=20, env_ids=[UNLOCK],
    )
    outer.policy = FirstAvailable()
    trace = capture_trace(outer, UNLOCK, NO_HOOK, 31)
    assert outer.n == 2
    assert trace["max_stack_depth"] == 3
    assert 0 in trace["levels_selected"] and 1 in trace["levels_selected"]
