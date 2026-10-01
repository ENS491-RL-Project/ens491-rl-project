"""
Executor unit tests (H-1).

Scripted policies on a toy state -- no SB3, no gym, no MiniGrid. The point of
these tests is that the executor is depth-agnostic and cannot hang: a depth-5
graph, interrupt semantics, goal-only masking, the empty-mask guard, bounded
re-selection, and the availability cache.
"""

import pytest

from src.skills.executor import (
    DETERMINISTIC,
    STOCHASTIC,
    Executor,
    ExecutorError,
    Frame,
)
from src.skills.node import ALWAYS, NEVER, Predicate, SkillNode
from src.skills.state import StateView
from src.types import Outcome


# ---------------------------------------------------------------------------
# Toy world: a mutable dict of flags, snapshotted into a StateView each step.
# ---------------------------------------------------------------------------

class ToyWorld:
    """A counter and a set of flags. One `step` sets whatever the action says."""

    def __init__(self, **flags):
        self.flags = dict(flags)
        self.version = 0

    def view(self) -> StateView:
        return StateView(version=self.version, extra=dict(self.flags))

    def step(self, action: int) -> None:
        self.version += 1


def flag(name):
    return Predicate(f"flag:{name}", lambda s: bool(s.extra.get(name, False)))


class FixedAction:
    """Primitive policy: always the same env action."""

    def __init__(self, action=0):
        self.action = action
        self.calls = 0
        self.deterministic_seen = []

    def act(self, obs, deterministic=True):
        self.calls += 1
        self.deterministic_seen.append(deterministic)
        return self.action


class FirstAvailable:
    """Composite policy: the lowest-index available child."""

    def __init__(self):
        self.masks_seen = []

    def select(self, obs, mask, deterministic=True):
        self.masks_seen.append(list(mask))
        return next(i for i, ok in enumerate(mask) if ok)


class ScriptedSelect:
    """Composite policy returning a fixed sequence of child indices."""

    def __init__(self, script):
        self.script = list(script)
        self.i = 0

    def select(self, obs, mask, deterministic=True):
        idx = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return idx


def prim(node_id, goal=NEVER, timeout=10, action=0):
    n = SkillNode(node_id, goal=goal, timeout=timeout)
    n.policy = FixedAction(action)
    return n


def comp(node_id, children, goal=NEVER, timeout=100, policy=None, max_decisions=64):
    n = SkillNode(node_id, children=children, goal=goal, timeout=timeout,
                  max_decisions=max_decisions)
    n.policy = policy or FirstAvailable()
    return n


def run(ex, world, max_env_steps=200):
    """Drive the executor to completion. Returns the number of env steps taken."""
    steps = 0
    while steps < max_env_steps:
        a = ex.act(None, world.view())
        if a is None:
            return steps
        world.step(a)
        ex.observe_reward(0.0)
        steps += 1
    raise AssertionError(f"did not finish within {max_env_steps} env steps")


# ---------------------------------------------------------------------------
# Depth-agnosticism
# ---------------------------------------------------------------------------

def build_depth5():
    """A chain of 5 levels: L0 primitive under L1 under ... under L4."""
    nodes = [prim("L0", goal=flag("done"), timeout=20, action=3)]
    for lvl in range(1, 5):
        nodes.append(comp(f"L{lvl}", [nodes[-1]], goal=flag("done"), timeout=20 + lvl))
    return nodes


def test_depth5_graph_runs_and_reaches_depth_5():
    nodes = build_depth5()
    root = nodes[-1]
    assert root.n == 4, "level must be derived, not assigned"

    world = ToyWorld(done=False)
    ex = Executor(root)

    a = ex.act(None, world.view())
    assert a == 3, "the primitive at the bottom of the stack supplies the action"
    assert ex.depth == 5, f"stack should hold all five frames, got {ex.depth}"
    assert [f.node.id for f in ex.stack] == ["L4", "L3", "L2", "L1", "L0"]


def test_depth5_completes_when_goal_is_reached():
    nodes = build_depth5()
    world = ToyWorld(done=False)
    ex = Executor(nodes[-1])

    ex.act(None, world.view())
    world.step(0)
    world.flags["done"] = True
    ex.observe_reward(1.0)

    assert ex.act(None, world.view()) is None
    assert ex.finished
    assert ex.last_outcome is Outcome.SUCCESS


def test_executor_never_branches_on_level():
    """Same code path for depth 1 and depth 5 -- the unbounded-by-design claim."""
    import inspect

    from src.skills import executor as ex_mod

    src = inspect.getsource(ex_mod)
    assert ".n ==" not in src and ".n >" not in src and ".n <" not in src, \
        "executor must not branch on node level"


# ---------------------------------------------------------------------------
# Termination and interrupt semantics -- v2 §4.2
# ---------------------------------------------------------------------------

def test_ancestor_termination_closes_the_whole_substack():
    leaf = prim("leaf", goal=flag("leaf_done"), timeout=50)
    mid = comp("mid", [leaf], goal=flag("mid_done"), timeout=50)
    root = comp("root", [mid], goal=flag("root_done"), timeout=50)

    world = ToyWorld()
    ex = Executor(root)
    ex.act(None, world.view())
    assert ex.depth == 3

    world.step(0)
    world.flags["root_done"] = True       # the ancestor finishes
    ex.observe_reward(0.0)
    assert ex.act(None, world.view()) is None

    closes = [e for e in ex.trace if e.event == "CLOSE"]
    by_id = {e.node_id: e.outcome for e in closes}
    assert by_id["root"] == Outcome.SUCCESS.value
    assert by_id["mid"] == Outcome.INTERRUPTED.value
    assert by_id["leaf"] == Outcome.INTERRUPTED.value


def test_highest_terminated_node_is_closed_first():
    """Top-down: when both mid and leaf are done, mid's closure subsumes leaf's."""
    both = flag("both")
    leaf = prim("leaf", goal=both, timeout=50)
    mid = comp("mid", [leaf], goal=both, timeout=50)
    root = comp("root", [mid], goal=NEVER, timeout=50)

    world = ToyWorld()
    ex = Executor(root)
    ex.act(None, world.view())

    world.step(0)
    world.flags["both"] = True
    ex.observe_reward(0.0)
    ex.act(None, world.view())

    closes = [(e.node_id, e.outcome) for e in ex.trace if e.event == "CLOSE"]
    assert ("mid", Outcome.SUCCESS.value) in closes
    assert ("leaf", Outcome.INTERRUPTED.value) in closes


def test_child_timeout_returns_control_to_parent():
    slow = prim("slow", goal=NEVER, timeout=3)
    other = prim("other", goal=NEVER, timeout=3)
    root = comp("root", [slow, other], goal=NEVER, timeout=50,
                policy=ScriptedSelect([0, 1, 0, 1, 0, 1, 0, 1]))

    world = ToyWorld()
    ex = Executor(root)
    for _ in range(4):
        assert ex.act(None, world.view()) is not None
        world.step(0)
        ex.observe_reward(0.0)

    closes = [e for e in ex.trace if e.event == "CLOSE" and e.node_id == "slow"]
    assert closes and closes[0].outcome == Outcome.TIMEOUT.value
    assert closes[0].steps == 3, "timeout fires exactly at the step budget"


def test_root_timeout_ends_the_episode():
    leaf = prim("leaf", goal=NEVER, timeout=5)
    root = comp("root", [leaf], goal=NEVER, timeout=5)
    world = ToyWorld()
    ex = Executor(root)
    steps = run(ex, world, max_env_steps=50)
    assert steps == 5
    assert ex.last_outcome is Outcome.TIMEOUT


# ---------------------------------------------------------------------------
# Availability and masking -- v2 §4.3
# ---------------------------------------------------------------------------

def test_child_whose_goal_holds_is_masked_out():
    done = prim("done_already", goal=ALWAYS, timeout=10)
    todo = prim("todo", goal=NEVER, timeout=10, action=5)
    pol = FirstAvailable()
    root = comp("root", [done, todo], goal=NEVER, timeout=50, policy=pol)

    world = ToyWorld()
    ex = Executor(root)
    assert ex.act(None, world.view()) == 5, "must pick the only live child"
    assert pol.masks_seen == [[False, True]]


def test_initiation_does_not_mask():
    """[A: F1] The whole point: a child outside its initiation set stays selectable."""
    a = SkillNode("a", goal=NEVER, initiation=NEVER, timeout=10)
    a.policy = FixedAction(2)
    pol = FirstAvailable()
    root = comp("root", [a], goal=NEVER, timeout=50, policy=pol)

    world = ToyWorld()
    ex = Executor(root)
    assert ex.act(None, world.view()) == 2
    assert pol.masks_seen == [[True]]


def test_no_empty_mask_ever_reaches_a_policy():
    """[A: F3] MaskablePPO cannot take an all-False mask."""
    seen = []

    class Recorder:
        def select(self, obs, mask, deterministic=True):
            seen.append(list(mask))
            assert any(mask), "an all-False mask reached the policy"
            return next(i for i, ok in enumerate(mask) if ok)

    c1 = prim("c1", goal=flag("x"), timeout=10)
    c2 = prim("c2", goal=flag("x"), timeout=10)
    root = comp("root", [c1, c2], goal=NEVER, timeout=50, policy=Recorder())

    world = ToyWorld(x=True)          # both children already satisfied
    ex = Executor(root)
    assert ex.act(None, world.view()) is None, "episode ends instead of masking nothing"
    assert seen == [], "policy must not be consulted at all"
    assert ex.last_outcome is Outcome.TIMEOUT


def test_composite_with_no_available_child_closes_and_parent_reselects():
    dead = comp("dead", [prim("gone", goal=ALWAYS, timeout=5)], goal=NEVER, timeout=20)
    live = prim("live", goal=NEVER, timeout=20, action=4)
    root = comp("root", [dead, live], goal=NEVER, timeout=50, policy=FirstAvailable())

    world = ToyWorld()
    ex = Executor(root)
    # `dead` is unavailable (its only child's goal holds), so the recursive second
    # term masks it and `live` is selected directly.
    assert ex.act(None, world.view()) == 4
    assert [f.node.id for f in ex.stack] == ["root", "live"]


def test_policy_choosing_a_masked_child_is_an_error():
    done = prim("done_already", goal=ALWAYS, timeout=10)
    todo = prim("todo", goal=NEVER, timeout=10)
    root = comp("root", [done, todo], goal=NEVER, timeout=50,
                policy=ScriptedSelect([0]))
    ex = Executor(root)
    with pytest.raises(ExecutorError, match="not available"):
        ex.act(None, ToyWorld().view())


# ---------------------------------------------------------------------------
# Re-selection and the decision cap -- [A: F2]
# ---------------------------------------------------------------------------

def test_timed_out_primitive_may_be_reselected_but_is_counted():
    slow = prim("slow", goal=NEVER, timeout=2)
    root = comp("root", [slow], goal=NEVER, timeout=100, max_decisions=50)

    world = ToyWorld()
    ex = Executor(root)
    for _ in range(9):
        a = ex.act(None, world.view())
        if a is None:
            break
        world.step(a)
        ex.observe_reward(0.0)

    assert ex.reselections >= 1, "re-selection of a timed-out child must be counted"
    assert ex.decisions >= 2


def test_max_decisions_terminates_the_episode():
    """Without this cap, goal-only masking lets a parent loop forever."""
    slow = prim("slow", goal=NEVER, timeout=1)
    root = comp("root", [slow], goal=NEVER, timeout=10_000, max_decisions=4)

    world = ToyWorld()
    ex = Executor(root)
    run(ex, world, max_env_steps=100)
    assert ex.decisions == 4
    assert ex.last_outcome is Outcome.TIMEOUT


def test_executor_cannot_hang():
    """
    [A: F6] A pathological graph must raise, not spin. `inner` has no children
    available and a huge timeout, so selection would thrash forever if the
    executor recursed without a bound.
    """
    class AlwaysZero:
        def select(self, obs, mask, deterministic=True):
            return 0

    leaves = [prim(f"leaf{i}", goal=ALWAYS, timeout=5) for i in range(3)]
    mids = [comp(f"mid{i}", [leaf], goal=NEVER, timeout=5000, policy=AlwaysZero())
            for i, leaf in enumerate(leaves)]
    root = comp("root", mids, goal=NEVER, timeout=5000, policy=AlwaysZero(),
                max_decisions=10_000)

    ex = Executor(root)
    # Every mid is unavailable (its leaf's goal holds), so the mask is all-False
    # at the root and the episode ends -- it must not loop.
    assert ex.act(None, ToyWorld().view()) is None


# ---------------------------------------------------------------------------
# Availability cache -- [A: F4]
# ---------------------------------------------------------------------------

def build_ladder(counting, levels=5):
    """
    A diamond ladder: every level has two composites, both pointing at both nodes
    of the next level. `bottom` is reachable by 2**levels distinct paths, so an
    unmemoized recursive available() evaluates it 2**levels times.
    """
    bottom = prim("bottom", goal=counting, timeout=10)
    lower = [bottom, bottom]
    for lvl in range(levels, 0, -1):
        kids = [lower[0]] if lower[0] is lower[1] else list(lower)
        # Deeper levels get the smaller budget: a parent must outlive its
        # children [A: F13], and lvl counts down from the bottom.
        lower = [
            comp(f"L{lvl}{side}", kids, goal=NEVER, timeout=10 + (levels - lvl))
            for side in ("a", "b")
        ]
    return comp("root", lower, goal=NEVER, timeout=100), bottom


def test_availability_does_not_blow_up_exponentially_on_a_dag():
    """
    [A: F4] Without memoization `bottom` would be evaluated 2**5 = 32 times per
    decision. Memoized it is evaluated once for availability (plus at most one
    more for its own termination check once it is on the stack).
    """
    calls = []
    counting = Predicate("counting", lambda s: calls.append(s.version) or False)
    root, _bottom = build_ladder(counting, levels=5)

    world = ToyWorld()
    ex = Executor(root)
    ex.act(None, world.view())

    assert len(calls) <= 4, (
        f"bottom evaluated {len(calls)} times in one env step; "
        "unmemoized would be ~32 -- the availability cache is not working"
    )


def test_availability_cache_is_invalidated_each_env_step():
    calls = []
    counting = Predicate("counting", lambda s: calls.append(s.version) or False)
    root, _ = build_ladder(counting, levels=3)

    world = ToyWorld()
    ex = Executor(root)
    ex.act(None, world.view())
    first = len(calls)

    world.step(0)
    ex.observe_reward(0.0)
    ex.act(None, world.view())
    assert len(calls) > first, "a new env step must not reuse the previous cache"
    assert set(calls) == {0, 1}, "each evaluation must see its own step's version"


# ---------------------------------------------------------------------------
# Execution mode and discounting
# ---------------------------------------------------------------------------

def test_execution_mode_is_passed_to_policies():
    p = prim("p", goal=NEVER, timeout=10)
    root = comp("root", [p], goal=NEVER, timeout=50)

    ex = Executor(root, execution_mode=DETERMINISTIC)
    ex.act(None, ToyWorld().view())
    assert p.policy.deterministic_seen == [True]

    p2 = prim("p2", goal=NEVER, timeout=10)
    root2 = comp("root2", [p2], goal=NEVER, timeout=50)
    ex2 = Executor(root2, execution_mode=STOCHASTIC)
    ex2.act(None, ToyWorld().view())
    assert p2.policy.deterministic_seen == [False]


def test_observe_reward_discounts_before_advancing_the_counter():
    """[A: F8] A frame pushed at step t has steps == 0 while acting at t."""
    p = prim("p", goal=NEVER, timeout=10)
    root = comp("root", [p], goal=NEVER, timeout=50)
    ex = Executor(root, gamma=0.5)
    ex.act(None, ToyWorld().view())

    ex.observe_reward(1.0)
    assert ex.stack[0].ret == pytest.approx(1.0)      # gamma**0
    assert all(f.steps == 1 for f in ex.stack)

    ex.observe_reward(1.0)
    assert ex.stack[0].ret == pytest.approx(1.5)      # + gamma**1


def test_gamma_is_per_executor_not_global():
    p = prim("p", goal=NEVER, timeout=10)
    root = comp("root", [p], goal=NEVER, timeout=50)
    a, b = Executor(root, gamma=0.9), Executor(root, gamma=0.1)
    assert a.gamma != b.gamma


# ---------------------------------------------------------------------------
# Trace -- [A: F28]
# ---------------------------------------------------------------------------

def test_trace_is_serializable_and_records_selections():
    import json

    leaf = prim("leaf", goal=flag("done"), timeout=20)
    root = comp("root", [leaf], goal=flag("done"), timeout=20)

    world = ToyWorld(done=False)
    ex = Executor(root)
    ex.act(None, world.view())
    world.step(0)
    world.flags["done"] = True
    ex.observe_reward(0.0)
    ex.act(None, world.view())

    json.dumps(ex.trace_dicts())            # must not raise
    assert ex.selected_children() == ["leaf"]
