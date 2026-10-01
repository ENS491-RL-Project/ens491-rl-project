"""
SkillNode unit tests (H-1).

Scripted predicates on synthetic StateViews -- no SB3, no gym, no MiniGrid.
"""

import pytest

from src.skills.node import ALWAYS, NEVER, Predicate, SkillNode
from src.skills.state import StateView
from src.types import NodeStatus, Outcome


def sv(version=0, **extra):
    return StateView(version=version, extra=extra)


def flag(name):
    """Predicate reading a boolean out of StateView.extra."""
    return Predicate(f"flag:{name}", lambda s: bool(s.extra.get(name, False)))


def prim(node_id, goal=NEVER, timeout=10, **kw):
    return SkillNode(node_id, goal=goal, timeout=timeout, **kw)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_level_is_derived_from_children():
    a = prim("a")
    b = prim("b")
    assert a.n == 0

    c = SkillNode("c", children=[a, b], timeout=50)
    assert c.n == 1

    d = SkillNode("d", children=[c, a], timeout=80)   # mixed-level DAG, v2 §3
    assert d.n == 2


def test_children_tuple_is_immutable():
    a = prim("a")
    c = SkillNode("c", children=[a], timeout=50)
    assert isinstance(c.children, tuple)
    with pytest.raises(AttributeError):
        c.children = ()
    with pytest.raises(TypeError):
        c.children[0] = prim("other")


def test_is_primitive():
    assert prim("a").is_primitive
    assert not SkillNode("c", children=[prim("a")], timeout=50).is_primitive


def test_node_may_have_several_parents():
    """A frozen child shared by two parents is the point of the DAG (v2 §3)."""
    shared = prim("shared")
    p1 = SkillNode("p1", children=[shared], timeout=50)
    p2 = SkillNode("p2", children=[shared, prim("other")], timeout=50)
    assert shared in p1.children and shared in p2.children


def test_cycle_is_rejected():
    """[A: F21] Acyclic-by-construction only holds if creation order is respected."""
    a = prim("a")
    c = SkillNode("c", children=[a], timeout=50)
    # Force a cycle the way a bad loader would, then build a parent over it.
    a._children = (c,)
    with pytest.raises(ValueError, match="cycle"):
        SkillNode("top", children=[c], timeout=90)


def test_timeout_invariant_is_enforced():
    """[A: F13] A parent must outlive its children."""
    child = prim("child", timeout=100)
    with pytest.raises(ValueError, match="timeout"):
        SkillNode("parent", children=[child], timeout=50)
    SkillNode("ok", children=[child], timeout=100)  # equal is allowed


def test_nonpositive_timeout_rejected():
    with pytest.raises(ValueError, match="timeout must be positive"):
        prim("a", timeout=0)


def test_at_most_one_lateral_source():
    """[A: F20] LateralMlpExtractor implements exactly one source."""
    s1, s2 = prim("s1"), prim("s2")
    prim("ok", lateral_sources=[s1])
    with pytest.raises(ValueError, match="lateral sources"):
        prim("bad", lateral_sources=[s1, s2])


def test_laterals_are_primitive_only():
    s = prim("s")
    with pytest.raises(ValueError, match="primitive"):
        SkillNode("c", children=[prim("a")], timeout=50, lateral_sources=[s])


# ---------------------------------------------------------------------------
# Availability -- v2 §4.3 [A: F1]
# ---------------------------------------------------------------------------

def test_available_is_goal_only_not_initiation():
    """
    [A: F1] initiation must NOT mask. This is the whole reason the composite in
    the curriculum has something to learn.
    """
    node = prim("a", goal=NEVER, initiation=NEVER)
    assert node.available(sv()) is True


def test_node_whose_goal_holds_is_unavailable():
    """The zero-length-option guard: selecting it would terminate instantly."""
    node = prim("a", goal=flag("done"))
    assert node.available(sv(done=False)) is True
    assert node.available(sv(done=True)) is False


def test_composite_unavailable_when_no_child_available():
    child = prim("child", goal=flag("done"))
    parent = SkillNode("parent", children=[child], goal=NEVER, timeout=50)
    assert parent.available(sv(done=False)) is True
    assert parent.available(sv(done=True)) is False   # recursive second term


def test_availability_cache_evaluates_a_shared_child_once():
    """
    [A: F4] Without memoization this is exponential in depth on a DAG. A diamond
    reaches `shared` by two paths; its goal predicate must run once per env step.
    """
    calls = []
    counting = Predicate("counting", lambda s: calls.append(s.version) or False)
    shared = prim("shared", goal=counting)

    left = SkillNode("left", children=[shared], timeout=50)
    right = SkillNode("right", children=[shared], timeout=50)
    top = SkillNode("top", children=[left, right], timeout=80)

    cache = {}
    state = sv(version=7)
    top.available(state, cache)
    assert len(calls) == 1, f"shared child evaluated {len(calls)} times, expected 1"

    top.available(state, cache)          # same version -> still cached
    assert len(calls) == 1

    top.available(sv(version=8), {})     # new step, fresh cache
    assert len(calls) == 2


def test_availability_without_cache_still_correct():
    shared = prim("shared", goal=flag("done"))
    top = SkillNode("top", children=[shared], timeout=50)
    assert top.available(sv(done=False)) is True
    assert top.available(sv(done=True)) is False


# ---------------------------------------------------------------------------
# Termination -- v2 §4.2
# ---------------------------------------------------------------------------

def test_check_termination_success_timeout_none():
    node = prim("a", goal=flag("done"), timeout=5)
    assert node.check_termination(sv(done=True), 0) is Outcome.SUCCESS
    assert node.check_termination(sv(done=False), 5) is Outcome.TIMEOUT
    assert node.check_termination(sv(done=False), 4) is None


def test_success_takes_precedence_over_timeout():
    node = prim("a", goal=ALWAYS, timeout=1)
    assert node.check_termination(sv(), 99) is Outcome.SUCCESS


# ---------------------------------------------------------------------------
# Identity -- v2 §2 [A: F18]
# ---------------------------------------------------------------------------

def test_spec_hash_is_stable_and_sensitive():
    kw = dict(goal=flag("g"), timeout=10, env_ids=["MiniGrid-Unlock-v0"])
    a = prim("a", **kw)
    assert a.spec_hash() == prim("a", **kw).spec_hash()

    assert prim("a", goal=flag("g"), timeout=11,
                env_ids=["MiniGrid-Unlock-v0"]).spec_hash() != a.spec_hash()
    assert prim("a", goal=flag("other"), timeout=10,
                env_ids=["MiniGrid-Unlock-v0"]).spec_hash() != a.spec_hash()
    assert prim("a", **{**kw, "normalize_obs": False}).spec_hash() != a.spec_hash()


def test_spec_hash_depends_on_children():
    a, b = prim("a"), prim("b")
    p1 = SkillNode("p", children=[a], timeout=50)
    p2 = SkillNode("p", children=[a, b], timeout=50)
    assert p1.spec_hash() != p2.spec_hash()


def test_predicate_needs_a_name_for_identity():
    """A bare lambda has no stable identity; Predicate carries a spec string."""
    assert Predicate("carrying:key", lambda s: True).spec == "carrying:key"


# ---------------------------------------------------------------------------
# Lifecycle -- v2 §9
# ---------------------------------------------------------------------------

def test_status_starts_declared_and_freeze_requires_a_policy():
    node = prim("a")
    assert node.status is NodeStatus.DECLARED
    assert not node.frozen
    with pytest.raises(RuntimeError, match="policy is None"):
        node.freeze()


def test_freeze_clears_grads_and_sets_status():
    import torch

    class P:
        def __init__(self):
            self._p = [torch.nn.Parameter(torch.randn(2, 2))]

        def parameters(self):
            return iter(self._p)

    node = prim("a")
    node.policy = P()
    node.freeze()
    assert node.frozen and node.status is NodeStatus.FROZEN
    assert all(not p.requires_grad for p in node.policy.parameters())
