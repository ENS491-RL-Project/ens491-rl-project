"""
NodeRegistry unit tests (H-1).

The property that matters: a checkpoint knows what spec it was trained under,
and loading it into a node with a different spec fails loudly [A: F18].
"""

import json

import pytest
import torch

from src.skills.node import NEVER, Predicate, SkillNode
from src.skills.registry import NodeRegistry, SpecMismatch
from src.types import NodeStatus


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)


def make_node(node_id="{0,1} get_key", timeout=120, goal_spec="carrying:key"):
    n = SkillNode(
        node_id,
        goal=Predicate(goal_spec, lambda sv: False),
        timeout=timeout,
        env_ids=["MiniGrid-Unlock-v0"],
    )
    n.policy = TinyPolicy()
    return n


def test_save_writes_three_artifacts_and_a_full_meta(tmp_path):
    node = make_node()
    node.freeze()
    d = NodeRegistry(tmp_path).save(node, extra_meta={"seed": 42, "success_rate": 0.94})

    assert (d / "policy_state_dict.pt").exists()
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))

    # The v1 checkpoints recorded {"n", "m", "frozen"} and nothing else.
    for key in ("id", "n", "m", "status", "spec_hash", "spec", "seed", "success_rate"):
        assert key in meta, f"meta.json is missing {key}"
    assert meta["status"] == "frozen"
    assert meta["spec"]["timeout"] == 120
    assert meta["spec"]["goal"] == "carrying:key"
    assert meta["spec"]["env_ids"] == ["MiniGrid-Unlock-v0"]


def test_round_trip_restores_weights(tmp_path):
    reg = NodeRegistry(tmp_path)
    src = make_node()
    src.freeze()
    reg.save(src)

    dst = make_node()
    before = dst.policy.fc.weight.clone()
    reg.load_into(dst)

    assert not torch.allclose(before, dst.policy.fc.weight), "weights were not loaded"
    assert torch.allclose(src.policy.fc.weight, dst.policy.fc.weight)
    assert dst.status is NodeStatus.FROZEN
    assert all(not p.requires_grad for p in dst.policy.parameters())


def test_changed_timeout_is_refused(tmp_path):
    reg = NodeRegistry(tmp_path)
    src = make_node(timeout=120)
    src.freeze()
    reg.save(src)

    dst = make_node(timeout=60)                      # same id, different spec
    with pytest.raises(SpecMismatch, match="spec_hash"):
        reg.load_into(dst, run_dir=reg.node_dir(src.id))


def test_changed_goal_predicate_is_refused(tmp_path):
    reg = NodeRegistry(tmp_path)
    src = make_node(goal_spec="carrying:key")
    src.freeze()
    reg.save(src)

    dst = make_node(goal_spec="door:open")
    with pytest.raises(SpecMismatch):
        reg.load_into(dst, run_dir=reg.node_dir(src.id))


def test_unfrozen_checkpoint_is_refused_by_default(tmp_path):
    """Only FROZEN nodes may be children or roots (v2 §9)."""
    reg = NodeRegistry(tmp_path)
    src = make_node()
    src.status = NodeStatus.STABLE
    reg.save(src)

    dst = make_node()
    with pytest.raises(RuntimeError, match="not frozen"):
        reg.load_into(dst)
    reg.load_into(dst, require_frozen=False)         # explicit opt-in works


def test_composite_hash_covers_its_children(tmp_path):
    reg = NodeRegistry(tmp_path)
    a, b = make_node("{0,1} a"), make_node("{0,2} b")

    parent = SkillNode("{1,1} p", children=[a], goal=NEVER, timeout=300)
    parent.policy = TinyPolicy()
    parent.freeze()
    reg.save(parent)

    # Same parent spec, but rebuilt over a different child set.
    other = SkillNode("{1,1} p", children=[a, b], goal=NEVER, timeout=300)
    other.policy = TinyPolicy()
    with pytest.raises(SpecMismatch):
        reg.load_into(other)


def test_list_nodes(tmp_path):
    reg = NodeRegistry(tmp_path)
    for nid in ("{0,1} get_key", "{0,2} open_door"):
        n = make_node(nid)
        n.freeze()
        reg.save(n)
    assert reg.list_nodes() == ["0-1_get_key", "0-2_open_door"]
