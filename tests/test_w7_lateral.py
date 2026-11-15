"""
W7 (C-2) lateral-source plumbing: scripts/train_w7_lateral.py.

The lateral path had never run in v2. These pin the parts the experiment's
validity rests on: the source stays outside the target network, its weights do
not move, and arm C's source is a fixed function of the training seed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import train_w7_lateral as w7  # noqa: E402
from src.continual_learning.column_policy import ColumnPolicy  # noqa: E402
from src.training.node_trainer import PPO_DEFAULTS, subgoal_env_factory  # noqa: E402
from src.training.run_store import _json_value, policy_hash  # noqa: E402
from src.types import NodeStatus  # noqa: E402


def _target_model(holder, seed=1000):
    spec = w7.specs()["go_to_goal"]
    node = w7.node_trainer_build_node(spec, node_id="{0,3} go_to_goal w7-test")
    vec = DummyVecEnv([subgoal_env_factory(node, w7.DOORKEY, spec["hook"], monitor=False)])
    model = PPO(ColumnPolicy, vec, policy_kwargs={"lateral_source_column": holder},
                seed=seed, device="cpu", verbose=0, **PPO_DEFAULTS)
    return model, vec


def test_arm_c_source_is_a_fixed_function_of_the_seed():
    a, seed_a = w7.untrained_source(1000)
    b, _ = w7.untrained_source(1000)
    c, seed_c = w7.untrained_source(1100)
    assert seed_a == 901000 and seed_c == 901100
    assert policy_hash(a) == policy_hash(b)
    assert policy_hash(a) != policy_hash(c)
    assert not any(p.requires_grad for p in a.parameters())


def test_holder_is_not_a_module_and_stays_outside_the_target():
    holder, lateral_sources = w7.make_holder("C", 1000)
    assert not isinstance(holder, nn.Module)
    assert lateral_sources == ()
    before = holder.policy_sha256
    model, vec = _target_model(holder)
    try:
        pol = model.policy
        assert pol.mlp_extractor.lateral_source_column is holder
        # SB3's orthogonal init did not touch the source
        assert holder.current_hash() == before
        assert not any("lateral_source" in k for k in pol.state_dict())
        src_ids = {id(p) for p in holder.policy.parameters()}
        opt_ids = {id(p) for g in pol.optimizer.param_groups for p in g["params"]}
        assert not src_ids & opt_ids
        # one PPO update moves the target's laterals, not the source
        lat1 = pol.mlp_extractor.lat1.weight.detach().clone()
        model.learn(total_timesteps=PPO_DEFAULTS["n_steps"])
        assert not torch.equal(lat1, pol.mlp_extractor.lat1.weight)
        assert holder.current_hash() == before
    finally:
        vec.close()


def test_holder_records_its_source_in_run_json_config():
    holder, _ = w7.make_holder("C", 1000)
    rec = _json_value({"lateral_source": holder})["lateral_source"]
    assert rec == {"kind": "untrained_network", "node_id": None, "spec_hash": None,
                   "torch_seed": 901000, "policy_sha256": holder.policy_sha256}


def test_arm_b_source_is_the_frozen_open_door():
    holder, lateral_sources = w7.make_holder("B", 1000)
    (src,) = lateral_sources
    assert src.id == "{0,2} open_door" and src.status is NodeStatus.FROZEN
    assert holder.policy is src.policy.sb3_policy
    assert holder.as_dict()["spec_hash"] == src.spec_hash()
    node = w7.node_trainer_build_node(w7.specs()["go_to_goal"], node_id="x",
                                      lateral_sources=lateral_sources)
    assert node.spec()["lateral_sources"] == [src.spec_hash()]


# --------------------------------------------------------------------------
# The paired test of the plan's "Criterion" (scripts/analyse_w7_lateral.py)
# --------------------------------------------------------------------------

import analyse_w7_lateral as an  # noqa: E402


def test_wilcoxon_exact_hand_worked_tie_free():
    # |d| ranks 1, 2, 3, 4; only d = 4 is positive, so W+ = 4. Subsets of
    # {1,2,3,4} with sum <= 4: {}, {1}, {2}, {3}, {4}, {1,2}, {1,3} -> 7 of 16.
    t = an.wilcoxon_exact_lower([-1, -2, -3, 4])
    assert t["m"] == 4 and t["w_plus"] == 4 and t["p"] == 7 / 16


def test_wilcoxon_all_faster_smallest_p_and_m_below_5_cannot_pass():
    assert an.wilcoxon_exact_lower([-1, -2, -3, -4, -5])["p"] == 1 / 32
    assert an.wilcoxon_exact_lower([-1, -2, -3, -4])["p"] == 1 / 16 > an.ALPHA


def test_wilcoxon_drops_zeros_and_averages_tied_ranks():
    # zeros dropped (m = 3); |d| = 4096, 4096, 8192 -> ranks 1.5, 1.5, 3; W+ = 3.
    # Sign subsets with rank sum <= 3: {}, {1.5}, {1.5'}, {3}, {1.5, 1.5'} -> 5 of 8.
    t = an.wilcoxon_exact_lower([0, -4096, 0, -4096, 8192])
    assert t["m"] == 3 and t["w_plus"] == 3 and t["p"] == 5 / 8
    assert an.wilcoxon_exact_lower([0, 0])["p"] == 1.0


def test_beats_needs_both_p_and_ratio():
    a = {s: 200_000 for s in range(10)}
    fast = {s: 100_000 + s for s in range(10)}        # every pair faster, ratio ~0.5
    slightly = {s: 190_000 + s for s in range(10)}    # every pair faster, ratio 0.95
    assert an.beats(fast, a)["beats"]
    r = an.beats(slightly, a)
    assert r["p"] < an.ALPHA and not r["beats"]
    assert not an.beats(a, fast)["beats"]
