"""
C1 controls for {1,1} unlock_door, reported together.

Two things the H-5 run surfaced that must not be quoted without context:

1. The initiation-masking ablation produced numbers identical to the main run.
   That is either a plumbing bug or a real null result, and the difference
   matters. This script measures how often the strict mask would actually
   withhold a child that the learned policy was going to pass over anyway.

2. Random choice over the available children already scores ~0.90. At this
   depth the mask and the frozen children do most of the work, and the learned
   policy's contribution shows up in efficiency rather than success rate.
   Reporting success rate alone would overstate what composition bought.

Usage:  python scripts/analyse_c1_controls.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

from src.envs.subgoal_env import NO_HOOK  # noqa: E402
from src.skills.registry import NodeRegistry  # noqa: E402
from src.skills.state import build_state_view  # noqa: E402
from src.training.evaluation import evaluate_node, heldout_seeds, run_episode  # noqa: E402
from src.training.node_trainer import (  # noqa: E402
    CompositePolicyAdapter,
    RandomOverAvailable,
    execution_env_factory,
)
from train_unlock_door import UNLOCK, build_t3, load_children  # noqa: E402

RUNS = REPO / "runs" / "v2"
N = 50


class RandomOverAll:
    """Floor: ignore the mask entirely. Only legal because the executor would
    still refuse an unavailable child -- so this is clipped to the available set
    and exists to show what the mask itself is worth."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def select(self, obs, mask, deterministic=True):
        idx = int(self.rng.integers(0, len(mask)))
        if not mask[idx]:
            idx = next(i for i, ok in enumerate(mask) if ok)
        return idx


def mask_divergence(node, seeds):
    """How often does the strict (initiation) mask differ from the goal-only one?"""
    env = execution_env_factory(UNLOCK)()
    differ = total = 0
    first_choice_differs = 0
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        sv = build_state_view(env, 0)
        ex_free, ex_strict = [], []
        version = 0
        for _ in range(node.timeout):
            goal_only = [c.available(sv, {}) for c in node.children]
            strict = [a and bool(c.initiation(sv))
                      for a, c in zip(goal_only, node.children)]
            if any(goal_only):
                total += 1
                if goal_only != strict:
                    differ += 1
                    pick_free = node.policy.select(obs, goal_only, deterministic=True)
                    if not strict[pick_free]:
                        first_choice_differs += 1
            # advance one env step under the learned stack
            from src.skills.executor import Executor
            ex = Executor(node)
            a = ex.act(obs, sv)
            if a is None:
                break
            obs, _, term, trunc, _ = env.step(a)
            version += 1
            sv = build_state_view(env, version)
            if term or trunc or node.goal(sv):
                break
    return differ, total, first_choice_differs


def main() -> int:
    registry = NodeRegistry(RUNS)
    children = load_children(registry)

    node = build_t3(children)
    from src.training.node_trainer import attach_sb3_primitive_policy  # noqa: F401
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from src.training.node_trainer import option_env_factory

    vec = DummyVecEnv([option_env_factory(node, UNLOCK, NO_HOOK, seed=0)])
    model = MaskablePPO("MlpPolicy", vec, device="cpu", verbose=0)
    node.policy = CompositePolicyAdapter(model.policy)
    registry.load_into(node, require_frozen=True)
    vec.close()
    print(f"loaded {node}\n")

    seeds = heldout_seeds(N)
    rows = []

    for label, policy, strict in [
        ("random over all children (mask clipped)", RandomOverAll(seed=0), False),
        ("random over available children", RandomOverAvailable(seed=0), False),
        ("random, initiation masking on", RandomOverAvailable(seed=0), True),
        ("learned composite", node.policy, False),
        ("learned composite, initiation masking on", node.policy, True),
    ]:
        probe = build_t3(children)
        probe.policy = policy
        res = evaluate_node(
            probe, execution_env_factory(UNLOCK), seeds=seeds,
            initiation_masking=strict,
        )
        rows.append((label, res))
        print(f"  {label:<44} SR {res.success_rate:5.2f}   mean env steps {res.mean_length:6.1f}")

    differ, total, fc = mask_divergence(node, seeds[:20])
    print(f"\nmask divergence over 20 episodes of the learned policy:")
    print(f"  decisions where the strict mask differs from goal-only: {differ}/{total}")
    print(f"  of those, where it would have blocked the learned choice: {fc}")

    out = {
        "controls": [{"label": l, **r.as_dict()} for l, r in rows],
        "mask_divergence": {
            "decisions_where_masks_differ": differ,
            "decisions_total": total,
            "would_have_blocked_learned_choice": fc,
        },
    }
    (RUNS / "c1_controls.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote runs/v2/c1_controls.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
