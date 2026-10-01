"""
H-6.0: the F29 pre-flight for the depth-2 stack, before `{2,1} solve_doorkey` trains.

[A: F29] A node trained in one env only ever sees the states that env and its
reset hook produce. `{1,1} unlock_door` was trained on `MiniGrid-Unlock-v0`, an
11x6 two-room grid; `{2,1} solve_doorkey` will run it inside DoorKey-6x6, a 6x6
grid with a different wall layout. And `{0,3} go_to_goal` will be handed whatever
state `unlock_door` leaves behind, which is not the state its own reset hook
produces. H-4 paid for learning this the expensive way: `go_to_goal` scored 0.96
on its hook distribution and 0.52 from the real handoff.

So the three transfers the depth-2 stack depends on are measured here, cheaply,
before any budget is committed to training the parent:

  1. unlock_door on Unlock       -- a regression against its H-5 score, which
                                    also proves the checkpoint reloads as frozen
  2. unlock_door on DoorKey-6x6  -- the layout transfer
  3. go_to_goal after unlock_door ran to completion on DoorKey-6x6 -- the handoff

A fourth row is reported but does not gate: `go_to_goal` from episode start with
the door still shut. Its goal is unreachable there, so timing out is correct, and
that failure is the signal the parent learns from under goal-only masking.

Usage:  python scripts/eval_h6_preflight.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.envs.subgoal_env import NO_HOOK  # noqa: E402
from src.skills.executor import Executor  # noqa: E402
from src.skills.registry import NodeRegistry  # noqa: E402
from src.skills.state import build_state_view  # noqa: E402
from src.training.device import describe  # noqa: E402
from src.training.evaluation import evaluate_node, heldout_seeds  # noqa: E402
from src.training.node_trainer import (  # noqa: E402
    execution_env_factory,
    load_frozen_composite,
    load_frozen_primitive,
)
from train_primitives import DOORKEY, UNLOCK, build_node, specs  # noqa: E402
from train_unlock_door import build_t3  # noqa: E402

RUNS = REPO / "runs" / "v2"
N_EVAL = 50
GATE = 0.90

# H-5's recorded held-out score for `{1,1} unlock_door`. Row 1 is a regression
# against this number, not a free-standing measurement.
H5_UNLOCK_DOOR_SR = 0.98


def load_all(registry, device="cpu"):
    """Children first, then the parent: `spec_hash` recurses [A: F18]."""
    all_specs = specs()
    prims = {}
    for name in ("get_key", "open_door", "go_to_goal"):
        spec = all_specs[name]
        node = build_node(spec)
        load_frozen_primitive(node, registry, spec["env_ids"][0], spec["hook"],
                              device=device)
        prims[name] = node
        print(f"  loaded {node}  spec_hash={node.spec_hash()}")

    t3 = build_t3([prims["get_key"], prims["open_door"]])
    load_frozen_composite(t3, registry, UNLOCK, NO_HOOK, device=device)
    print(f"  loaded {t3}  spec_hash={t3.spec_hash()}")
    return prims, t3


def roll(node, env, obs):
    """Run one frozen node to its own termination; return the obs it leaves."""
    version = 0
    sv = build_state_view(env, version)
    ex = Executor(node)
    for _ in range(node.timeout):
        action = ex.act(obs, sv)
        if action is None:
            break
        obs, _, term, trunc, _ = env.step(action)
        version += 1
        sv = build_state_view(env, version)
        ex.observe_reward(0.0)
        if term or trunc:
            break
    return obs


def main() -> int:
    print(describe())
    registry = NodeRegistry(RUNS)
    print("loading the frozen H-5 graph:")
    prims, t3 = load_all(registry)

    cases = [
        (t3, UNLOCK, None, "episode start (H-5 regression)", True),
        (t3, DOORKEY, None, "episode start (layout transfer)", True),
        (prims["go_to_goal"], DOORKEY, lambda env, obs: roll(t3, env, obs),
         "after unlock_door ran (the handoff)", True),
        (prims["go_to_goal"], DOORKEY, None,
         "[expected fail] episode start, door shut", False),
    ]

    print(f"\n{'node':<22}{'env':<18}{'start state':<40}{'SR':>6}{'LB':>7}{'len':>7}   gate")
    print("-" * 104)
    results = []
    for node, env_id, prepare, label, gate in cases:
        res = evaluate_node(
            node, execution_env_factory(env_id, NO_HOOK),
            seeds=heldout_seeds(N_EVAL), prepare=prepare,
        )
        print(f"{node.id:<22}{env_id.replace('MiniGrid-', ''):<18}{label:<40}"
              f"{res.success_rate:>6.2f}{res.wilson_lower:>7.3f}{res.mean_length:>7.1f}"
              f"   {'GATE' if gate else '-'}")
        results.append({
            "node": node.id, "env": env_id, "start_state": label, "gate": gate,
            **res.as_dict(),
        })

    out = {
        "n_eval": N_EVAL,
        "seed_base": heldout_seeds(1)[0],
        "gate_threshold": GATE,
        "h5_recorded_unlock_door_sr": H5_UNLOCK_DOOR_SR,
        "rows": results,
    }
    path = RUNS / "h6_preflight.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path.relative_to(REPO)}")

    failed = [r for r in results if r["gate"] and r["success_rate"] < GATE]
    regression = results[0]["success_rate"]
    if abs(regression - H5_UNLOCK_DOOR_SR) > 1e-9:
        print(f"\nNOTE: unlock_door scores {regression:.2f} on Unlock, H-5 recorded "
              f"{H5_UNLOCK_DOOR_SR:.2f}. Frozen weights cannot change, so a difference "
              f"is a bug in the loading path, not drift.")
    if failed:
        print("\nGATE FAILED -- below 0.90 from a state the depth-2 stack produces:")
        for r in failed:
            print(f"  {r['node']} on {r['env']} from '{r['start_state']}': "
                  f"SR {r['success_rate']:.2f}")
        print("H-6 is blocked until the primitive is retrained on a wider mix "
              "(plan H-6.0, risk row 1).")
        return 1

    print("\nGATE PASSED: every transfer the depth-2 stack depends on clears 0.90.")
    for r in results:
        if not r["gate"]:
            print(f"  (non-gating) {r['node']} '{r['start_state']}': "
                  f"SR {r['success_rate']:.2f} -- failing here is correct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
