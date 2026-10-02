"""
H-7.3 follow-up: where does the unstable `{0,4} pickup_box` (seed 42, saved as
TRAINING, never frozen) succeed and fail?

A2 failed -- no seed stable within 600,000 env steps -- and the plan stops there.
This is not a remedy; it splits the held-out score of the seed-42 checkpoint by
env and by what the agent holds at the start, plus the real handoff (after
`{1,1} unlock_door` ran on UnlockPickup), so the decision on what to change
starts from a measurement.

    python scripts/diagnose_pickup_box.py   ->  runs/v2/pickup_box_diagnosis.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.envs.subgoal_env import NO_HOOK, compose, give_key, open_door, relocate_blocker  # noqa: E402
from src.skills.registry import NodeRegistry  # noqa: E402
from src.training.evaluation import evaluate_node, heldout_seeds  # noqa: E402
from src.training.node_trainer import attach_sb3_primitive_policy, execution_env_factory  # noqa: E402
from eval_h7_preflight import chain  # noqa: E402
from h7_nodes import BLOCKED, RUNS, UNLOCKPICKUP, h7_primitive_specs, load_unlock_graph  # noqa: E402
from train_primitives import build_node  # noqa: E402

N = 50


def main() -> int:
    registry = NodeRegistry(RUNS)
    spec = h7_primitive_specs()["pickup_box"]
    node = build_node(spec)
    attach_sb3_primitive_policy(node, UNLOCKPICKUP, spec["hook"])
    registry.load_into(node, require_frozen=False)       # TRAINING, not frozen
    _, t3 = load_unlock_graph(registry, verbose=False)

    cases = []
    for env_id in (UNLOCKPICKUP, BLOCKED):
        for hand, extra in (("nothing", []), ("key", [give_key()])):
            hook = compose(relocate_blocker(), open_door(), *extra)
            cases.append((env_id, hook, None, f"door open, holding {hand}"))
    cases.append((UNLOCKPICKUP, NO_HOOK, chain(t3), "after unlock_door ran (the real handoff)"))

    rows = []
    for env_id, hook, prep, label in cases:
        r = evaluate_node(node, execution_env_factory(env_id, hook),
                          seeds=heldout_seeds(N), prepare=prep)
        rows.append({"env": env_id, "start_state": label, "hook": hook.spec, **r.as_dict()})
        print(f"  {env_id.replace('MiniGrid-', ''):<26}{label:<42}SR {r.success_rate:.2f}"
              f"  mean length {r.mean_length:.1f}")
    out = {"node": node.id, "checkpoint": "seed 42, TRAINING (not stable at 600,000)",
           "spec_hash": node.spec_hash(), "rows": rows}
    path = RUNS / "pickup_box_diagnosis.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
