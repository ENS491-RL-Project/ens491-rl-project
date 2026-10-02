"""
H-7.6: T9a `{2,3} blocked_pickup_a` and T9b `{3,1} blocked_pickup_b` on BlockedUnlockPickup.

    --variant a   children (move_blocker, unlock_door, pickup_box)   level 2
    --variant b   children (move_blocker, unlock_pickup)              level 3

Same skills, same env, same pretraining cost on both sides: the depth-reuse
comparison R4 (plan A7). T9b's stack reaches depth 4 in one episode
(`blocked_pickup_b -> unlock_pickup -> unlock_door -> get_key`, plan A6).

If H-7.5 took the rebuild branch, `--rebuilt` uses `{1,3} unlock_door_b` and
`{2,4} unlock_pickup_b` instead of `{1,1}` and `{2,2}`.

    python scripts/train_blocked_pickup.py --variant b --smoke
    python scripts/train_blocked_pickup.py --variant b --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from h7_composite import common_args, f29, run  # noqa: E402
from h7_nodes import (  # noqa: E402
    BLOCKED,
    RUNS,
    UNLOCKPICKUP,
    build_blocked_pickup,
    build_unlock_pickup,
    load_h7_primitive,
    load_unlock_graph,
)
from src.envs.subgoal_env import NO_HOOK  # noqa: E402
from src.skills.registry import NodeRegistry  # noqa: E402
from src.training.node_trainer import load_frozen_composite  # noqa: E402


def load_library(registry, rebuilt: bool, device="cpu"):
    move_blocker = load_h7_primitive("move_blocker", registry, device=device)
    pickup_box = load_h7_primitive("pickup_box", registry, device=device)
    if rebuilt:
        from h7_rebuild import load_rebuilt
        unlock_door, unlock_pickup = load_rebuilt(registry, pickup_box, device=device)
    else:
        _, unlock_door = load_unlock_graph(registry, device=device)
        unlock_pickup = build_unlock_pickup(unlock_door, pickup_box)
        load_frozen_composite(unlock_pickup, registry, UNLOCKPICKUP, NO_HOOK, device=device)
        print(f"  loaded {unlock_pickup}  spec_hash={unlock_pickup.spec_hash()}")
    return move_blocker, unlock_door, pickup_box, unlock_pickup


def main() -> int:
    ap = common_args(argparse.ArgumentParser())
    ap.add_argument("--variant", choices=["a", "b"], required=True)
    ap.add_argument("--rebuilt", action="store_true",
                    help="use the H-7.5 rebuild-branch nodes (_b) for unlock_door / unlock_pickup")
    args = ap.parse_args()

    registry = NodeRegistry(RUNS)
    print("loading the frozen children:")
    mb, ud, pb, up = load_library(registry, args.rebuilt, device=args.device)

    rows = {"move_blocker from episode start": f29("blocked", "move_blocker from episode start")}
    if args.variant == "a":
        children = (mb, ud, pb)
        rows["unlock_door after move_blocker"] = f29("blocked", "A5: unlock_door after move_blocker"
                                                     if not args.rebuilt else
                                                     "unlock_door_b after move_blocker")
        rows["pickup_box after move_blocker, unlock_door"] = f29(
            "blocked", "pickup_box after move_blocker, unlock_door" if not args.rebuilt else
            "pickup_box after move_blocker, unlock_door_b")
        name, depth = f"blocked_pickup_a_s{args.seed}", 3
    else:
        children = (mb, up)
        rows["unlock_pickup after move_blocker"] = f29("blocked", "A5: unlock_pickup after move_blocker"
                                                       if not args.rebuilt else
                                                       "unlock_pickup_b after move_blocker")
        name, depth = f"blocked_pickup_b_s{args.seed}", 4
    value = 1.0
    for v in rows.values():
        value *= v
    ceiling = {"path": [c.id for c in children], "rows": rows, "value": round(value, 4),
               "reuse": "rebuilt" if args.rebuilt else "transferred"}

    out = run(args, lambda: build_blocked_pickup(args.variant, children), BLOCKED,
              chain_ceiling=ceiling, default_name=name, expected_depth=depth)
    if args.variant == "b" and not args.smoke:
        import json
        (RUNS / "blocked_pickup_b_trace.json").write_text(
            json.dumps(out["trace"], indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
