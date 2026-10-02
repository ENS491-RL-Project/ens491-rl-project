"""
H-7.4: train and freeze T7 `{2,2} unlock_pickup` = composite(unlock_door, pickup_box).

The first task that reuses a frozen composite in a new env: `{1,1} unlock_door`
was trained on Unlock and runs here untouched (A1: 0.98 from episode start on
UnlockPickup). `{0,4} pickup_box` is the only new primitive.

The subgoal order is strict (door, then box), so the strict-mask random control
is expected to tie the learned composite on success rate (plan A3). That tie is
the curriculum's shape, not a finding, and it is reported in the same sentence as
the score; `selection_sequences` is the mechanism measure beside it.

    python scripts/train_unlock_pickup.py --smoke
    python scripts/train_unlock_pickup.py --seed 42
    python scripts/train_unlock_pickup.py --seed 43 --no-freeze
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
    UNLOCKPICKUP,
    build_unlock_pickup,
    load_h7_primitive,
    load_unlock_graph,
)
from src.skills.registry import NodeRegistry  # noqa: E402
from h7_nodes import RUNS  # noqa: E402


def main() -> int:
    args = common_args(argparse.ArgumentParser()).parse_args()
    registry = NodeRegistry(RUNS)
    print("loading the frozen children:")
    _, unlock_door = load_unlock_graph(registry, device=args.device)
    pickup_box = load_h7_primitive("pickup_box", registry, device=args.device)

    a1 = f29("base", "episode start (A1)")
    pb = f29("pickup_box", "after unlock_door ran (the handoff)")
    ceiling = {"path": ["{1,1} unlock_door", "{0,4} pickup_box"],
               "rows": {"unlock_door from episode start (A1)": a1,
                        "pickup_box after unlock_door": pb},
               "value": round(a1 * pb, 4)}
    run(args, lambda: build_unlock_pickup(unlock_door, pickup_box), UNLOCKPICKUP,
        chain_ceiling=ceiling, default_name=f"unlock_pickup_s{args.seed}", expected_depth=3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
