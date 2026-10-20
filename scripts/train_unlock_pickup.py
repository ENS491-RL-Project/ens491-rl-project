"""
H-7a.3 (plan H-7.4): train and freeze T7 `{2,2} unlock_pickup` =
composite(unlock_door, drop_held, pickup_box) (H-7 amendment 1).

The first task that reuses a frozen composite in a new env: `{1,1} unlock_door`
was trained on Unlock and runs here untouched (A1: 0.98 from episode start on
UnlockPickup). `{0,9} drop_held` and `{0,4} pickup_box` are the new primitives.

Under the goal-only mask `drop_held` is unavailable while the hand is empty (its
goal already holds) and `pickup_box` stays selectable with a full hand, where it
times out; the selector has to learn unlock -> drop -> pick up from that. The
strict mask gives the order for free, so the strict-mask random control is
expected to tie the learned composite on success rate (plan A3). That tie is the
curriculum's shape, not a finding, and it is reported in the same sentence as the
score; `selection_sequences` is the mechanism measure beside it.

`chain_ceiling` = A1's row x A2a's row x A2b's row (amendment 1, H-7a.3).

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
    RUNS,
    UNLOCKPICKUP,
    build_unlock_pickup,
    load_h7_primitive,
    load_unlock_graph,
)
from src.skills.registry import NodeRegistry  # noqa: E402

A1_ROW = ("base", "episode start (A1)")
A2A_ROW = ("pickup_box", "drop_held after unlock_door (A2a)")
A2B_ROW = ("pickup_box", "pickup_box after unlock_door, drop_held (A2b)")


def ceiling() -> dict:
    a1, a2a, a2b = f29(*A1_ROW), f29(*A2A_ROW), f29(*A2B_ROW)
    return {"path": ["{1,1} unlock_door", "{0,9} drop_held", "{0,4} pickup_box"],
            "rows": {"unlock_door from episode start (A1)": a1,
                     "drop_held after unlock_door (A2a)": a2a,
                     "pickup_box after unlock_door, drop_held (A2b)": a2b},
            "value": round(a1 * a2a * a2b, 4)}


def main() -> int:
    args = common_args(argparse.ArgumentParser()).parse_args()
    registry = NodeRegistry(RUNS)
    print("loading the frozen children:")
    _, unlock_door = load_unlock_graph(registry, device=args.device)
    drop_held = load_h7_primitive("drop_held", registry, device=args.device)
    pickup_box = load_h7_primitive("pickup_box", registry, device=args.device)

    run(args, lambda: build_unlock_pickup(unlock_door, drop_held, pickup_box), UNLOCKPICKUP,
        chain_ceiling=ceiling(), default_name=f"unlock_pickup_s{args.seed}", expected_depth=3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
