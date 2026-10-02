"""
H-7.3 follow-up: where does the unstable `{0,4} pickup_box` (seed 42, saved as
TRAINING, never frozen) succeed and fail?

A2 failed -- no seed stable within 600,000 env steps -- and the plan stops there.
This is not a remedy; it splits the held-out score of the seed-42 checkpoint by
env and by what the agent holds at the start, plus the real handoff (after
`{1,1} unlock_door` ran on UnlockPickup), so the decision on what to change
starts from a measurement.

    python scripts/diagnose_pickup_box.py           ->  runs/v2/pickup_box_diagnosis.json
    python scripts/diagnose_pickup_box.py --spec2   ->  runs/v2/pickup_box_spec2_diagnosis.json

[H-7 amendment 1] The attempt-1 spec was withdrawn and its checkpoint moved to
`runs/v2/attempts/0-4_pickup_box_attempt1/`; it is loaded from there with the
withdrawn spec, so this diagnosis still reproduces.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

from src.envs.subgoal_env import (  # noqa: E402
    NO_HOOK,
    ResetHook,
    UnsuitableLayout,
    compose,
    give_key,
    key_off_door,
    open_door,
    reachable_cells,
    relocate_blocker,
)
from src.skills.registry import NodeRegistry  # noqa: E402
from src.training.evaluation import evaluate_node, heldout_seeds  # noqa: E402
from src.training.node_trainer import attach_sb3_primitive_policy, execution_env_factory  # noqa: E402
from eval_h7_preflight import chain  # noqa: E402
from h7_nodes import (  # noqa: E402
    ATTEMPT1_DIR,
    BLOCKED,
    RUNS,
    UNLOCKPICKUP,
    load_unlock_graph,
    pickup_box_attempt1_spec,
)
from train_primitives import build_node  # noqa: E402

N = 50


def main() -> int:
    registry = NodeRegistry(RUNS)
    spec = pickup_box_attempt1_spec()
    node = build_node(spec)
    attach_sb3_primitive_policy(node, UNLOCKPICKUP, spec["hook"])
    registry.load_into(node, run_dir=ATTEMPT1_DIR, require_frozen=False)   # TRAINING, not frozen
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


def pose_in_room(side: str) -> ResetHook:
    """Probe only: a random free pose on one side of the door ("left" = the parent's side)."""

    def fn(env):
        u = env.unwrapped
        door_pos = next((x, y) for x in range(u.width) for y in range(u.height)
                        if u.grid.get(x, y) is not None and u.grid.get(x, y).type == "door")
        start = (int(u.agent_pos[0]), int(u.agent_pos[1]))
        cells = [c for c in reachable_cells(u, start) if u.grid.get(*c) is None
                 and ((c[0] < door_pos[0]) if side == "left" else (c[0] > door_pos[0]))]
        if not cells:      # the box stands right behind the door: no right-room pose
            raise UnsuitableLayout(f"probe_pose_in_room:{side}: no free cell")
        c = cells[u._rand_int(0, len(cells))]
        u.agent_pos = np.array(c)
        u.agent_dir = u._rand_int(0, 4)

    return ResetHook(f"probe_pose_in_room:{side}", fn)


def main_spec2() -> int:
    """
    [H-7a.2, after the A2b stop] The same question for amendment 1's pickup_box:
    the seed-42 checkpoint (TRAINING, not stable at 1,000,000) under its own
    reset distribution, split by the room the agent starts in, against the real
    handoff. Writes `runs/v2/pickup_box_spec2_diagnosis.json`; the attempt-1 file
    is left as it is.
    """
    from h7_nodes import h7_primitive_specs, load_h7_primitive

    registry = NodeRegistry(RUNS)
    spec = h7_primitive_specs()["pickup_box"]
    node = build_node(spec)
    attach_sb3_primitive_policy(node, UNLOCKPICKUP, spec["hook"])
    registry.load_into(node, require_frozen=False)                       # TRAINING, not frozen
    _, t3 = load_unlock_graph(registry, verbose=False)
    dh = load_h7_primitive("drop_held", registry, verbose=False)

    base = [open_door(), key_off_door()]
    cases = [
        (spec["hook"], None, "own hook (pose over both rooms)"),
        (compose(*base, pose_in_room("left")), None, "own hook, pose in the LEFT room (parent's side)"),
        (compose(*base, pose_in_room("right")), None, "own hook, pose in the RIGHT room (box's side)"),
        (NO_HOOK, chain(t3, dh), "after unlock_door, drop_held (the real handoff)"),
    ]
    rows = []
    for hook, prep, label in cases:
        r = evaluate_node(node, execution_env_factory(UNLOCKPICKUP, hook),
                          seeds=heldout_seeds(N), prepare=prep)
        rows.append({"env": UNLOCKPICKUP, "start_state": label, "hook": hook.spec, **r.as_dict()})
        print(f"  {label:<52}SR {r.success_rate:.2f} (LB {r.wilson_lower:.3f})"
              f"  mean length {r.mean_length:.1f}")
    out = {"node": node.id, "spec": "H-7 amendment 1",
           "checkpoint": "seed 42, TRAINING (not stable at 1,001,472)",
           "spec_hash": node.spec_hash(), "rows": rows,
           "note": "diagnostic; a reading of one checkpoint, not a tested cause"}
    path = RUNS / "pickup_box_spec2_diagnosis.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_spec2() if "--spec2" in sys.argv else main())
