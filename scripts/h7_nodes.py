"""
H-7: the node specifications of the reuse experiment, in one place.

Every H-7 script (pre-flight, primitive training, the three composites, the flat
baselines, the amortisation table) builds its nodes from here, so a node's
`spec_hash` cannot drift between the script that trains it and the script that
loads it. v2 §11 (T6-T9b, amended H-7 A.1-A.3) is the reference.

    {0,4} pickup_box        primitive   UnlockPickup + BlockedUnlockPickup
    {0,5} move_blocker      primitive   BlockedUnlockPickup
    {2,2} unlock_pickup     composite   (unlock_door, pickup_box)                UnlockPickup
    {2,3} blocked_pickup_a  composite   (move_blocker, unlock_door, pickup_box)  BlockedUnlockPickup
    {3,1} blocked_pickup_b  composite   (move_blocker, unlock_pickup)            BlockedUnlockPickup
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.envs.subgoal_env import (  # noqa: E402
    NO_HOOK,
    compose,
    open_door,
    randomize_agent_pose,
    randomize_carrying,
    relocate_blocker,
)
from src.skills.node import SkillNode  # noqa: E402
from src.skills.predicates import (  # noqa: E402
    all_of,
    carrying,
    carrying_nothing,
    door_approach_blocked,
    door_approach_clear,
    door_locked,
    door_open,
)
from src.training.node_trainer import (  # noqa: E402
    load_frozen_composite,
    load_frozen_primitive,
)
from src.types import NodeStatus  # noqa: E402
from train_primitives import UNLOCK, build_node, specs  # noqa: E402
from train_unlock_door import build_t3  # noqa: E402

UNLOCKPICKUP = "MiniGrid-UnlockPickup-v0"
BLOCKED = "MiniGrid-BlockedUnlockPickup-v0"
RUNS = REPO / "runs" / "v2"

PRIMITIVE_TIMEOUT = 120
# [A: F13] T7's window is 240 (unlock_door) .. 288 (UnlockPickup's max_steps).
T7_TIMEOUT, T7_MAX_DECISIONS = 288, 8
T9_TIMEOUT = 576
T9A_MAX_DECISIONS, T9B_MAX_DECISIONS = 12, 8


def h7_primitive_specs():
    return {
        # [H-7 A.1] At the unlock_door handoff the agent holds the key (Door.toggle
        # does not consume it) and `pickup` needs an empty hand, so this node has
        # to learn drop-then-pickup. Trained only empty-handed it would fail at
        # the real handoff as go_to_goal did in H-4 (0.96 -> 0.52).
        "pickup_box": dict(
            node_id="{0,4} pickup_box", m=4,
            goal=carrying("box"),
            initiation=door_open(),
            timeout=PRIMITIVE_TIMEOUT,
            env_ids=[UNLOCKPICKUP, BLOCKED],
            hook=compose(relocate_blocker(), open_door(),
                         randomize_carrying(0.5, "key"),
                         randomize_agent_pose(anchor=None)),
        ),
        # [H-7 A.2] "door approach clear" alone is true the moment the ball is
        # picked up (a carried object is off the grid, V9), which would hand
        # get_key a full hand. The ball has to be put down and the hand empty.
        # randomize_carrying: a parent may run unlock_door first; get_key then
        # succeeds, open_door cannot, and control comes back holding the key.
        "move_blocker": dict(
            node_id="{0,5} move_blocker", m=5,
            goal=all_of(door_approach_clear(), carrying_nothing()),
            initiation=door_approach_blocked(),
            timeout=PRIMITIVE_TIMEOUT,
            env_ids=[BLOCKED],
            hook=compose(randomize_carrying(0.5, "key"),
                         randomize_agent_pose(anchor=None)),
        ),
    }


def move_blocker_fallback_spec():
    """Part B fallback: empty-handed only. A new spec string, hence a new hash."""
    s = dict(h7_primitive_specs()["move_blocker"])
    s["hook"] = randomize_agent_pose(anchor=None)
    return s


def build_unlock_pickup(unlock_door, pickup_box):
    return SkillNode(
        "{2,2} unlock_pickup",
        children=(unlock_door, pickup_box),
        goal=carrying("box"),
        initiation=door_locked(),
        timeout=T7_TIMEOUT,
        max_decisions=T7_MAX_DECISIONS,
        env_ids=[UNLOCKPICKUP],
        m=2,
    )


def build_blocked_pickup(variant: str, children):
    if variant == "a":
        node_id, m, md = "{2,3} blocked_pickup_a", 3, T9A_MAX_DECISIONS
    elif variant == "b":
        node_id, m, md = "{3,1} blocked_pickup_b", 1, T9B_MAX_DECISIONS
    else:
        raise ValueError(variant)
    return SkillNode(
        node_id,
        children=tuple(children),
        goal=carrying("box"),
        initiation=all_of(door_locked(), door_approach_blocked()),
        timeout=T9_TIMEOUT,
        max_decisions=md,
        env_ids=[BLOCKED],
        m=m,
    )


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_unlock_graph(registry, device="cpu", verbose=True):
    """`{0,1} get_key`, `{0,2} open_door` and `{1,1} unlock_door`, all FROZEN."""
    all_specs = specs()
    prims = {}
    for name in ("get_key", "open_door"):
        spec = all_specs[name]
        node = build_node(spec)
        load_frozen_primitive(node, registry, spec["env_ids"][0], spec["hook"], device=device)
        prims[name] = node
    t3 = build_t3([prims["get_key"], prims["open_door"]])
    load_frozen_composite(t3, registry, UNLOCK, NO_HOOK, device=device)
    for n in (*prims.values(), t3):
        if n.status is not NodeStatus.FROZEN:
            raise RuntimeError(f"{n.id} did not load as FROZEN")
        if verbose:
            print(f"  loaded {n}  spec_hash={n.spec_hash()}")
    return prims, t3


def load_h7_primitive(name, registry, device="cpu", spec=None, verbose=True):
    spec = spec or h7_primitive_specs()[name]
    node = build_node(spec)
    load_frozen_primitive(node, registry, spec["env_ids"][0], spec["hook"], device=device)
    if node.status is not NodeStatus.FROZEN:
        raise RuntimeError(f"{node.id} did not load as FROZEN")
    if verbose:
        print(f"  loaded {node}  spec_hash={node.spec_hash()}")
    return node
