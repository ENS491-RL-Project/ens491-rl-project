"""
Goal and initiation predicates for the §11 curriculum.

Every predicate reads a `StateView` snapshot, never `env.unwrapped` [A: F5], and
carries a spec string so it can be hashed into a node's identity [A: F18].
"""

from __future__ import annotations

from typing import Optional

from src.skills.node import Predicate
from src.skills.state import StateView


def carrying(type_: str, color: Optional[str] = None) -> Predicate:
    """The agent holds an object of this type (T1 `get_key`, T6 `pickup_box`)."""
    spec = f"carrying:{type_}" + (f":{color}" if color else "")

    def fn(sv: StateView) -> bool:
        if sv.carrying_type != type_:
            return False
        return color is None or sv.carrying_color == color

    return Predicate(spec, fn)


def carrying_nothing() -> Predicate:
    return Predicate("carrying:none", lambda sv: sv.carrying_type is None)


def door_open(color: Optional[str] = None) -> Predicate:
    """A door is open (T2 `open_door`, T3 `unlock_door`)."""
    spec = "door:open" + (f":{color}" if color else "")

    def fn(sv: StateView) -> bool:
        d = sv.door(color)
        return d is not None and d.is_open

    return Predicate(spec, fn)


def door_locked(color: Optional[str] = None) -> Predicate:
    spec = "door:locked" + (f":{color}" if color else "")

    def fn(sv: StateView) -> bool:
        d = sv.door(color)
        return d is not None and d.is_locked

    return Predicate(spec, fn)


def on_goal_tile() -> Predicate:
    """The agent stands on the goal tile (T4 `go_to_goal`)."""

    def fn(sv: StateView) -> bool:
        return sv.goal_pos is not None and tuple(sv.agent_pos) == tuple(sv.goal_pos)

    return Predicate("agent:on_goal", fn)


def door_approach_clear(color: Optional[str] = None) -> Predicate:
    """
    No ball blocks the door (T8 `move_blocker`, BlockedUnlockPickup).

    BlockedUnlockPickup places a `Ball` in the cell immediately left of the door.
    Phrasing the predicate as "no ball adjacent to the door" rather than "cell
    (door_x - 1, door_y) is empty" keeps it readable and avoids hard-coding a
    layout detail that a different blocked variant could change.
    """
    spec = "door:approach_clear" + (f":{color}" if color else "")

    def fn(sv: StateView) -> bool:
        d = sv.door(color)
        if d is None:
            return True
        dx, dy = d.pos
        for o in sv.objects:
            if o.type != "ball":
                continue
            ox, oy = o.pos
            if abs(ox - dx) + abs(oy - dy) == 1:
                return False
        return True

    return Predicate(spec, fn)


def env_success() -> Predicate:
    """The underlying MiniGrid env signalled termination with success."""
    return Predicate("env:success", lambda sv: sv.env_terminated)
