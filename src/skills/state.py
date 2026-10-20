"""
StateView: an immutable snapshot of the environment, built once per env step.

Target Architecture v2 §4.2 [A: F5]. Predicates never read `env.unwrapped`
directly. That object is live and mutable, so a predicate, an availability mask
and the policy's observation could otherwise be evaluated at three different
instants, and an execution trace written for the GUI would hold references to a
mutating object rather than a record of what happened.

`version` increments once per env step. It is the cache key for the availability
memoization in `Executor` (v2 §4.3 [A: F4]).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

# Agent-relative direction codes, as MiniGrid defines them.
DIR_RIGHT, DIR_DOWN, DIR_LEFT, DIR_UP = 0, 1, 2, 3


@dataclass(frozen=True)
class DoorView:
    pos: tuple[int, int]
    color: str
    is_open: bool
    is_locked: bool


@dataclass(frozen=True)
class ObjView:
    pos: tuple[int, int]
    type: str
    color: str


@dataclass(frozen=True)
class StateView:
    """One immutable snapshot of the environment state."""

    agent_pos: tuple[int, int] = (0, 0)
    agent_dir: int = DIR_RIGHT
    carrying_type: Optional[str] = None
    carrying_color: Optional[str] = None
    doors: tuple[DoorView, ...] = ()
    objects: tuple[ObjView, ...] = ()
    goal_pos: Optional[tuple[int, int]] = None
    front_pos: Optional[tuple[int, int]] = None
    front_type: Optional[str] = None
    step_count: int = 0
    env_terminated: bool = False
    version: int = 0
    # Escape hatch for toy envs in tests and for env-specific fields that do not
    # belong in the MiniGrid schema above. Never read by shared predicates.
    extra: Mapping[str, Any] = field(default_factory=dict)

    def door(self, color: Optional[str] = None) -> Optional[DoorView]:
        """The single door, or the first door of `color`. None if there is none."""
        for d in self.doors:
            if color is None or d.color == color:
                return d
        return None

    def obj(self, type_: str) -> Optional[ObjView]:
        for o in self.objects:
            if o.type == type_:
                return o
        return None


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------

_SKIP_TYPES = frozenset({"wall", "empty", "unseen", "floor"})


def build_state_view(env, version: int = 0) -> StateView:
    """
    Snapshot a MiniGrid env. Works for every env in the §11 curriculum.

    `RoomGrid` envs (Unlock, UnlockPickup, BlockedUnlockPickup) expose
    `env.unwrapped.door` and `.obj` directly; `DoorKey` does not, so the grid is
    scanned instead [A: F30]. The scan is the general path and runs for every env,
    so there is no per-env branch in the predicates themselves.
    """
    u = env.unwrapped
    width, height = u.width, u.height

    doors: list[DoorView] = []
    objects: list[ObjView] = []
    goal_pos: Optional[tuple[int, int]] = None

    for x in range(width):
        for y in range(height):
            cell = u.grid.get(x, y)
            if cell is None:
                continue
            t = cell.type
            if t == "door":
                doors.append(
                    DoorView(
                        pos=(x, y),
                        color=cell.color,
                        is_open=bool(cell.is_open),
                        is_locked=bool(cell.is_locked),
                    )
                )
            elif t == "goal":
                goal_pos = (x, y)
            elif t not in _SKIP_TYPES:
                objects.append(ObjView(pos=(x, y), type=t, color=cell.color))

    carrying = getattr(u, "carrying", None)

    front = tuple(int(v) for v in u.front_pos)
    front_cell = None
    if 0 <= front[0] < width and 0 <= front[1] < height:
        front_cell = u.grid.get(*front)

    return StateView(
        agent_pos=(int(u.agent_pos[0]), int(u.agent_pos[1])),
        agent_dir=int(u.agent_dir),
        carrying_type=carrying.type if carrying is not None else None,
        carrying_color=carrying.color if carrying is not None else None,
        doors=tuple(doors),
        objects=tuple(objects),
        goal_pos=goal_pos,
        front_pos=front,
        front_type=front_cell.type if front_cell is not None else None,
        step_count=int(u.step_count),
        env_terminated=False,
        version=version,
    )
