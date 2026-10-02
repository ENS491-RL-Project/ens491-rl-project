"""
H-7.1: the blocker hook and predicate, against real minigrid 3.1.0.

`relocate_blocker()` reproduces the state `{0,5} move_blocker` leaves -- the ball
still in the room, away from the door -- and `door_approach_blocked()` is
`move_blocker`'s initiation. The test that matters most is the V9 regression:
`door_approach_clear()` reads the grid, a carried ball is not on the grid, so
"clear" alone becomes true the moment the ball is picked up. `move_blocker`'s
goal must therefore stay false until the ball is put down and the hand is empty.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from src.envs.option_env import ResetHookEnv
from src.envs.subgoal_env import (
    UnsuitableLayout,
    compose,
    open_door,
    randomize_agent_pose,
    randomize_carrying,
    reachable_cells,
    relocate_blocker,
)
from src.skills.predicates import (
    all_of,
    carrying_nothing,
    door_approach_blocked,
    door_approach_clear,
)
from src.skills.state import StateView, build_state_view

UNLOCKPICKUP = "MiniGrid-UnlockPickup-v0"
BLOCKED = "MiniGrid-BlockedUnlockPickup-v0"
N = 400

PICKUP_BOX_HOOK = compose(relocate_blocker(), open_door(),
                          randomize_carrying(0.5, "key"),
                          randomize_agent_pose(anchor=None))
MOVE_BLOCKER_GOAL = all_of(door_approach_clear(), carrying_nothing())

# minigrid actions
LEFT, RIGHT, FORWARD, PICKUP, DROP = 0, 1, 2, 3, 4


def _adjacent(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _env(env_id, hook):
    return ResetHookEnv(gym.make(env_id), hook)


# -- door_approach_blocked ---------------------------------------------------

def test_door_approach_blocked_holds_at_every_blocked_reset():
    env = gym.make(BLOCKED)
    try:
        for seed in range(100):
            env.reset(seed=seed)
            sv = build_state_view(env)
            assert door_approach_blocked()(sv)
            assert not door_approach_clear()(sv)
    finally:
        env.close()


def test_door_approach_blocked_is_false_without_a_ball_or_a_door():
    env = gym.make(UNLOCKPICKUP)
    try:
        for seed in range(50):
            env.reset(seed=seed)
            assert not door_approach_blocked()(build_state_view(env))
    finally:
        env.close()
    no_door = StateView()
    assert not door_approach_blocked()(no_door)
    assert door_approach_clear()(no_door)


def test_door_approach_blocked_spec_differs_from_clear():
    assert door_approach_blocked().spec == "door:approach_blocked"
    assert door_approach_blocked("yellow").spec == "door:approach_blocked:yellow"
    assert door_approach_blocked().spec != door_approach_clear().spec


# -- relocate_blocker ----------------------------------------------------------

def test_relocate_blocker_never_leaves_the_ball_next_to_the_door():
    env = _env(BLOCKED, relocate_blocker())
    try:
        for seed in range(N):
            env.reset(seed=seed)
            sv = build_state_view(env)
            balls = [o for o in sv.objects if o.type == "ball"]
            assert len(balls) == 1, seed           # moved, not deleted
            assert not _adjacent(balls[0].pos, sv.door().pos), seed
            assert door_approach_clear()(sv)
            assert sv.door().is_locked
    finally:
        env.close()


def test_relocate_blocker_keeps_key_and_door_reachable():
    env = _env(BLOCKED, relocate_blocker())
    try:
        for seed in range(N):
            env.reset(seed=seed)
            u = env.unwrapped
            sv = build_state_view(env)
            start = tuple(int(v) for v in u.agent_pos)
            reach = set(reachable_cells(u, start)) | {start}
            key = sv.obj("key").pos
            door = sv.door().pos
            assert any(_adjacent(c, key) for c in reach), seed
            assert any(_adjacent(c, door) for c in reach), seed
    finally:
        env.close()


def test_relocate_blocker_is_a_no_op_without_a_ball():
    plain, hooked = gym.make(UNLOCKPICKUP), _env(UNLOCKPICKUP, relocate_blocker())
    try:
        for seed in range(50):
            plain.reset(seed=seed)
            hooked.reset(seed=seed)
            a, b = build_state_view(plain), build_state_view(hooked)
            assert a.objects == b.objects and a.doors == b.doors
            assert a.agent_pos == b.agent_pos and a.agent_dir == b.agent_dir
    finally:
        plain.close()
        hooked.close()


def test_relocate_blocker_is_reproducible_from_the_seed():
    e1, e2 = _env(BLOCKED, relocate_blocker()), _env(BLOCKED, relocate_blocker())
    try:
        for seed in (0, 7, 123):
            e1.reset(seed=seed)
            e2.reset(seed=seed)
            assert build_state_view(e1).objects == build_state_view(e2).objects
    finally:
        e1.close()
        e2.close()


def test_relocate_blocker_raises_when_no_cell_is_acceptable():
    env = gym.make(BLOCKED)
    env.reset(seed=0)
    u = env.unwrapped
    from minigrid.core.world_object import Wall
    # wall in every free cell of the agent's room except the agent's own
    start = tuple(int(v) for v in u.agent_pos)
    for c in reachable_cells(u, start):
        if c != start and u.grid.get(*c) is None:
            u.grid.set(c[0], c[1], Wall())
    with pytest.raises(UnsuitableLayout):
        relocate_blocker()(env)
    env.close()


# -- pickup_box's hook ---------------------------------------------------------

@pytest.mark.parametrize("env_id", [UNLOCKPICKUP, BLOCKED])
def test_pickup_box_hook_puts_the_key_in_hand_about_half_the_time(env_id):
    env = _env(env_id, PICKUP_BOX_HOOK)
    held = 0
    try:
        for seed in range(N):
            env.reset(seed=seed)
            sv = build_state_view(env)
            assert sv.door().is_open
            assert sv.carrying_type in (None, "key")
            held += sv.carrying_type == "key"
    finally:
        env.close()
    assert 0.40 * N <= held <= 0.60 * N, held


# -- the V9 regression ---------------------------------------------------------

def _blocked_pose(env, seed):
    """Agent two cells left of the door, facing the ball, with free cells behind."""
    env.reset(seed=seed)
    u = env.unwrapped
    sv = build_state_view(env)
    dx, dy = sv.door().pos
    stand, behind = (dx - 2, dy), (dx - 3, dy)
    if any(u.grid.get(*c) is not None for c in (stand, behind)):
        return None
    u.agent_pos = np.array(stand)
    u.agent_dir = 0                                       # facing right, at the ball
    return sv


def test_move_blocker_goal_is_false_while_the_ball_is_held():
    env = gym.make(BLOCKED)
    tried = 0
    try:
        for seed in range(200):
            if _blocked_pose(env, seed) is None:
                continue
            tried += 1
            env.step(PICKUP)
            sv = build_state_view(env)
            assert sv.carrying_type == "ball"
            # V9: the grid-based predicate alone already says "clear"...
            assert door_approach_clear()(sv)
            # ...which is why move_blocker's goal also requires an empty hand.
            assert not MOVE_BLOCKER_GOAL(sv)

            env.step(LEFT)
            env.step(LEFT)                                # now facing away from the door
            env.step(DROP)
            sv = build_state_view(env)
            assert sv.carrying_type is None
            ball = [o for o in sv.objects if o.type == "ball"][0]
            assert not _adjacent(ball.pos, sv.door().pos)
            assert MOVE_BLOCKER_GOAL(sv)
    finally:
        env.close()
    assert tried >= 20


def test_move_blocker_goal_is_false_if_the_ball_is_dropped_back_by_the_door():
    env = gym.make(BLOCKED)
    try:
        for seed in range(200):
            if _blocked_pose(env, seed) is None:
                continue
            env.step(PICKUP)
            env.step(DROP)                                # straight back on the door-front cell
            sv = build_state_view(env)
            assert not MOVE_BLOCKER_GOAL(sv)
            assert door_approach_blocked()(sv)
            return
    finally:
        env.close()
    pytest.fail("no usable seed")
