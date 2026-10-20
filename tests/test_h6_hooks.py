"""
H-6.6: the two-key reset hooks, against real minigrid 3.1.0.

These hooks define the initial-condition distribution of the genuine-choice task
(plan Part C), and three of their properties are what make that task a choice at
all rather than a coin flip:

  * **exactly one key opens the door.** `Door.toggle` is colour-matched, so a
    wrong pick cannot open it.
  * **both keys are independently reachable.** `pickup` needs an empty hand and
    neither `get_key_<colour>` primitive is trained to drop, so a decoy reachable
    only *through* the matching key would make its primitive unachievable.
  * **the door is inside the agent's view.** The composite chooses from one
    observation with a feedforward policy; a signal it cannot see makes the
    experiment a measurement of partial observability instead.

Each is asserted over many real resets rather than on a constructed grid,
because every one of them is a property of MiniGrid's layout generator as much
as of the hook. The second was not in the original design: DoorKey draws its
splitting wall at x=2 or x=3, so the left room is often a single column and the
layout's own key can box the agent in.
"""

from __future__ import annotations

import gymnasium as gym
import pytest
from minigrid.core.constants import OBJECT_TO_IDX

from src.envs.option_env import ResetHookEnv
from src.envs.subgoal_env import (
    NO_HOOK,
    SubgoalEnv,
    UnsuitableLayout,
    compose,
    face_toward,
    place_two_keys,
    randomize_agent_pose,
    recolor_door,
    remove_keys,
    two_keys,
)
from src.skills.node import SkillNode
from src.skills.predicates import carrying, door_open
from src.skills.state import build_state_view

DOORKEY = "MiniGrid-DoorKey-6x6-v0"
COLORS = ("yellow", "blue")
N = 120


@pytest.fixture
def hooked_env():
    env = ResetHookEnv(gym.make(DOORKEY), two_keys(COLORS))
    yield env
    env.close()


def _keys(sv):
    return [o for o in sv.objects if o.type == "key"]


def _standable_from(u, cell, start):
    """Mirror of the hook's own reachability test, recomputed independently."""
    from src.envs.subgoal_env import reachable_cells

    reach = set(reachable_cells(u, start)) | {start}
    x, y = cell
    return any((nx, ny) in reach
               for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))


# --------------------------------------------------------------------------
# two_keys: the three properties the choice task depends on
# --------------------------------------------------------------------------

def test_two_keys_places_exactly_one_matching_key_and_one_decoy(hooked_env):
    for seed in range(N):
        hooked_env.reset(seed=seed)
        sv = build_state_view(hooked_env, 0)
        keys = _keys(sv)
        door = sv.door()
        assert len(keys) == 2, f"seed {seed}: {len(keys)} keys"
        matching = [k for k in keys if k.color == door.color]
        assert len(matching) == 1, f"seed {seed}: {len(matching)} keys match the door"
        assert {k.color for k in keys} == set(COLORS), f"seed {seed}"


def test_two_keys_leaves_both_keys_independently_reachable(hooked_env):
    """
    The property the hook had to be rewritten for. Recomputed here from the
    finished grid rather than trusting the hook's own check, so a bug in that
    check cannot hide behind itself.
    """
    for seed in range(N):
        hooked_env.reset(seed=seed)
        u = hooked_env.unwrapped
        sv = build_state_view(hooked_env, 0)
        start = sv.agent_pos
        for key in _keys(sv):
            assert _standable_from(u, key.pos, start), (
                f"seed {seed}: the {key.color} key at {key.pos} cannot be picked "
                f"up from the agent's region without moving the other key"
            )


def test_two_keys_keeps_the_door_in_the_observation(hooked_env):
    """[H-6 V7] 54% with a random facing; the hook must make it 100%."""
    for seed in range(N):
        hooked_env.reset(seed=seed)
        u = hooked_env.unwrapped
        img = u.gen_obs()["image"]
        assert (img[:, :, 0] == OBJECT_TO_IDX["door"]).any(), f"seed {seed}"


def test_two_keys_never_blocks_the_door_approach(hooked_env):
    """
    The guard that cost the most to find. MiniGrid `toggle` acts on
    `front_pos`, so a locked door can only be opened from a cell orthogonally
    adjacent to it, and DoorKey's door sits in a vertical wall -- on the agent's
    side there is exactly one such cell. A key placed there makes the door
    impossible to open for ANY policy.

    Before the guard, 59 of 400 layouts (14.7%) were like that, which capped
    every door-opening primitive at ~0.85 held-out. It was first misread as a
    learning limit, and a 600k-step retrain under a stricter stopping rule
    plateaued at exactly 0.92 before the failures were inspected.
    """
    for seed in range(N):
        hooked_env.reset(seed=seed)
        u = hooked_env.unwrapped
        sv = build_state_view(hooked_env, 0)
        dx, dy = sv.door().pos
        usable = 0
        for nx, ny in ((dx - 1, dy), (dx + 1, dy), (dx, dy - 1), (dx, dy + 1)):
            if not (0 <= nx < u.width and 0 <= ny < u.height):
                continue
            cell = u.grid.get(nx, ny)
            if cell is None or cell.type != "wall":
                usable += 1
        assert usable >= 1, f"seed {seed}: the door at {(dx, dy)} cannot be reached"
        for key in _keys(sv):
            assert key.pos not in {(dx - 1, dy), (dx + 1, dy), (dx, dy - 1), (dx, dy + 1)}, (
                f"seed {seed}: the {key.color} key blocks the door's approach"
            )


def test_two_keys_keeps_the_door_approachable(hooked_env):
    """
    The second layer of the same bug, and the more expensive one to see.

    Keeping a key off the door's approach cell is not enough: in a one-column
    room a key between the agent and that cell blocks the path without occupying
    it, so the door is unopenable for any policy. That happened on 3 of the 50
    stopping seeds and 4 of the 50 held-out seeds, which is exactly the gap
    between {0,8} open_door_tk's 0.94 stopping score and the 0.84 it generalized
    to -- half its held-out failures were unsolvable episodes.
    """
    from src.envs.subgoal_env import reachable_cells

    for seed in range(N):
        hooked_env.reset(seed=seed)
        u = hooked_env.unwrapped
        sv = build_state_view(hooked_env, 0)
        dx, dy = sv.door().pos
        reach = set(reachable_cells(u, sv.agent_pos)) | {tuple(sv.agent_pos)}
        usable = [
            (nx, ny)
            for nx, ny in ((dx - 1, dy), (dx + 1, dy), (dx, dy - 1), (dx, dy + 1))
            if 0 <= nx < u.width and 0 <= ny < u.height
            and not (u.grid.get(nx, ny) is not None
                     and u.grid.get(nx, ny).type == "wall")
            and (nx, ny) in reach
        ]
        assert usable, (
            f"seed {seed}: no cell the agent can reach is adjacent to the door at "
            f"{(dx, dy)} -- this episode is unsolvable for any policy"
        )


def test_two_keys_starts_empty_handed_with_the_door_locked(hooked_env):
    for seed in range(N):
        hooked_env.reset(seed=seed)
        sv = build_state_view(hooked_env, 0)
        assert sv.carrying_type is None, f"seed {seed}"
        door = sv.door()
        assert door.is_locked and not door.is_open, f"seed {seed}"


def test_two_keys_draws_both_door_colours(hooked_env):
    seen = set()
    for seed in range(N):
        hooked_env.reset(seed=seed)
        seen.add(build_state_view(hooked_env, 0).door().color)
    assert seen == set(COLORS)


def test_two_keys_is_reproducible_from_the_seed(hooked_env):
    """
    Includes the UnsuitableLayout retry path: rejecting a layout consumes draws
    from the env's RNG, so reproducibility is not free -- it holds only because
    the retries happen inside a stream seeded by the episode's seed.
    """
    for seed in (3, 11, 17, 29):
        hooked_env.reset(seed=seed)
        first = build_state_view(hooked_env, 0)
        hooked_env.reset(seed=seed)
        assert build_state_view(hooked_env, 0) == first, f"seed {seed}"


def test_two_keys_spec_names_every_step_in_order():
    """The spec string goes into spec_hash, so the order has to be visible."""
    assert two_keys(COLORS).spec == (
        "recolor_door:yellow|blue+remove_keys+randomize_agent_pose:any"
        "+place_two_keys:yellow|blue:door_clear+face_toward:door"
    )


# --------------------------------------------------------------------------
# the individual hooks
# --------------------------------------------------------------------------

def test_recolor_door_only_touches_the_door():
    env = gym.make(DOORKEY)
    hook = recolor_door(COLORS)
    for seed in range(20):
        env.reset(seed=seed)
        hook(env)
        sv = build_state_view(env, 0)
        assert sv.door().color in COLORS
        # DoorKey's own key is still yellow: recolour_door does not touch it,
        # which is why place_two_keys reads the door's colour rather than the
        # key's.
        assert all(k.color == "yellow" for k in _keys(sv))
    env.close()


def test_remove_keys_clears_the_grid_and_the_hand():
    from src.envs.subgoal_env import give_key

    env = gym.make(DOORKEY)
    hook = compose(give_key(), remove_keys())
    for seed in range(20):
        env.reset(seed=seed)
        hook(env)
        sv = build_state_view(env, 0)
        assert _keys(sv) == []
        assert sv.carrying_type is None
    env.close()


def test_place_two_keys_rejects_a_layout_with_too_few_free_cells():
    """
    Build the pathological case by hand: the agent walled into its own cell.
    `UnsuitableLayout` rather than `RuntimeError` is the contract -- the former
    means "draw another layout", the latter means "the hook is wrong".
    """
    from minigrid.core.world_object import Wall

    env = gym.make(DOORKEY)
    env.reset(seed=0)
    u = env.unwrapped
    remove_keys()(env)
    ax, ay = int(u.agent_pos[0]), int(u.agent_pos[1])
    for nx, ny in ((ax + 1, ay), (ax - 1, ay), (ax, ay + 1), (ax, ay - 1)):
        if 0 <= nx < u.width and 0 <= ny < u.height:
            u.grid.set(nx, ny, Wall())
    with pytest.raises(UnsuitableLayout):
        place_two_keys(COLORS)(env)
    env.close()


def test_face_toward_leaves_the_direction_alone_when_nothing_is_visible():
    env = gym.make(DOORKEY)
    env.reset(seed=0)
    u = env.unwrapped
    u.agent_dir = 2
    face_toward("box")(env)          # DoorKey has no box
    assert u.agent_dir == 2
    env.close()


def test_face_toward_is_idempotent_on_an_already_facing_pose():
    env = gym.make(DOORKEY)
    hook = face_toward("door")
    for seed in range(20):
        env.reset(seed=seed)
        hook(env)
        first = int(env.unwrapped.agent_dir)
        img = env.unwrapped.gen_obs()["image"]
        assert (img[:, :, 0] == OBJECT_TO_IDX["door"]).any()
        # Applying it again must keep the door visible (it may pick another of
        # the valid directions, which is deliberate -- pose diversity).
        hook(env)
        img = env.unwrapped.gen_obs()["image"]
        assert (img[:, :, 0] == OBJECT_TO_IDX["door"]).any(), seed
        assert isinstance(first, int)
    env.close()


# --------------------------------------------------------------------------
# the retry contract, on both env wrappers
# --------------------------------------------------------------------------

def _always_unsuitable():
    from src.envs.subgoal_env import ResetHook

    calls = {"n": 0}

    def fn(env):
        calls["n"] += 1
        raise UnsuitableLayout("test")

    return ResetHook("always_unsuitable", fn), calls


def test_reset_hook_env_retries_then_raises():
    hook, calls = _always_unsuitable()
    env = ResetHookEnv(gym.make(DOORKEY), hook, max_reset_tries=5)
    with pytest.raises(RuntimeError, match="no suitable layout"):
        env.reset(seed=0)
    assert calls["n"] == 5
    env.close()


def test_subgoal_env_retries_an_unsuitable_layout():
    """
    One bad draw must not kill a long training run. `two_keys` rejects roughly
    the layouts where the left room is a single column, so this path is taken
    routinely, not exceptionally.
    """
    node = SkillNode("{0,t} t", goal=carrying("key", "yellow"), timeout=20,
                     env_ids=[DOORKEY], train_hook=two_keys(COLORS).spec)
    env = SubgoalEnv(gym.make(DOORKEY), node, two_keys(COLORS))
    for seed in range(30):
        obs, _ = env.reset(seed=seed)
        sv = env.state_view
        assert len(_keys(sv)) == 2
        assert not node.goal(sv)          # [A: F7]
    env.close()


def test_subgoal_env_raises_when_no_layout_ever_works():
    hook, _ = _always_unsuitable()
    node = SkillNode("{0,t} t", goal=door_open(), timeout=20, env_ids=[DOORKEY])
    env = SubgoalEnv(gym.make(DOORKEY), node, hook, max_reset_tries=4)
    with pytest.raises(RuntimeError):
        env.reset(seed=0)
    env.close()
