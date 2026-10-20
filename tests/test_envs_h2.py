"""
H-2: observation pipeline, predicates, reset hooks and SubgoalEnv.

These run against the real minigrid==3.1.0 envs, not mocks: the whole point of
H-2 is that the predicates describe the actual environments.
"""

import gymnasium as gym
import numpy as np
import pytest

gym_minigrid = pytest.importorskip("minigrid")

from src.envs.obs import (  # noqa: E402
    CHANNEL_MAX,
    OBS_DIM,
    ChannelNormalize,
    apply_obs_pipeline,
    make_base_env,
    normalize_flat_obs,
    obs_pipeline_name,
)
from src.envs.subgoal_env import (  # noqa: E402
    NO_HOOK,
    SubgoalEnv,
    clear_blocker,
    compose,
    give_key,
    make_subgoal_env,
    open_door,
    randomize_agent_pose,
    reachable_cells,
)
from src.skills.node import NEVER, SkillNode  # noqa: E402
from src.skills.predicates import (  # noqa: E402
    carrying,
    door_approach_clear,
    door_open,
    on_goal_tile,
)
from src.skills.state import build_state_view  # noqa: E402

CURRICULUM = [
    "MiniGrid-Unlock-v0",
    "MiniGrid-DoorKey-6x6-v0",
    "MiniGrid-UnlockPickup-v0",
    "MiniGrid-BlockedUnlockPickup-v0",
]


def raw(env_id, seed=0):
    env = gym.make(env_id)
    env.reset(seed=seed)
    return env


# ---------------------------------------------------------------------------
# Env IDs and budgets (v2 §11, verified not assumed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("env_id", CURRICULUM)
def test_curriculum_env_exists(env_id):
    assert env_id in gym.registry


@pytest.mark.parametrize(
    "env_id,max_steps",
    [
        ("MiniGrid-Unlock-v0", 288),
        ("MiniGrid-DoorKey-6x6-v0", 360),
        ("MiniGrid-UnlockPickup-v0", 288),
        ("MiniGrid-BlockedUnlockPickup-v0", 576),
    ],
)
def test_env_step_budgets_match_the_timeout_invariant(env_id, max_steps):
    """[A: F13] Root timeouts are checked against these numbers."""
    assert raw(env_id).unwrapped.max_steps == max_steps


# ---------------------------------------------------------------------------
# Observation pipeline -- v2 §5.1 [A: F16, F24]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("env_id", CURRICULUM)
def test_pipeline_gives_147_normalized_dims(env_id):
    env = make_base_env(env_id, seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert obs.min() >= 0.0 and obs.max() <= 1.0


def test_normalization_divides_the_right_axis():
    """
    The flatten is C-order over (7,7,3), so channels are fastest-varying. A
    wrong-axis reshape still produces 147 numbers in [0,1] and would pass a
    naive range check -- this pins the per-channel maxima instead.
    """
    grid = np.zeros((7, 7, 3), dtype=np.uint8)
    grid[..., 0] = 10   # object
    grid[..., 1] = 5    # colour
    grid[..., 2] = 2    # state
    out = normalize_flat_obs(grid.reshape(-1)).reshape(7, 7, 3)
    assert np.allclose(out[..., 0], 1.0)
    assert np.allclose(out[..., 1], 1.0)
    assert np.allclose(out[..., 2], 1.0)

    half = np.zeros((7, 7, 3), dtype=np.uint8)
    half[..., 0] = 5
    got = normalize_flat_obs(half.reshape(-1)).reshape(7, 7, 3)
    assert np.allclose(got[..., 0], 0.5)


def test_normalization_is_batched_and_single_consistent():
    single = np.arange(OBS_DIM, dtype=np.uint8) % 11
    batch = np.stack([single, single])
    assert np.allclose(normalize_flat_obs(batch)[0], normalize_flat_obs(single))


def test_raw_pipeline_remains_available_for_the_legacy_column():
    """col01 was trained on raw integers; C-2 needs to be able to say so."""
    env = apply_obs_pipeline(gym.make("MiniGrid-Empty-8x8-v0"), normalize=False)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.uint8
    assert obs_pipeline_name(False) != obs_pipeline_name(True)


def test_channel_normalize_rejects_an_unflattened_input():
    with pytest.raises(ValueError, match="flattened"):
        ChannelNormalize(gym.make("MiniGrid-Unlock-v0"))


# ---------------------------------------------------------------------------
# StateView -- v2 §4.2 [A: F5, F30]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("env_id", CURRICULUM)
def test_state_view_finds_the_door_in_every_env(env_id):
    """[A: F30] RoomGrid exposes .door; DoorKey does not, so the builder scans."""
    sv = build_state_view(raw(env_id))
    assert sv.door() is not None, f"no door found in {env_id}"


def test_state_view_finds_key_and_goal():
    sv = build_state_view(raw("MiniGrid-DoorKey-6x6-v0"))
    assert sv.obj("key") is not None
    assert sv.goal_pos is not None


def test_state_view_version_is_the_cache_key():
    env = raw("MiniGrid-Unlock-v0")
    assert build_state_view(env, version=7).version == 7


def test_state_view_is_a_snapshot_not_a_live_reference():
    """[A: F5] Mutating the env must not change a StateView already taken."""
    env = raw("MiniGrid-Unlock-v0")
    before = build_state_view(env)
    assert before.door().is_open is False
    open_door()(env)
    assert before.door().is_open is False, "snapshot changed under us"
    assert build_state_view(env, 1).door().is_open is True


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------

def test_carrying_predicate_truth_table():
    env = raw("MiniGrid-Unlock-v0")
    pred = carrying("key")
    assert pred(build_state_view(env)) is False
    give_key()(env)
    assert pred(build_state_view(env, 1)) is True
    assert carrying("box")(build_state_view(env, 1)) is False


def test_door_open_predicate_truth_table():
    env = raw("MiniGrid-Unlock-v0")
    pred = door_open()
    assert pred(build_state_view(env)) is False
    open_door()(env)
    assert pred(build_state_view(env, 1)) is True


def test_on_goal_tile_predicate_truth_table():
    env = raw("MiniGrid-DoorKey-6x6-v0")
    pred = on_goal_tile()
    sv = build_state_view(env)
    assert pred(sv) is False
    env.unwrapped.agent_pos = np.array(sv.goal_pos)
    assert pred(build_state_view(env, 1)) is True


def test_door_approach_clear_predicate_truth_table():
    env = raw("MiniGrid-BlockedUnlockPickup-v0")
    pred = door_approach_clear()
    assert pred(build_state_view(env)) is False, "a ball blocks the door at reset"
    clear_blocker()(env)
    assert pred(build_state_view(env, 1)) is True


def test_predicates_carry_a_spec_for_hashing():
    assert carrying("key").spec == "carrying:key"
    assert door_open("yellow").spec == "door:open:yellow"
    assert on_goal_tile().spec == "agent:on_goal"


# ---------------------------------------------------------------------------
# Reset hooks
# ---------------------------------------------------------------------------

def test_give_key_removes_it_from_the_grid():
    env = raw("MiniGrid-Unlock-v0")
    before = build_state_view(env)
    assert before.obj("key") is not None
    give_key()(env)
    after = build_state_view(env, 1)
    assert after.carrying_type == "key"
    assert after.obj("key") is None, "the key must leave the grid when picked up"


def test_open_door_unlocks_as_well_as_opens():
    env = raw("MiniGrid-DoorKey-6x6-v0")
    open_door()(env)
    d = build_state_view(env, 1).door()
    assert d.is_open and not d.is_locked


@pytest.mark.parametrize("env_id", ["MiniGrid-Unlock-v0", "MiniGrid-DoorKey-6x6-v0"])
def test_randomize_agent_pose_keeps_the_key_reachable(env_id):
    """
    [A: F29] The agent must never spawn behind a locked door: the episode would
    be unsolvable and would look like a learning failure.
    """
    pred = carrying("key")
    for seed in range(12):
        env = raw(env_id, seed=seed)
        randomize_agent_pose(anchor="key")(env)
        sv = build_state_view(env, 1)
        key = sv.obj("key")
        region = reachable_cells(env.unwrapped, key.pos)
        assert tuple(sv.agent_pos) in region, (
            f"{env_id} seed {seed}: agent at {sv.agent_pos} cannot reach the key at {key.pos}"
        )
        assert pred(sv) is False


def test_randomize_agent_pose_actually_varies_the_pose():
    poses = set()
    for seed in range(10):
        env = raw("MiniGrid-Unlock-v0", seed=seed)
        randomize_agent_pose(anchor="key")(env)
        sv = build_state_view(env, 1)
        poses.add((tuple(sv.agent_pos), sv.agent_dir))
    assert len(poses) > 3, f"only {len(poses)} distinct poses over 10 seeds"


def test_compose_concatenates_specs():
    h = compose(give_key(), randomize_agent_pose(anchor="key"))
    assert h.spec == "give_key+randomize_agent_pose:key"
    assert NO_HOOK.spec == "none"


def test_reachable_cells_treats_a_locked_door_as_a_wall():
    env = raw("MiniGrid-Unlock-v0")
    sv = build_state_view(env)
    key_region = reachable_cells(env.unwrapped, sv.obj("key").pos)
    assert sv.door().pos not in key_region, "a locked door must block the flood fill"

    open_door()(env)
    sv2 = build_state_view(env, 1)
    bigger = reachable_cells(env.unwrapped, sv2.obj("key").pos)
    assert len(bigger) > len(key_region), "opening the door must enlarge the region"


# ---------------------------------------------------------------------------
# SubgoalEnv -- v2 §5.1
# ---------------------------------------------------------------------------

def make_node(node_id, goal, timeout=120, env_ids=("MiniGrid-Unlock-v0",)):
    return SkillNode(node_id, goal=goal, timeout=timeout, env_ids=list(env_ids))


def test_subgoal_env_observation_contract():
    node = make_node("{0,1} get_key", carrying("key"))
    env = make_subgoal_env(node, "MiniGrid-Unlock-v0", seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32


def test_subgoal_env_reward_only_on_its_own_goal():
    """
    [A: F10] Unlock's own reward fires on opening the door. A `get_key` node must
    see none of it -- only its own goal pays.
    """
    node = make_node("{0,1} get_key", carrying("key"), timeout=50)
    env = gym.make("MiniGrid-Unlock-v0")
    sg = SubgoalEnv(env, node)
    sg.reset(seed=0)

    total = 0.0
    for _ in range(20):
        _, r, term, trunc, _ = sg.step(sg.action_space.sample())
        total += r
        if term or trunc:
            break
    assert total == 0.0 or term, "reward may only appear on the node's own goal"


def test_subgoal_env_rewards_the_goal_with_the_right_shape():
    node = make_node("{0,2} open_door", door_open(), timeout=100)
    env = gym.make("MiniGrid-Unlock-v0")
    sg = SubgoalEnv(env, node, reset_hook=give_key())
    sg.reset(seed=0)

    # Force the goal on the next step rather than solving the task here.
    open_door()(sg.env)
    _, r, term, trunc, info = sg.step(6)   # 'done' is a no-op action
    assert term is True and trunc is False
    assert info["subgoal_reached"] is True
    assert r == pytest.approx(1.0 - 0.9 * (1 / 100))


def test_subgoal_env_truncates_at_the_node_timeout():
    node = make_node("{0,1} get_key", carrying("key"), timeout=5)
    env = gym.make("MiniGrid-Unlock-v0")
    sg = SubgoalEnv(env, node)
    sg.reset(seed=0)
    for i in range(5):
        _, r, term, trunc, _ = sg.step(0)   # turn left: never reaches the key
    assert trunc is True and term is False
    assert r == 0.0


def test_subgoal_env_refuses_a_goal_that_holds_at_reset():
    """[A: F7] Otherwise the episode ends at step 0 and inflates the success rate."""
    node = make_node("{0,2} open_door", door_open(), timeout=50)
    env = gym.make("MiniGrid-Unlock-v0")
    sg = SubgoalEnv(env, node, reset_hook=open_door(), max_reset_tries=3)
    with pytest.raises(RuntimeError, match="goal held at reset"):
        sg.reset(seed=0)


def test_subgoal_env_reset_hook_is_visible_in_the_observation():
    """The hook mutates the grid after reset, so the obs must be regenerated."""
    node = make_node("{0,2} open_door", door_open(), timeout=50)
    env = make_subgoal_env(node, "MiniGrid-Unlock-v0", reset_hook=give_key(), seed=0)
    obs, _ = env.reset(seed=0)
    # The carried object is written into the agent's own cell, (3, 6).
    agent_cell = obs.reshape(7, 7, 3)[3, 6]
    key_idx = 5 / CHANNEL_MAX[0]          # OBJECT_TO_IDX["key"] == 5
    assert agent_cell[0] == pytest.approx(key_idx), (
        "the held key is not visible in the observation -- the hook ran after "
        "the observation was generated"
    )


def test_subgoal_env_works_on_both_layouts_for_the_same_node():
    """[A: F29] T1/T2 are trained on Unlock and DoorKey alike."""
    node = make_node("{0,1} get_key", carrying("key"),
                     env_ids=("MiniGrid-Unlock-v0", "MiniGrid-DoorKey-6x6-v0"))
    for env_id in node.env_ids:
        env = make_subgoal_env(node, env_id, reset_hook=randomize_agent_pose("key"), seed=1)
        obs, _ = env.reset(seed=1)
        assert obs.shape == (OBS_DIM,)
