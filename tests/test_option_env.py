"""
OptionEnv tests (H-5).

Scripted frozen children on real MiniGrid envs: these check the SMDP contract,
not whether a policy learns.
"""

import numpy as np
import pytest

pytest.importorskip("minigrid")
pytest.importorskip("sb3_contrib")

import gymnasium as gym  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from src.envs.option_env import OptionEnv, ResetHookEnv, make_option_env  # noqa: E402
from src.skills.node import ALWAYS, NEVER, Predicate, SkillNode  # noqa: E402
from src.skills.predicates import carrying, door_open  # noqa: E402
from src.types import NodeStatus  # noqa: E402

UNLOCK = "MiniGrid-Unlock-v0"


class Scripted:
    """Primitive policy emitting one fixed action."""

    def __init__(self, action=2):
        self.action = action
        self.calls = 0

    def act(self, obs, deterministic=True):
        self.calls += 1
        return self.action


def primitive(node_id, goal, timeout=60, action=2, initiation=ALWAYS):
    n = SkillNode(node_id, goal=goal, initiation=initiation, timeout=timeout,
                  env_ids=[UNLOCK])
    n.policy = Scripted(action)
    n.status = NodeStatus.FROZEN
    return n


def unlock_door_node(timeout=288, max_decisions=12, children=None):
    kids = children or [
        primitive("{0,1} get_key", carrying("key"), timeout=120),
        primitive("{0,2} open_door", door_open(), timeout=60,
                  initiation=carrying("key")),
    ]
    return SkillNode("{1,1} unlock_door", children=kids, goal=door_open(),
                     timeout=timeout, max_decisions=max_decisions, env_ids=[UNLOCK])


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def test_action_space_is_the_child_count():
    env = make_option_env(unlock_door_node(), UNLOCK, seed=0)
    assert env.action_space.n == 2


def test_rejects_a_primitive_node():
    p = primitive("{0,1} x", NEVER)
    with pytest.raises(ValueError, match="composite"):
        OptionEnv(gym.make(UNLOCK), p)


def test_one_step_is_one_whole_child_execution():
    """The defining property of the SMDP: a decision spans many env steps."""
    node = unlock_door_node()
    env = make_option_env(node, UNLOCK, seed=0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(0)
    assert info["k"] > 1, "one SMDP step must cover a whole child execution"
    assert info["child_id"] == "{0,1} get_key"


def test_child_outcome_and_k_are_logged_every_decision():
    """[A: F14] Without k from run one, the exact-gamma^k study needs a re-run."""
    node = unlock_door_node()
    env = make_option_env(node, UNLOCK, seed=0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(0)
    for key in ("k", "child_id", "child_index", "child_outcome", "reselections",
                "decisions", "option_steps"):
        assert key in info, f"info is missing {key}"


# ---------------------------------------------------------------------------
# Reward -- [A: F10]
# ---------------------------------------------------------------------------

def test_reward_comes_from_the_nodes_own_goal_not_the_env():
    """
    Unlock pays on opening the door. A composite whose goal is something else
    must see none of that.
    """
    other = Predicate("never", lambda sv: False)
    node = SkillNode(
        "{1,9} never_satisfied",
        children=[primitive("{0,1} get_key", carrying("key"), timeout=120)],
        goal=other, timeout=288, max_decisions=4, env_ids=[UNLOCK],
    )
    env = make_option_env(node, UNLOCK, seed=0)
    env.reset(seed=0)
    total = 0.0
    for _ in range(4):
        _, r, term, trunc, _ = env.step(0)
        total += r
        if term or trunc:
            break
    assert total == 0.0, "env reward leaked into the composite"


def test_reward_shape_on_success():
    """A child that opens the door immediately: reward = 1 - 0.9*steps/timeout."""

    class OpenIt:
        def act(self, obs, deterministic=True):
            return 5          # toggle

    env_raw = gym.make(UNLOCK)
    env_raw.reset(seed=0)

    opener = primitive("{0,2} open_door", door_open(), timeout=60)
    opener.policy = OpenIt()
    node = SkillNode("{1,1} unlock_door", children=[opener], goal=door_open(),
                     timeout=288, max_decisions=8, env_ids=[UNLOCK])

    from src.envs.subgoal_env import give_key

    env = make_option_env(node, UNLOCK, reset_hook=give_key(), seed=0)
    env.reset(seed=0)
    total, steps = 0.0, 0
    for _ in range(8):
        _, r, term, trunc, info = env.step(0)
        total += r
        steps = info["option_steps"]
        if term or trunc:
            break
    if total > 0:
        assert total == pytest.approx(1.0 - 0.9 * steps / node.timeout, abs=1e-6)


# ---------------------------------------------------------------------------
# Masks -- [A: F1, F3]
# ---------------------------------------------------------------------------

def test_masks_are_goal_only_by_default():
    """open_door's initiation (holding the key) must not mask it out."""
    node = unlock_door_node()
    env = make_option_env(node, UNLOCK, seed=0)
    env.reset(seed=0)
    assert env.action_masks() == [True, True], (
        "initiation must not mask -- the composite has to learn that open_door "
        "fails without the key"
    )


def test_initiation_masking_ablation_restores_the_strict_mask():
    """C1 control (b): the same run with v2-as-locked masking."""
    node = unlock_door_node()
    env = make_option_env(node, UNLOCK, seed=0, initiation_masking=True)
    env.reset(seed=0)
    assert env.action_masks() == [True, False], "open_door needs the key"


def test_mask_never_all_false():
    node = unlock_door_node()
    env = make_option_env(node, UNLOCK, seed=0)
    env.reset(seed=0)
    for _ in range(6):
        masks = env.action_masks()
        assert any(masks), "MaskablePPO cannot take an all-False mask"
        _, _, term, trunc, _ = env.step(int(np.argmax(masks)))
        if term or trunc:
            break


def test_satisfied_child_is_masked_out():
    node = unlock_door_node()
    from src.envs.subgoal_env import give_key

    env = make_option_env(node, UNLOCK, reset_hook=give_key(), seed=0)
    env.reset(seed=0)
    # get_key's goal already holds, so it is withdrawn.
    assert env.action_masks() == [False, True]


# ---------------------------------------------------------------------------
# Bounds -- [A: F2, F7]
# ---------------------------------------------------------------------------

def test_max_decisions_truncates():
    node = unlock_door_node(max_decisions=3)
    env = make_option_env(node, UNLOCK, seed=0)
    env.reset(seed=0)
    decisions = 0
    for _ in range(10):
        _, _, term, trunc, info = env.step(0)
        decisions = info["decisions"]
        if term or trunc:
            break
    assert decisions <= 3
    assert trunc is True and term is False


def test_timeout_truncates():
    # Children must fit inside the parent's budget [A: F13], so shrink them too.
    small = [
        primitive("{0,1} get_key", carrying("key"), timeout=30),
        primitive("{0,2} open_door", door_open(), timeout=30, initiation=carrying("key")),
    ]
    node = unlock_door_node(timeout=100, max_decisions=50, children=small)
    env = make_option_env(node, UNLOCK, seed=0)
    env.reset(seed=0)
    for _ in range(50):
        _, _, term, trunc, info = env.step(0)
        if term or trunc:
            break
    assert info["option_steps"] >= 100 or term


def test_reset_refuses_a_goal_that_already_holds():
    """[A: F7]"""
    from src.envs.subgoal_env import open_door as open_door_hook

    node = unlock_door_node()
    # seed=None so the factory does not reset during construction -- the raise
    # must happen inside the assertion block.
    env = make_option_env(node, UNLOCK, reset_hook=open_door_hook(), seed=None)
    with pytest.raises(RuntimeError, match="goal still|no child"):
        env.reset(seed=0)


def test_only_frozen_children_may_be_trained_over():
    from src.training.node_trainer import train_composite

    child = primitive("{0,1} get_key", carrying("key"))
    child.status = NodeStatus.STABLE          # stable but not frozen
    node = SkillNode("{1,1} p", children=[child], goal=door_open(),
                     timeout=288, env_ids=[UNLOCK])
    with pytest.raises(RuntimeError, match="not frozen"):
        train_composite(node, [UNLOCK], total_timesteps=16, n_envs=1)


# ---------------------------------------------------------------------------
# Vectorization -- [A: F15]
# ---------------------------------------------------------------------------

def test_dummyvecenv_n4_matches_n1_for_a_scripted_policy():
    """
    The stop-and-report condition from CLAUDE.md: frozen children are shared
    modules, so a vectorized OptionEnv must still behave like a single env.
    """
    def rollout(n_envs):
        node = unlock_door_node(max_decisions=6)
        venv = DummyVecEnv([
            make_option_env_factory(node, seed=100 + i) for i in range(n_envs)
        ])
        venv.reset()
        rewards = []
        for _ in range(6):
            obs, r, done, infos = venv.step(np.zeros(n_envs, dtype=int))
            rewards.append([round(float(x), 6) for x in r])
        venv.close()
        return rewards

    def make_option_env_factory(node, seed):
        def _f():
            return make_option_env(node, UNLOCK, seed=seed)
        return _f

    single = rollout(1)
    quad = rollout(4)
    # Env 0 of the 4-env run uses the same seed as the single-env run.
    assert [row[0] for row in quad] == [row[0] for row in single], (
        "a shared frozen child behaves differently under vectorization"
    )


def test_maskable_ppo_can_read_the_masks_through_the_wrapper_stack():
    """
    Gymnasium 1.x stopped forwarding arbitrary attributes through Wrapper, so a
    plain Monitor on top would hide action_masks from sb3-contrib.
    """
    from sb3_contrib.common.maskable.utils import get_action_masks

    node = unlock_door_node()
    venv = DummyVecEnv([lambda: make_option_env(node, UNLOCK, seed=0)])
    venv.reset()
    masks = get_action_masks(venv)
    assert masks.shape == (1, 2)


def test_maskable_ppo_trains_a_few_steps():
    from sb3_contrib import MaskablePPO

    node = unlock_door_node()
    venv = DummyVecEnv([lambda: make_option_env(node, UNLOCK, seed=0)])
    model = MaskablePPO("MlpPolicy", venv, n_steps=16, batch_size=16,
                        seed=0, device="cpu", verbose=0)
    model.learn(total_timesteps=32)
    venv.close()


# ---------------------------------------------------------------------------
# ResetHookEnv
# ---------------------------------------------------------------------------

def test_reset_hook_env_regenerates_the_observation():
    from src.envs.obs import apply_obs_pipeline
    from src.envs.subgoal_env import give_key

    env = apply_obs_pipeline(ResetHookEnv(gym.make(UNLOCK), give_key()))
    obs, _ = env.reset(seed=0)
    agent_cell = obs.reshape(7, 7, 3)[3, 6]
    assert agent_cell[0] == pytest.approx(5 / 10), "the held key is not in the obs"
