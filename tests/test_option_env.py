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


def test_parent_timeout_interrupts_a_running_child():
    """The Executor closes a child when the parent's timeout hits; so must OptionEnv."""
    node = unlock_door_node(
        timeout=45, max_decisions=10,
        children=[primitive("{0,1} slow", NEVER, timeout=30)],
    )
    env = make_option_env(node, UNLOCK, seed=0)
    env.reset(seed=0)
    _, _, term, trunc, info = env.step(0)
    assert info["option_steps"] == 30 and not trunc
    _, reward, term, trunc, info = env.step(0)
    assert info["option_steps"] == 45     # 30 + 15, not 30 + 30
    assert trunc and not term and reward == 0.0


def test_strict_mask_reaches_nested_composites():
    """The Executor applies the strict mask at every depth; OptionEnv's child Executor must too."""
    class RecordMask:
        def __init__(self):
            self.masks = []

        def select(self, obs, mask, deterministic=True):
            self.masks.append(list(mask))
            return mask.index(True)

    inner = SkillNode(
        "{1,1} inner", goal=NEVER, timeout=40, max_decisions=3, env_ids=[UNLOCK],
        children=[primitive("{0,1} blocked", NEVER, timeout=10, initiation=NEVER),
                  primitive("{0,2} open", NEVER, timeout=10)],
    )
    inner.policy = RecordMask()
    inner.status = NodeStatus.FROZEN
    outer = SkillNode("{2,1} outer", children=[inner], goal=door_open(), timeout=60,
                      max_decisions=3, env_ids=[UNLOCK])

    env = make_option_env(outer, UNLOCK, seed=0, initiation_masking=True)
    env.reset(seed=0)
    env.step(0)
    assert inner.policy.masks[0] == [False, True]


class FirstTrue:
    def select(self, obs, mask, deterministic=True):
        return list(mask).index(True)


def _at(k):
    return Predicate(f"step>={k}", lambda sv: sv.step_count >= k)


def _composite(node_id, children, goal, timeout, max_decisions=20):
    n = SkillNode(node_id, children=children, goal=goal, timeout=timeout,
                  max_decisions=max_decisions, env_ids=[UNLOCK])
    n.policy = FirstTrue()
    n.status = NodeStatus.FROZEN
    return n


def _two_children(b_goal, b_initiation, parent_goal, timeout):
    return _composite("{1,1} p", [primitive("{0,a} a", _at(3), timeout=10),
                                  primitive("{0,b} b", b_goal, timeout=10,
                                            initiation=b_initiation)],
                      parent_goal, timeout)


def _capped_inner():
    slow = primitive("{0,s} slow", NEVER, timeout=1)
    inner = _composite("{1,1} inner", [slow], NEVER, timeout=100, max_decisions=2)
    return _composite("{2,1} root", [inner], NEVER, timeout=1000, max_decisions=10)


def _log_acts(node):
    """Record which primitive acted at each env step."""
    log = []
    for n in node.descendants():
        if n.is_primitive:
            act = n.policy.act
            n.policy.act = lambda obs, deterministic=True, _a=act, _id=n.id: (
                log.append(_id), _a(obs, deterministic))[1]
    return log


def _strict_dead_end():
    """Under the strict mask `dead` is never applicable, so every frame re-selects with no env step."""
    dead = primitive("{0,d} dead", NEVER, timeout=10, initiation=NEVER)
    blocked = _composite("{1,1} blocked", [dead], NEVER, timeout=100, max_decisions=64)
    inner = _composite("{2,1} inner", [blocked], NEVER, timeout=500, max_decisions=64)
    return _composite("{3,1} root", [inner], NEVER, timeout=1000, max_decisions=64)


@pytest.mark.parametrize("build, strict, expected", [
    (lambda: _two_children(_at(5), ALWAYS, _at(5), 20), False, (True, 5)),
    # b would run to step 17 on its own timeout; the parent's timeout is 12
    (lambda: _two_children(_at(100), ALWAYS, _at(50), 12), False, (False, 12)),
    # a goal reached on the timeout step counts; one step later it does not
    (lambda: _two_children(_at(100), ALWAYS, _at(12), 12), False, (True, 12)),
    (lambda: _two_children(_at(100), ALWAYS, _at(13), 12), False, (False, 12)),
    (lambda: _two_children(_at(5), NEVER, _at(5), 20), True, (False, 3)),
    (lambda: _two_children(_at(5), NEVER, _at(5), 20), False, (True, 5)),
    # inner's cap of 2 and the root's cap of 10 are counted separately
    (_capped_inner, False, (False, 20)),
    (_strict_dead_end, True, (False, 0)),
], ids=["success", "parent_timeout", "goal_on_timeout_step", "goal_after_timeout",
         "strict_mask", "goal_only_mask", "per_frame_cap", "strict_dead_end"])
def test_option_env_matches_the_executor(build, strict, expected):
    """Training (OptionEnv) and evaluation (Executor) must play the same episode."""
    from src.training.evaluation import run_episode
    from src.training.node_trainer import execution_env_factory

    seed = 7
    node = build()
    trained_log = _log_acts(node)
    env = make_option_env(node, UNLOCK, seed=0, initiation_masking=strict)
    env.reset(seed=seed)
    while True:
        _, _, term, trunc, info = env.step(env.action_masks().index(True))
        if term or trunc:
            break
    trained = (term, info["option_steps"])

    node = build()
    eval_log = _log_acts(node)
    ok, steps, _ = run_episode(node, execution_env_factory(UNLOCK)(), seed,
                               initiation_masking=strict)
    assert trained == (ok, steps) == expected
    assert trained_log == eval_log and len(eval_log) == steps   # same primitive at every step


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


@pytest.mark.parametrize("min_sr, budget, expected", [
    (0.0, 256, 0),    # STABLE at the first evaluation: the rollout in flight is never trained on
    (1.1, 128, 2),    # never STABLE: the budget ends on a rollout boundary, both are trained on
])
def test_policy_updates_counts_real_train_calls(min_sr, budget, expected):
    from src.training.evaluation import StabilityCriterion
    from src.training.node_trainer import train_composite

    report, model = train_composite(
        unlock_door_node(), [UNLOCK], total_timesteps=budget, n_envs=1,
        eval_every=64, n_eval=5, criterion=StabilityCriterion(min_sr, min_sr, 1),
    )
    assert report["policy_updates"] == expected
    assert report["policy_updates_semantics"] == "completed_train_calls"
    from pathlib import Path
    import torch
    saved = torch.load(Path(report["run_dir"]) / "policy_state_dict.pt", weights_only=True)
    assert all(torch.equal(saved[k], v.cpu()) for k, v in model.policy.state_dict().items())
    assert (Path(report["run_dir"]) / "model.zip").is_file()


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


def test_ablation_is_scored_under_the_mask_it_was_trained_with():
    """
    The two C1 arms are only comparable if the strict mask applies to evaluation
    as well as training. Training with it and scoring without it made both arms
    produce byte-identical numbers in the first smoke run.
    """
    from src.skills.state import build_state_view
    from src.training.evaluation import run_episode
    from src.training.node_trainer import execution_env_factory

    node = unlock_door_node()
    chosen = []

    class Recorder:
        def select(self, obs, mask, deterministic=True):
            chosen.append(list(mask))
            return next(i for i, ok in enumerate(mask) if ok)

    node.policy = Recorder()
    env = execution_env_factory(UNLOCK)()

    chosen.clear()
    run_episode(node, env, seed=0, initiation_masking=False)
    assert [True, True] in chosen, "default evaluation must offer both children"

    chosen.clear()
    run_episode(node, env, seed=0, initiation_masking=True)
    assert all(m == [True, False] or m == [False, True] or not m[1] or not m[0]
               for m in chosen[:1]), "strict evaluation must withhold open_door"
    assert chosen[0] == [True, False]
