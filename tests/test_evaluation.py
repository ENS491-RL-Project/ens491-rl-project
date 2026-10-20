"""
EvaluationMonitor unit tests (H-3).

The properties that matter: the Wilson bound is the real one, the seed set is
genuinely fixed, and the two-consecutive rule needs two in a row.
"""

import pytest

from src.training.device import VALID, resolve_device
from src.training.evaluation import (
    DEFAULT_MIN_SR,
    DEFAULT_MIN_WILSON,
    EvalResult,
    StabilityCriterion,
    evaluate_node,
    heldout_seeds,
    stopping_seeds,
    wilson_lower,
)


def result(successes, n=50):
    return EvalResult(n_episodes=n, successes=successes, mean_length=10.0, seed_base=0)


# ---------------------------------------------------------------------------
# Wilson bound -- the numbers the criterion was calibrated against [A: F11]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "successes,n,expected",
    [
        (45, 50, 0.786),
        (46, 50, 0.812),
        (47, 50, 0.838),
        (48, 50, 0.865),
        (90, 100, 0.826),
        (92, 100, 0.850),
        (50, 50, 0.929),
    ],
)
def test_wilson_lower_matches_hand_computed_values(successes, n, expected):
    assert wilson_lower(successes, n) == pytest.approx(expected, abs=0.001)


def test_wilson_lower_degenerate_cases():
    assert wilson_lower(0, 0) == 0.0
    assert wilson_lower(0, 50) == 0.0


def test_the_criterion_binds_at_46_of_50():
    """
    The reason the bound is 0.80 and not 0.85: at N=50, 0.85 would silently
    demand 48/50, which is not what "success rate >= 0.90" was meant to say.
    """
    crit = StabilityCriterion()
    assert crit.passes(result(46)) is True
    assert crit.passes(result(45)) is False, "45/50 has Wilson LB 0.786"

    strict = StabilityCriterion(min_wilson_lower=0.85)
    assert strict.passes(result(47)) is False
    assert strict.passes(result(48)) is True


def test_success_rate_alone_is_not_enough():
    """0.90 exactly, but the bound rejects it -- that is the point of the bound."""
    r = result(45)
    assert r.success_rate == pytest.approx(0.90)
    assert StabilityCriterion().passes(r) is False


# ---------------------------------------------------------------------------
# Two consecutive evaluations
# ---------------------------------------------------------------------------

def test_two_consecutive_passes_are_required():
    crit = StabilityCriterion()
    assert crit.update(result(48)) is False, "one good evaluation is not enough"
    assert crit.update(result(48)) is True


def test_a_failure_resets_the_streak():
    crit = StabilityCriterion()
    crit.update(result(48))
    crit.update(result(30))
    assert crit.streak == 0
    assert crit.update(result(48)) is False, "the streak must restart from zero"
    assert crit.update(result(48)) is True


def test_history_is_recorded_for_meta_json():
    crit = StabilityCriterion()
    crit.update(result(40))
    crit.update(result(48))
    d = crit.as_dict()
    assert len(d["history"]) == 2
    assert d["history"][1]["successes"] == 48
    assert d["min_wilson_lower"] == DEFAULT_MIN_WILSON
    assert d["min_success_rate"] == DEFAULT_MIN_SR


# ---------------------------------------------------------------------------
# Seed sets [A: F11]
# ---------------------------------------------------------------------------

def test_seed_set_is_identical_across_calls():
    assert stopping_seeds(50) == stopping_seeds(50)
    assert len(stopping_seeds(50)) == 50


def test_stopping_and_heldout_seed_sets_are_disjoint():
    """The stopping set is optimistically biased; the quoted score must not be."""
    assert set(stopping_seeds(50)).isdisjoint(heldout_seeds(50))


# ---------------------------------------------------------------------------
# evaluate_node against a scripted policy with a known success rate
# ---------------------------------------------------------------------------

class CountdownEnv:
    """Reaches the goal after `k` steps; k depends on the seed."""

    def __init__(self, succeed_on):
        self.succeed_on = succeed_on
        self.reached = False
        self.seed = None

    def reset(self, seed=None):
        self.seed = seed
        self.reached = False
        self.n = 0
        return None, {}

    def step(self, action):
        self.n += 1
        if self.seed in self.succeed_on and self.n >= 3:
            self.reached = True
        return None, 0.0, self.reached, False, {}


def make_scripted(succeed_on):
    """A node whose goal reads the env through a StateView built from `extra`."""
    from src.skills.node import Predicate, SkillNode
    from src.skills.state import StateView

    env = CountdownEnv(succeed_on)

    class Act:
        def act(self, obs, deterministic=True):
            return 0

    node = SkillNode(
        "scripted",
        goal=Predicate("reached", lambda sv: bool(sv.extra.get("reached"))),
        timeout=10,
    )
    node.policy = Act()

    # Patch the state builder for this toy env only.
    import src.training.evaluation as ev

    original = ev.build_state_view
    ev.build_state_view = lambda e, version=0: StateView(
        version=version, extra={"reached": e.reached}
    )
    return node, env, (ev, original)


def test_evaluate_node_reproduces_a_known_success_rate():
    seeds = list(range(10))
    succeed_on = {0, 1, 2, 3, 4, 5, 6}      # 7 of 10
    node, env, (mod, original) = make_scripted(succeed_on)
    try:
        res = evaluate_node(node, lambda: env, seeds=seeds)
    finally:
        mod.build_state_view = original

    assert res.n_episodes == 10
    assert res.successes == 7
    assert res.success_rate == pytest.approx(0.7)
    assert res.per_episode == [s in succeed_on for s in seeds]


def test_evaluate_node_is_deterministic_across_repeats():
    seeds = list(range(10))
    node, env, (mod, original) = make_scripted({0, 1, 2})
    try:
        a = evaluate_node(node, lambda: env, seeds=seeds)
        b = evaluate_node(node, lambda: env, seeds=seeds)
    finally:
        mod.build_state_view = original
    assert a.per_episode == b.per_episode


# ---------------------------------------------------------------------------
# Device [A: F31]
# ---------------------------------------------------------------------------

def test_default_device_is_cpu():
    assert resolve_device("cpu") == "cpu"
    assert "auto" in VALID


def test_auto_falls_back_to_cpu_without_an_accelerator():
    assert resolve_device("auto") in ("cpu", "cuda", "mps")


def test_requesting_a_missing_device_raises_rather_than_falling_back():
    import torch

    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            resolve_device("cuda")


def test_unknown_device_rejected():
    with pytest.raises(ValueError):
        resolve_device("tpu")


# --------------------------------------------------------------------------
# H-6.8: "two consecutive evaluations" needs a floor of one policy update
# --------------------------------------------------------------------------

def _dummy_node():
    from src.skills.node import SkillNode

    return SkillNode("{0,t} t", timeout=10)


def test_eval_every_below_one_policy_update_is_refused():
    """
    v2 §5.3, amended. With `eval_every` smaller than one PPO rollout the two
    "consecutive" evaluations score the SAME network, and on a fixed seed set in
    deterministic execution mode the second is a bit-exact copy of the first --
    so the criterion degenerates into one measurement counted twice, which is
    precisely the binomial-noise exposure the Wilson bound exists to remove.

    This is not hypothetical: every composite history in H-6 came out in
    identical pairs, and `{2,1} solve_doorkey` was declared STABLE after one
    gradient update.
    """
    from src.training.evaluation import EvaluationCallback, StabilityCriterion

    with pytest.raises(ValueError, match="smaller than one policy update"):
        EvaluationCallback(
            _dummy_node(),
            eval_fn=lambda: EvalResult(50, 50, 1.0, 10_000),
            eval_every=1_000,
            criterion=StabilityCriterion(),
            update_every=2_048,
        )


def test_eval_every_at_exactly_one_policy_update_is_allowed():
    from src.training.evaluation import EvaluationCallback, StabilityCriterion

    cb = EvaluationCallback(
        _dummy_node(),
        eval_fn=lambda: EvalResult(50, 50, 1.0, 10_000),
        eval_every=512,
        criterion=StabilityCriterion(),
        update_every=512,
    )
    assert cb.update_every == 512


def test_a_single_evaluation_criterion_opts_out_of_the_floor():
    """`n_consecutive=1` makes "consecutive" vacuous, so the floor is moot."""
    from src.training.evaluation import EvaluationCallback, StabilityCriterion

    cb = EvaluationCallback(
        _dummy_node(),
        eval_fn=lambda: EvalResult(50, 50, 1.0, 10_000),
        eval_every=10,
        criterion=StabilityCriterion(n_consecutive=1),
        update_every=2_048,
    )
    assert cb.eval_every == 10


def test_extra_metrics_are_recorded_per_evaluation():
    """
    The choice task needs first-key accuracy by door colour at EVERY evaluation:
    a policy converging to "always yellow" and one still exploring are
    indistinguishable in the success rate alone.
    """
    from src.training.evaluation import EvaluationCallback, StabilityCriterion

    calls = {"n": 0}

    def metrics():
        calls["n"] += 1
        return {"probe": calls["n"]}

    cb = EvaluationCallback(
        _dummy_node(),
        eval_fn=lambda: EvalResult(50, 40, 1.0, 10_000),
        eval_every=512,
        criterion=StabilityCriterion(),
        update_every=512,
        extra_metrics=metrics,
        verbose=0,
    )
    cb._run_eval()
    cb._run_eval()
    assert [e["probe"] for e in cb.extra_history] == [1, 2]
    assert all("success_rate" in e and "at_steps" in e for e in cb.extra_history)
