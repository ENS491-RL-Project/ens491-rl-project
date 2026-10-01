"""
EvaluationMonitor: the v2 stabilization criterion.

Target Architecture v2 §5.3, as amended.

    Every `eval_every` steps, run N=50 episodes in execution mode on a FIXED seed
    set. A node is STABLE when, in two consecutive evaluations, the success rate
    is >= 0.90 AND its Wilson 95% lower bound is >= 0.80.

[A: F11] Both halves of that matter. With freshly drawn eval seeds the rule is
selection on noise (sigma ~ 0.042 at p=0.90, N=50) and stops on a lucky pair, so
the seed set is fixed. The Wilson bound then makes the stop robust to the
remaining binomial noise: at N=50 it binds at 46/50. The originally proposed
bound of 0.85 would silently have demanded 48/50, which is not what "0.90" means.

The score quoted for a node is the one on a SECOND seed set that is never used
for stopping -- the stopping set is optimistically biased by construction.

This replaces `StabilizationMonitor` (reward mean > 0.85, std < 0.05), which
MiniGrid's path-length-dependent reward can make unsatisfiable regardless of how
well a policy has learned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

from stable_baselines3.common.callbacks import BaseCallback

from src.skills.executor import DETERMINISTIC, Executor
from src.skills.node import SkillNode
from src.skills.state import build_state_view

DEFAULT_N_EVAL = 50
DEFAULT_MIN_SR = 0.90
DEFAULT_MIN_WILSON = 0.80
DEFAULT_N_CONSECUTIVE = 2

# Disjoint by construction: the stopping set must never overlap the held-out set.
STOPPING_SEED_BASE = 10_000
HELDOUT_SEED_BASE = 20_000


def fixed_seeds(n: int = DEFAULT_N_EVAL, base: int = STOPPING_SEED_BASE) -> list[int]:
    """The same seeds at every evaluation -- that is the whole point."""
    return list(range(base, base + n))


def stopping_seeds(n: int = DEFAULT_N_EVAL) -> list[int]:
    return fixed_seeds(n, STOPPING_SEED_BASE)


def heldout_seeds(n: int = DEFAULT_N_EVAL) -> list[int]:
    return fixed_seeds(n, HELDOUT_SEED_BASE)


def wilson_lower(successes: int, n: int, z: float = 1.959963985) -> float:
    """Wilson score interval, lower bound. 0.0 for n == 0."""
    if n <= 0:
        return 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half)


@dataclass
class EvalResult:
    n_episodes: int
    successes: int
    mean_length: float
    seed_base: int
    per_episode: list[bool] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    timeouts: int = 0
    reselections: int = 0

    @property
    def success_rate(self) -> float:
        return self.successes / self.n_episodes if self.n_episodes else 0.0

    @property
    def wilson_lower(self) -> float:
        return wilson_lower(self.successes, self.n_episodes)

    def as_dict(self) -> dict:
        return {
            "n_episodes": self.n_episodes,
            "successes": self.successes,
            "success_rate": round(self.success_rate, 4),
            "wilson_lower": round(self.wilson_lower, 4),
            "mean_length": round(self.mean_length, 2),
            "seed_base": self.seed_base,
            "timeouts": self.timeouts,
            "reselections": self.reselections,
        }

    def __str__(self) -> str:
        return (
            f"SR {self.success_rate:.3f} ({self.successes}/{self.n_episodes}), "
            f"Wilson LB {self.wilson_lower:.3f}, mean length {self.mean_length:.1f}"
        )


def run_episode(
    node: SkillNode,
    env,
    seed: int,
    *,
    gamma: float = 0.99,
    execution_mode: str = DETERMINISTIC,
    max_env_steps: Optional[int] = None,
    prepare: Optional[Callable] = None,
    initiation_masking: bool = False,
) -> tuple[bool, int, Executor]:
    """
    One evaluation episode, driven through the Executor (v2 §5.3: 'execution mode').

    `prepare(env, obs) -> obs` runs after reset and before the node starts. It is
    how a node is scored from **parent-reachable** states rather than only from
    its own reset hook [A: F29]: pass a function that rolls an earlier frozen
    node forward to the state a parent would hand this one.
    """
    obs, _ = env.reset(seed=seed)
    if prepare is not None:
        obs = prepare(env, obs)
    version = 0
    sv = build_state_view(env, version)
    ex = Executor(node, gamma=gamma, execution_mode=execution_mode,
                  initiation_masking=initiation_masking)

    limit = max_env_steps if max_env_steps is not None else node.timeout
    steps = 0
    while steps < limit:
        action = ex.act(obs, sv)
        if action is None:
            break
        obs, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        version += 1
        sv = build_state_view(env, version)
        ex.observe_reward(reward)
        if terminated or truncated:
            break

    return bool(node.goal(sv)), steps, ex


def evaluate_node(
    node: SkillNode,
    env_factory: Callable[[], object],
    *,
    seeds: Optional[Sequence[int]] = None,
    n_episodes: int = DEFAULT_N_EVAL,
    seed_base: int = STOPPING_SEED_BASE,
    gamma: float = 0.99,
    execution_mode: str = DETERMINISTIC,
    max_env_steps: Optional[int] = None,
    prepare: Optional[Callable] = None,
    initiation_masking: bool = False,
) -> EvalResult:
    """
    Run `n_episodes` on a fixed seed set and report the success rate.

    `env_factory()` must return a fresh env whose observation pipeline matches
    what the node was trained with -- the normalization flag is part of the
    node's spec_hash for exactly this reason.
    """
    seed_list = list(seeds) if seeds is not None else fixed_seeds(n_episodes, seed_base)
    env = env_factory()
    try:
        outcomes, lengths, timeouts, reselections = [], [], 0, 0
        for seed in seed_list:
            ok, steps, ex = run_episode(
                node, env, seed,
                gamma=gamma, execution_mode=execution_mode,
                max_env_steps=max_env_steps, prepare=prepare,
                initiation_masking=initiation_masking,
            )
            outcomes.append(ok)
            lengths.append(steps)
            reselections += ex.reselections
            if not ok:
                timeouts += 1
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    return EvalResult(
        n_episodes=len(seed_list),
        successes=sum(outcomes),
        mean_length=(sum(lengths) / len(lengths)) if lengths else 0.0,
        seed_base=seed_list[0] if seed_list else seed_base,
        per_episode=outcomes,
        lengths=lengths,
        timeouts=timeouts,
        reselections=reselections,
    )


class StabilityCriterion:
    """SR >= min_success_rate AND Wilson LB >= min_wilson_lower, N times running."""

    def __init__(
        self,
        min_success_rate: float = DEFAULT_MIN_SR,
        min_wilson_lower: float = DEFAULT_MIN_WILSON,
        n_consecutive: int = DEFAULT_N_CONSECUTIVE,
    ) -> None:
        self.min_success_rate = min_success_rate
        self.min_wilson_lower = min_wilson_lower
        self.n_consecutive = n_consecutive
        self.streak = 0
        self.history: list[EvalResult] = []

    def passes(self, result: EvalResult) -> bool:
        return (
            result.success_rate >= self.min_success_rate
            and result.wilson_lower >= self.min_wilson_lower
        )

    def update(self, result: EvalResult) -> bool:
        """Record an evaluation; return True once the node is STABLE."""
        self.history.append(result)
        self.streak = self.streak + 1 if self.passes(result) else 0
        return self.is_stable

    @property
    def is_stable(self) -> bool:
        return self.streak >= self.n_consecutive

    def as_dict(self) -> dict:
        return {
            "min_success_rate": self.min_success_rate,
            "min_wilson_lower": self.min_wilson_lower,
            "n_consecutive": self.n_consecutive,
            "streak": self.streak,
            "is_stable": self.is_stable,
            "history": [r.as_dict() for r in self.history],
        }


class EvaluationCallback(BaseCallback):
    """
    Evaluate every `eval_every` steps and stop training once STABLE.

    Replaces `_EpisodeRewardCallback`, which fed training-episode rewards into
    the retired mean/std monitor. Evaluation episodes are separate from training
    episodes and run deterministically on the fixed seed set.
    """

    def __init__(
        self,
        node: SkillNode,
        env_factory: Optional[Callable[[], object]] = None,
        *,
        eval_fn: Optional[Callable[[], EvalResult]] = None,
        eval_every: int = 10_000,
        n_episodes: int = DEFAULT_N_EVAL,
        criterion: Optional[StabilityCriterion] = None,
        execution_mode: str = DETERMINISTIC,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        if env_factory is None and eval_fn is None:
            raise ValueError("EvaluationCallback needs env_factory or eval_fn")
        self.node = node
        self.env_factory = env_factory
        self.eval_fn = eval_fn
        self.eval_every = eval_every
        self.n_episodes = n_episodes
        self.criterion = criterion or StabilityCriterion()
        self.execution_mode = execution_mode
        self._next_eval = eval_every

    def _run_eval(self) -> EvalResult:
        if self.eval_fn is not None:
            result = self.eval_fn()
        else:
            result = evaluate_node(
                self.node, self.env_factory,
                n_episodes=self.n_episodes, execution_mode=self.execution_mode,
            )
        stable = self.criterion.update(result)
        if self.verbose:
            print(
                f"  [{self.num_timesteps:>8}] {self.node.id}: {result}"
                f" | streak {self.criterion.streak}/{self.criterion.n_consecutive}"
                f"{' -> STABLE' if stable else ''}",
                flush=True,
            )
        if self.logger is not None:
            self.logger.record("eval/success_rate", result.success_rate)
            self.logger.record("eval/wilson_lower", result.wilson_lower)
            self.logger.record("eval/mean_length", result.mean_length)
        return result

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            self._next_eval += self.eval_every
            self._run_eval()
            if self.criterion.is_stable:
                return False   # stop training
        return True
