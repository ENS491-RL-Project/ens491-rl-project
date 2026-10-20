"""
OptionEnv: the SMDP a composite node is trained on.

Target Architecture v2 §5.2. One `step` of this env is one complete execution of
one child: the composite picks a child, that child runs under an `Executor`
until it reaches its goal, times out, or the parent's own goal interrupts it,
and only then does the composite see another observation.

Wrapping order is the mirror of `SubgoalEnv`'s. OptionEnv sits **outside** the
observation pipeline, because it has to feed observations to its frozen
children's policies, and those were trained on the processed 147-dim vector:

    gym.make -> ResetHookEnv -> ImgObs -> Flatten -> ChannelNormalize
             -> OptionEnv -> Monitor

(`SubgoalEnv` sits inside the pipeline instead, because its job is to mutate the
grid before the observation is generated.)
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete

from src.envs.obs import apply_obs_pipeline
from src.envs.subgoal_env import NO_HOOK, ResetHook, UnsuitableLayout
from src.skills.executor import DETERMINISTIC, Executor
from src.skills.node import SkillNode
from src.skills.state import build_state_view

MAX_RESET_TRIES = 50


class ResetHookEnv(gym.Wrapper):
    """
    Applies a reset hook and regenerates the observation.

    Split out of `SubgoalEnv` so a composite can use the same initial-condition
    machinery without inheriting the primitive reward logic.
    """

    def __init__(self, env: gym.Env, reset_hook: ResetHook = NO_HOOK,
                 *, max_reset_tries: int = MAX_RESET_TRIES) -> None:
        super().__init__(env)
        self.reset_hook = reset_hook
        self.max_reset_tries = max_reset_tries

    def reset(self, *, seed=None, options=None):
        _, info = self.env.reset(seed=seed, options=options)
        # Some layouts cannot host the hook's initial conditions at all; draw
        # the next one rather than crash. Same contract as SubgoalEnv.reset, and
        # it has to be here too because evaluation and OptionEnv both come
        # through this wrapper, not through SubgoalEnv.
        for _ in range(self.max_reset_tries):
            try:
                self.reset_hook(self.env)
                break
            except UnsuitableLayout:
                _, info = self.env.reset()
        else:
            raise RuntimeError(
                f'{self.reset_hook.spec}: no suitable layout in '
                f'{self.max_reset_tries} draws'
            )
        return self.env.unwrapped.gen_obs(), info


class OptionEnv(gym.Wrapper):
    """
    The SMDP seen by a composite node.

    action        an index into `node.children`
    reward        the node's OWN goal only [A: F10] -- the env reward is discarded
    terminated    node.goal(s)
    truncated     node timeout, max_decisions, no available child, or env end
    """

    def __init__(
        self,
        env: gym.Env,
        node: SkillNode,
        *,
        gamma: float = 0.99,
        execution_mode: str = DETERMINISTIC,
        max_reset_tries: int = MAX_RESET_TRIES,
        initiation_masking: bool = False,
    ) -> None:
        super().__init__(env)
        if node.is_primitive:
            raise ValueError(f"{node.id}: OptionEnv needs a composite node")
        self.node = node
        self.gamma = gamma
        self.execution_mode = execution_mode
        self.max_reset_tries = max_reset_tries
        self.initiation_masking = initiation_masking   # C1 ablation only [A: F1]

        self.action_space = Discrete(len(node.children))
        self.steps = 0
        self.decisions = 0
        # Cumulative across episodes and never reset [v2 §11, H-6 A.4]. `steps`
        # is per-episode and is what the reward uses; this one is the training
        # cost of the node, which is the number a flat baseline is compared
        # against. Without it a composite's env-step cost is simply unmeasured --
        # `runs/v2/unlock_door.json` has no such field for exactly that reason.
        self.total_env_steps = 0
        self.version = 0
        self.obs: Any = None
        self.sv = None
        self._cache: dict = {}

    # -- masks -----------------------------------------------------------

    def action_masks(self) -> list[bool]:
        """
        [A: F1] Goal-only availability. Consumed by MaskablePPO.

        The cache is keyed by state version, so repeated calls within one
        decision (MaskablePPO asks more than once) cost nothing [A: F4].
        """
        if self._cache.get("__version__") != self.version:
            self._cache = {"__version__": self.version}
        out = []
        for c in self.node.children:
            ok = c.available(self.sv, self._cache)
            if ok and self.initiation_masking:
                ok = bool(c.initiation(self.sv))
            out.append(ok)
        return out

    # -- accounting ------------------------------------------------------

    def get_total_env_steps(self) -> int:
        """
        Env steps consumed by this env since it was created, across episodes.

        A method rather than a bare attribute so `DummyVecEnv.env_method` can
        reach it; see `MaskableMonitor.get_total_env_steps` for why the attribute
        alone is not enough.
        """
        return self.total_env_steps

    # -- gym API ---------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        for _ in range(self.max_reset_tries):
            self.version = 0
            self.sv = build_state_view(self.env, version=0)
            self._cache = {"__version__": 0}
            # [A: F7] goal true at reset would end the episode at step 0 with
            # SUCCESS; [A: F3] an all-False mask cannot be handed to MaskablePPO.
            if not self.node.goal(self.sv) and any(self.action_masks()):
                break
            obs, info = self.env.reset()
        else:
            raise RuntimeError(
                f"{self.node.id}: after {self.max_reset_tries} resets the goal still "
                "held at reset, or no child was ever available"
            )

        self.obs = obs
        self.steps = 0
        self.decisions = 0
        return obs, info

    def step(self, idx):
        idx = int(idx)
        child = self.node.children[idx]
        ex = Executor(child, gamma=self.gamma, execution_mode=self.execution_mode)

        k = 0
        env_term = env_trunc = False
        while True:
            action = ex.act(self.obs, self.sv)
            if action is None:                       # child reached its own end
                break
            # [A: F10] the env's reward never reaches the composite
            self.obs, _, env_term, env_trunc, _ = self.env.step(action)
            self.version += 1
            self.sv = build_state_view(self.env, version=self.version)
            ex.observe_reward(0.0)
            k += 1
            if env_term or env_trunc or self.node.goal(self.sv):
                break

        self.steps += k
        self.total_env_steps += k
        self.decisions += 1

        reached = bool(self.node.goal(self.sv))
        reward = 1.0 - 0.9 * (self.steps / self.node.timeout) if reached else 0.0

        no_child = not any(self.action_masks())      # [A: F3]
        truncated = (not reached) and (
            env_term
            or env_trunc
            or self.steps >= self.node.timeout
            or self.decisions >= self.node.max_decisions   # [A: F2] mandatory
            or no_child
        )

        info = {
            "k": k,                                  # [A: F14] logged from run one
            "child_id": child.id,
            "child_index": idx,
            "child_outcome": ex.last_outcome.value if ex.last_outcome else None,
            "reselections": ex.reselections,
            "decisions": self.decisions,
            "option_steps": self.steps,
            "no_child_available": no_child,
        }
        return self.obs, reward, reached, truncated, info


def make_option_env(
    node: SkillNode,
    env_id: str,
    reset_hook: ResetHook = NO_HOOK,
    *,
    seed: Optional[int] = None,
    normalize: bool = True,
    monitor: bool = True,
    gamma: float = 0.99,
    execution_mode: str = DETERMINISTIC,
    initiation_masking: bool = False,
) -> gym.Env:
    """`gym.make -> ResetHookEnv -> obs pipeline -> OptionEnv -> Monitor`."""
    from stable_baselines3.common.monitor import Monitor

    env = gym.make(env_id)
    env = ResetHookEnv(env, reset_hook)
    env = apply_obs_pipeline(env, normalize=normalize)
    env = OptionEnv(env, node, gamma=gamma, execution_mode=execution_mode,
                    initiation_masking=initiation_masking)
    if monitor:
        env = MaskableMonitor(env)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


class MaskableMonitor(gym.Wrapper):
    """
    `Monitor` plus an explicit `action_masks` passthrough.

    sb3-contrib finds the mask with `venv.env_method("action_masks")`, which
    resolves the attribute on the outermost wrapper. Gymnasium 1.x no longer
    forwards arbitrary attributes through `Wrapper.__getattr__`, so a plain
    `Monitor` on top would hide the method and MaskablePPO would fail.
    """

    def __init__(self, env: gym.Env) -> None:
        from stable_baselines3.common.monitor import Monitor

        super().__init__(Monitor(env))
        self._option_env = env

    def action_masks(self) -> list[bool]:
        return self._option_env.action_masks()

    def get_total_env_steps(self) -> int:
        """Same passthrough reason as `action_masks` [H-6 A.4]."""
        return self._option_env.get_total_env_steps()
