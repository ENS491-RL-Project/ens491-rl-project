from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401 — registers MiniGrid envs
from gymnasium.wrappers import FlattenObservation
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.continual_learning.column import Column
from src.continual_learning.column_policy import ColumnPolicy
from src.continual_learning.stabilization import StabilizationMonitor


class _EpisodeRewardCallback(BaseCallback):
    """Feeds episode rewards into a StabilizationMonitor after each episode ends."""

    def __init__(self, monitor: StabilizationMonitor) -> None:
        super().__init__()
        self._monitor = monitor

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self._monitor.update(ep["r"])
        return True


class ColumnTrainer:
    """
    Trains a single Column using SB3 PPO with ColumnPolicy (with optional laterals).

    After training, the trained policy is attached to column.policy so that
    column.forward() works and column.freeze() can be called.
    """

    def __init__(
        self,
        column: Column,
        env_id: str,
        total_timesteps: int = 100_000,
        verbose: int = 0,
    ) -> None:
        self.column = column
        self.env_id = env_id
        self.total_timesteps = total_timesteps
        self.verbose = verbose
        self._monitor = StabilizationMonitor()

    def _make_env(self) -> gym.Env:
        # ImgObsWrapper extracts the 7×7×3 image; FlattenObservation gives (147,) uint8 obs.
        # This matches the obs_dim=147 used throughout the codebase.
        env = gym.make(self.env_id)
        return FlattenObservation(ImgObsWrapper(env))

    def train(self) -> None:
        env = self._make_env()
        policy_kwargs = {"lateral_source_column": self.column.lateral_source}
        model = PPO(
            ColumnPolicy,
            env,
            policy_kwargs=policy_kwargs,
            verbose=self.verbose,
        )
        callback = _EpisodeRewardCallback(self._monitor)
        model.learn(total_timesteps=self.total_timesteps, callback=callback)
        self.column.policy = model.policy
        env.close()

    def is_stable(self) -> bool:
        return self._monitor.is_stable()

    @property
    def monitor(self) -> StabilizationMonitor:
        return self._monitor
