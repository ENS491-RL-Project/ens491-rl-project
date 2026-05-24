from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import gymnasium as gym
import minigrid  # noqa: F401 — registers MiniGrid envs
import torch
from gymnasium.wrappers import FlattenObservation
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.continual_learning.column import Column
from src.continual_learning.column_policy import ColumnPolicy
from src.continual_learning.stabilization import StabilizationMonitor


class _EpisodeRewardCallback(BaseCallback):
    """Feeds episode rewards into a StabilizationMonitor after each episode ends."""

    def __init__(self, monitor: StabilizationMonitor) -> None:
        super().__init__()
        self._monitor = monitor
        self.total_episodes = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self._monitor.update(ep["r"])
                self.total_episodes += 1
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
        reward_threshold: float = 0.85,
        std_threshold: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.column = column
        self.env_id = env_id
        self.total_timesteps = total_timesteps
        self.verbose = verbose
        self.seed = seed
        self._monitor = StabilizationMonitor(
            reward_threshold=reward_threshold,
            std_threshold=std_threshold,
        )
        self._model: Optional[PPO] = None

    def _make_env(self) -> gym.Env:
        # ImgObsWrapper extracts the 7×7×3 image; FlattenObservation gives (147,) uint8 obs.
        # Monitor is explicit here — do not rely on SB3 auto-wrapping for episode info.
        env = gym.make(self.env_id)
        env = FlattenObservation(ImgObsWrapper(env))
        env = Monitor(env)
        # Seed the env for controlled-seed reproducibility (not bitwise deterministic).
        env.reset(seed=self.seed)
        env.action_space.seed(self.seed)
        env.observation_space.seed(self.seed)
        return env

    def _init_model(self, env: gym.Env) -> None:
        if self._model is None:
            policy_kwargs = {"lateral_source_column": self.column.lateral_source}
            self._model = PPO(
                ColumnPolicy, env,
                policy_kwargs=policy_kwargs,
                verbose=self.verbose,
                seed=self.seed,
            )

    def train_until_stable(
        self,
        chunk_size: int,
        max_total: int,
        log_every: int = 1,
    ) -> None:
        """Train in chunks of chunk_size steps, stopping when stable or max_total is reached."""
        env = self._make_env()
        self._init_model(env)
        callback = _EpisodeRewardCallback(self._monitor)

        trained = 0
        while trained < max_total:
            actual_chunk = min(chunk_size, max_total - trained)
            self._model.learn(
                total_timesteps=actual_chunk,
                callback=callback,
                reset_num_timesteps=(trained == 0),
            )
            trained += actual_chunk

            stable = self._monitor.is_stable()
            mean = self._monitor.mean_reward
            print(f"[{trained}] episodes={callback.total_episodes}  mean_reward={mean:.2f}  stable={stable}")

            if stable:
                print(f"[{trained}] STABLE  mean={mean:.3f}")
                break
        else:
            print(
                f"WARNING: max_total={max_total} reached without stabilization. "
                f"mean={self._monitor.mean_reward:.3f}"
            )

        self.column.policy = self._model.policy
        env.close()

    def train(self) -> None:
        """Backward-compatible single-shot training (one chunk equal to total_timesteps)."""
        self.train_until_stable(chunk_size=self.total_timesteps, max_total=self.total_timesteps)

    def is_stable(self) -> bool:
        return self._monitor.is_stable()

    @property
    def monitor(self) -> StabilizationMonitor:
        return self._monitor

    def save(self, run_dir) -> None:
        """
        Save three artifacts to run_dir:
          - model.zip  (full SB3 model: weights, optimizer state, hyperparams)
          - policy_state_dict.pt  (raw PyTorch state dict for lightweight reload)
          - meta.json  (column identity and frozen status)

        Works regardless of is_stable() — partial saves are valid.
        """
        if self._model is None:
            raise RuntimeError("No model to save. Call train() or train_until_stable() first.")
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self._model.save(run_dir / "model")  # SB3 appends .zip → model.zip
        torch.save(self._model.policy.state_dict(), run_dir / "policy_state_dict.pt")
        meta = {
            "n": self.column.n,
            "m": self.column.m,
            "frozen": self.column.frozen,
            "seed": self.seed,
            "env_id": self.env_id,
            "reward_threshold": self._monitor.reward_threshold,
            "std_threshold": self._monitor.std_threshold,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    @staticmethod
    def load_policy_into(column: Column, run_dir, env_id: str) -> None:
        """Load a saved policy state dict into a Column's policy slot."""
        run_dir = Path(run_dir)
        env = gym.make(env_id)
        env = FlattenObservation(ImgObsWrapper(env))
        env = Monitor(env)
        policy_kwargs = {"lateral_source_column": column.lateral_source}
        model = PPO(ColumnPolicy, env, policy_kwargs=policy_kwargs, verbose=0)
        state_dict = torch.load(
            run_dir / "policy_state_dict.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.policy.load_state_dict(state_dict)
        column.policy = model.policy
        env.close()
