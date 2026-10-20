"""
DEPRECATED (H-3, 2026-10-01) -- superseded by `src/training/evaluation.py`.

Target Architecture v2 §5.3 retires the reward mean/std criterion: MiniGrid's
reward depends on path length, so in an env with random start and goal positions
a policy that succeeds every time can still have reward std above 0.05, which
makes the old rule unsatisfiable regardless of how well the policy has learned.
The replacement is a success-rate criterion with a Wilson lower bound on a fixed
evaluation seed set.

This module is kept, not deleted, because `ColumnTrainer` imports it and
`src/continual_learning/__init__.py` re-exports it; H-3 itself needs
`ColumnTrainer.load_policy_into` to re-evaluate the legacy `col01` column.
Delete it together with `ColumnTrainer` (v2 §10).

Do not use it for new nodes.
"""

from collections import deque
import statistics


class StabilizationMonitor:
    """Tracks episode rewards and signals when a column has stabilised."""

    WINDOW = 50

    def __init__(
        self,
        reward_threshold: float = 0.85,
        std_threshold: float = 0.05,
    ) -> None:
        self._rewards: deque[float] = deque(maxlen=self.WINDOW)
        self.reward_threshold = reward_threshold
        self.std_threshold = std_threshold

    def update(self, episode_reward: float) -> None:
        self._rewards.append(episode_reward)

    def is_stable(self) -> bool:
        if len(self._rewards) < self.WINDOW:
            return False
        mean = statistics.mean(self._rewards)
        std = statistics.stdev(self._rewards)
        return mean >= self.reward_threshold and std <= self.std_threshold

    def reset(self) -> None:
        self._rewards.clear()

    @property
    def mean_reward(self) -> float:
        if not self._rewards:
            return 0.0
        return statistics.mean(self._rewards)

    @property
    def n_episodes(self) -> int:
        return len(self._rewards)
