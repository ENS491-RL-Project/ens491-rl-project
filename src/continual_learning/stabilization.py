from collections import deque
import statistics


class StabilizationMonitor:
    """Tracks episode rewards and signals when a column has stabilised."""

    WINDOW = 50
    REWARD_THRESHOLD = 0.85
    STD_THRESHOLD = 0.05

    def __init__(self) -> None:
        self._rewards: deque[float] = deque(maxlen=self.WINDOW)

    def update(self, episode_reward: float) -> None:
        self._rewards.append(episode_reward)

    def is_stable(self) -> bool:
        if len(self._rewards) < self.WINDOW:
            return False
        mean = statistics.mean(self._rewards)
        std = statistics.stdev(self._rewards)
        return mean >= self.REWARD_THRESHOLD and std <= self.STD_THRESHOLD

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
