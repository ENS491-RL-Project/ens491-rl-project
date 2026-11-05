
"""Collect a pilot GRU task-identification dataset from MiniGrid."""

from pathlib import Path

import numpy as np

from src.envs.obs import make_base_env, OBS_DIM


TASK_ENVS = {
    0: "MiniGrid-Unlock-v0",
    1: "MiniGrid-DoorKey-6x6-v0",
    2: "MiniGrid-UnlockPickup-v0",
    3: "MiniGrid-BlockedUnlockPickup-v0",
}

SEQ_LEN = 20
EPISODES_PER_TASK = 20
BASE_SEED = 42

OUTPUT_PATH = Path("data/gru/pilot_dataset.npz")


def collect_episode(env, seed, rng):
    """Collect observations from one episode using random actions."""
    obs, _ = env.reset(seed=seed)
    observations = [np.asarray(obs, dtype=np.float32).copy()]

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = int(rng.integers(env.action_space.n))
        obs, _, terminated, truncated, _ = env.step(action)
        observations.append(np.asarray(obs, dtype=np.float32).copy())

    return np.stack(observations)


def make_sequences(observations, seq_len):
    """Create non-overlapping-boundary, fixed-length sequences."""
    sequences = []

    for start in range(len(observations) - seq_len + 1):
        sequence = observations[start:start + seq_len]
        sequences.append(sequence)

    return sequences


def main():
    rng = np.random.default_rng(BASE_SEED)

    all_sequences = []
    all_labels = []
    all_episode_ids = []

    episode_id = 0

    for task_id, env_id in TASK_ENVS.items():
        print(f"\nCollecting task {task_id}: {env_id}")

        env = make_base_env(env_id, normalize=True, monitor=False)

        try:
            for ep in range(EPISODES_PER_TASK):
                seed = BASE_SEED + task_id * 10_000 + ep

                observations = collect_episode(env, seed, rng)

                assert observations.shape[1] == OBS_DIM
                assert np.isfinite(observations).all()
                assert observations.min() >= 0.0
                assert observations.max() <= 1.0

                sequences = make_sequences(observations, SEQ_LEN)

                all_sequences.extend(sequences)
                all_labels.extend([task_id] * len(sequences))
                all_episode_ids.extend([episode_id] * len(sequences))

                print(
                    f"  Episode {ep + 1:02d}: "
                    f"{len(observations)} observations, "
                    f"{len(sequences)} sequences"
                )

                episode_id += 1

        finally:
            env.close()

    X = np.asarray(all_sequences, dtype=np.float32)
    y = np.asarray(all_labels, dtype=np.int64)
    episode_ids = np.asarray(all_episode_ids, dtype=np.int64)

    assert X.ndim == 3
    assert X.shape[1:] == (SEQ_LEN, OBS_DIM)
    assert len(X) == len(y) == len(episode_ids)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT_PATH,
        X=X,
        y=y,
        episode_ids=episode_ids,
        env_names=np.asarray(list(TASK_ENVS.values())),
        seq_len=np.asarray(SEQ_LEN),
    )

    print("\nDataset saved:", OUTPUT_PATH)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Episode IDs shape:", episode_ids.shape)

    for task_id, env_id in TASK_ENVS.items():
        count = int(np.sum(y == task_id))
        print(f"Task {task_id} ({env_id}): {count} sequences")


if __name__ == "__main__":
    main()
