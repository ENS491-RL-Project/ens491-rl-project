
"""Validate the pilot GRU task-identification dataset."""

from pathlib import Path

import numpy as np

DATASET_PATH = Path("data/gru/pilot_dataset.npz")
SEQ_LEN = 20
OBS_DIM = 147
SEED = 42


def main():
    data = np.load(DATASET_PATH)

    X = data["X"]
    y = data["y"]
    episode_ids = data["episode_ids"]
    env_names = data["env_names"]

    print("=== 1. DATASET OVERVIEW ===")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Episode IDs shape:", episode_ids.shape)

    assert X.ndim == 3
    assert X.shape[1:] == (SEQ_LEN, OBS_DIM)
    assert len(X) == len(y) == len(episode_ids)
    assert len(X) > 0

    print("\n=== 2. TASK DISTRIBUTION ===")

    for task_id, env_name in enumerate(env_names):
        mask = y == task_id

        n_sequences = int(mask.sum())
        n_episodes = len(np.unique(episode_ids[mask]))

        print(
            f"Task {task_id}: {env_name}\n"
            f"  Episodes: {n_episodes}\n"
            f"  Sequences: {n_sequences}"
        )

    print("\n=== 3. NORMALIZATION CHECK ===")

    min_value = float(X.min())
    max_value = float(X.max())

    print("Minimum observation value:", min_value)
    print("Maximum observation value:", max_value)

    assert np.isfinite(X).all(), "Dataset contains NaN or Inf."
    assert min_value >= -1e-6, "Values below zero detected."
    assert max_value <= 1.0 + 1e-6, "Values above one detected."

    print("Normalization check: PASSED")

    print("\n=== 4. OBSERVATION SIMILARITY ===")

    # Use one representative observation per episode.
    # This avoids counting heavily overlapping windows as independent data.
    episode_vectors = {}
    episode_labels = {}

    for episode_id in np.unique(episode_ids):
        indices = np.flatnonzero(episode_ids == episode_id)
        first_idx = indices[0]

        # First observation of the first sequence in the episode.
        episode_vectors[int(episode_id)] = X[first_idx, 0]
        episode_labels[int(episode_id)] = int(y[first_idx])

    task_vectors = {}

    for task_id in range(len(env_names)):
        vectors = [
            episode_vectors[eid]
            for eid in episode_vectors
            if episode_labels[eid] == task_id
        ]
        task_vectors[task_id] = np.stack(vectors)

    print("Mean pairwise Euclidean distance between initial observations")
    print("(smaller distance means more similar observations):")

    for i in range(len(env_names)):
        for j in range(i + 1, len(env_names)):
            A = task_vectors[i]
            B = task_vectors[j]

            distances = np.linalg.norm(
                A[:, None, :] - B[None, :, :],
                axis=-1,
            )

            print(
                f"Task {i} vs Task {j}: "
                f"{distances.mean():.4f}"
            )

    print(
        "\nNote: These distances describe initial observations only. "
        "They do not prove whether full sequences are classifiable."
    )

    print("\n=== 5. EPISODE-LEVEL SPLIT CHECK ===")

    rng = np.random.default_rng(SEED)

    train_ids = []
    val_ids = []
    test_ids = []

    for task_id in range(len(env_names)):
        task_episodes = np.array(
            [
                eid for eid in episode_vectors
                if episode_labels[eid] == task_id
            ],
            dtype=np.int64,
        )

        rng.shuffle(task_episodes)

        n = len(task_episodes)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)

        train_ids.extend(task_episodes[:n_train])
        val_ids.extend(task_episodes[n_train:n_train + n_val])
        test_ids.extend(task_episodes[n_train + n_val:])

    train_ids = np.asarray(train_ids)
    val_ids = np.asarray(val_ids)
    test_ids = np.asarray(test_ids)

    assert not np.intersect1d(train_ids, val_ids).size
    assert not np.intersect1d(train_ids, test_ids).size
    assert not np.intersect1d(val_ids, test_ids).size

    assert set(np.concatenate([train_ids, val_ids, test_ids])) == set(
        np.unique(episode_ids)
    )

    for name, ids in [
        ("Train", train_ids),
        ("Validation", val_ids),
        ("Test", test_ids),
    ]:
        mask = np.isin(episode_ids, ids)

        print(
            f"{name}: {len(ids)} episodes, "
            f"{int(mask.sum())} sequences"
        )

        counts = np.bincount(y[mask], minlength=len(env_names))
        print("  Sequences per task:", counts.tolist())

    print("\nEpisode-level split check: PASSED")

    print("\n=== VALIDATION COMPLETED ===")


if __name__ == "__main__":
    main()
