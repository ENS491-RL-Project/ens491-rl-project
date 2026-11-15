"""The GRU task-identification pipeline (scripts/train_gru_task.py) on synthetic rollouts."""
import numpy as np
import pytest
import torch

from scripts.train_gru_task import (
    POLICIES, POLICY_TO_TASK, TASKS, WindowDataset, collate, load_episodes, split_episodes,
)

OBS_DIM = 147


def _write_rollouts(directory, n_episodes=10, seed=0):
    """One .npz per policy in the layout collect_gru_ppo_rollouts.py writes."""
    rng = np.random.default_rng(seed)
    for policy in POLICIES:
        lengths = rng.integers(1, 8, size=n_episodes)
        obs, ids = [], []
        for ep, length in enumerate(lengths):
            obs.append(rng.random((length + 1, OBS_DIM), dtype=np.float32))
            ids.extend([ep] * (length + 1))
        np.savez_compressed(
            directory / f"{policy}.npz",
            observations=np.concatenate(obs),
            episode_ids=np.asarray(ids, dtype=np.int64),
            episode_lengths=lengths.astype(np.int64),
            episode_seeds=np.arange(n_episodes, dtype=np.int64) + 20000,
            episode_successes=np.ones(n_episodes, dtype=bool),
        )


def test_every_policy_maps_to_a_task():
    assert set(POLICY_TO_TASK) == set(POLICIES)
    assert set(POLICY_TO_TASK.values()) == set(range(len(TASKS)))


def test_splits_are_disjoint_and_cover_every_task(tmp_path):
    _write_rollouts(tmp_path)
    episodes = load_episodes(tmp_path)
    assert len(episodes) == 10 * len(POLICIES)

    splits = split_episodes(episodes, seed=42)
    parts = [set(splits[name]) for name in ("train", "val", "test")]
    assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) and not (parts[1] & parts[2])
    assert set().union(*parts) == set(range(len(episodes)))
    for part in parts:
        assert {episodes[i]["label"] for i in part} == set(range(len(TASKS)))


@pytest.mark.parametrize("seq_len", [1, 5, 20])
def test_windows_end_at_every_step_and_never_exceed_seq_len(tmp_path, seq_len):
    _write_rollouts(tmp_path)
    episodes = load_episodes(tmp_path)
    indices = list(range(len(episodes)))
    ds = WindowDataset(episodes, indices, seq_len)
    assert len(ds) == sum(len(e["observations"]) for e in episodes)

    for k in range(len(ds)):
        seq, label = ds[k]
        ep, end = ds.windows[k]
        assert seq.shape == (min(seq_len, end), OBS_DIM)
        assert torch.equal(seq, torch.from_numpy(episodes[ep]["observations"][end - len(seq):end]))
        assert label == episodes[ep]["label"]


def test_collate_pads_to_the_longest_window(tmp_path):
    _write_rollouts(tmp_path)
    episodes = load_episodes(tmp_path)
    ds = WindowDataset(episodes, list(range(len(episodes))), seq_len=5)
    batch = [ds[k] for k in range(8)]
    x, lengths, labels = collate(batch)

    assert x.shape == (8, int(lengths.max()), OBS_DIM)
    assert lengths.tolist() == [len(seq) for seq, _ in batch]
    assert labels.tolist() == [label for _, label in batch]
    for i, (seq, _) in enumerate(batch):
        assert torch.equal(x[i, :len(seq)], seq)
        assert not x[i, len(seq):].any()
