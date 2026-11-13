
"""Evaluate a saved PPO-GRU with window and episode-balanced metrics.

Run from project root:
    python -m scripts.evaluate_gru_ppo --seq-len 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.train_gru_ppo import (
    POLICIES,
    WindowDataset,
    collate,
    load_episodes,
    logits_for,
)
from src.task_detection.gru_identifier import GRUTaskIdentifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/gru/ppo_pilot"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/gru_ppo_pilot"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    model_path = (
        args.model_dir / f"gru_seq{args.seq_len}_best.pt"
    )
    split_path = (
        args.model_dir / f"gru_seq{args.seq_len}_splits.json"
    )

    split_info = json.loads(
        split_path.read_text(encoding="utf-8")
    )

    episodes = load_episodes(args.data_dir)

    episode_lookup = {
        (e["policy"], int(e["episode_id"])): i
        for i, e in enumerate(episodes)
    }

    test_indices = [
        episode_lookup[(e["policy"], int(e["episode_id"]))]
        for e in split_info["test"]
    ]

    assert len(test_indices) == len(set(test_indices))

    for row, idx in zip(split_info["test"], test_indices):
        assert int(row["seed"]) == episodes[idx]["seed"]
        assert bool(row["success"]) == episodes[idx]["success"]

    dataset = WindowDataset(
        episodes,
        test_indices,
        args.seq_len,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["seq_len"] == args.seq_len
    assert checkpoint["policy_names"] == list(POLICIES)

    model = GRUTaskIdentifier(
        obs_dim=checkpoint["obs_dim"],
        hidden_size=checkpoint["hidden_size"],
        num_tasks=checkpoint["num_tasks"],
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    predictions = []

    with torch.no_grad():
        for x, lengths, _ in loader:
            logits = logits_for(
                model,
                x.to(device),
                lengths,
            )

            predictions.extend(
                logits.argmax(dim=1)
                .cpu()
                .numpy()
                .tolist()
            )

    assert len(predictions) == len(dataset.windows)

    # Calculate window correctness within each episode.
    correct_by_episode = {
        idx: [] for idx in test_indices
    }

    for (ep_idx, _), prediction in zip(
        dataset.windows,
        predictions,
    ):
        correct = int(
            prediction == episodes[ep_idx]["label"]
        )

        correct_by_episode[ep_idx].append(correct)

    # Average accuracy within each episode.
    episode_accuracy = {
        idx: float(np.mean(scores))
        for idx, scores in correct_by_episode.items()
    }

    # Calculate episode-balanced accuracy per PPO.
    per_policy = {}

    for label, policy in enumerate(POLICIES):
        indices = [
            idx
            for idx in test_indices
            if episodes[idx]["label"] == label
        ]

        assert indices

        per_policy[policy] = float(
            np.mean([
                episode_accuracy[idx]
                for idx in indices
            ])
        )

    # Metric 1: Overall window-level accuracy.
    window_accuracy = float(
        np.mean([
            prediction == episodes[ep_idx]["label"]
            for (ep_idx, _), prediction in zip(
                dataset.windows,
                predictions,
            )
        ])
    )

    # Metric 2: Macro accuracy across PPO classes.
    window_macro = float(
        np.mean([
            np.mean([
                prediction == label
                for (ep_idx, _), prediction in zip(
                    dataset.windows,
                    predictions,
                )
                if episodes[ep_idx]["label"] == label
            ])
            for label in range(len(POLICIES))
        ])
    )

    # Metric 3: Equal weight for every episode.
    balanced = float(
        np.mean(list(episode_accuracy.values()))
    )

    # Metric 4: Equal weight for episodes and PPO classes.
    policy_episode_macro = float(
        np.mean(list(per_policy.values()))
    )

    print(
        f"Device: {device} | "
        f"sequence length: {args.seq_len}"
    )

    print(
        f"Test: {len(test_indices)} episodes, "
        f"{len(dataset.windows)} windows"
    )

    print(
        f"Window-level accuracy:          "
        f"{window_accuracy:.4f}"
    )

    print(
        f"Window macro accuracy:          "
        f"{window_macro:.4f}"
    )

    print(
        f"Episode-balanced accuracy:      "
        f"{balanced:.4f}"
    )

    print(
        f"Policy-episode macro accuracy:  "
        f"{policy_episode_macro:.4f}"
    )

    print("\nPer-policy episode-balanced accuracy:")

    for policy, value in per_policy.items():
        print(f"  {policy:22s} {value:.4f}")

    print(
        "\nNote: this averages window correctness "
        "within each episode."
    )
    print(
        "It is not majority-vote episode "
        "classification accuracy."
    )


if __name__ == "__main__":
    main()
