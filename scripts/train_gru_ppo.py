"""Train a GRU to identify frozen PPO policies from observation sequences.

Run from the project root:
    python -m scripts.train_gru_ppo --seq-len 10
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.task_detection.gru_identifier import GRUTaskIdentifier

POLICIES = (
    "get_key", "open_door", "go_to_goal", "get_key_yellow",
    "get_key_blue", "open_door_tk", "drop_held",
    "flat_unlock_ppo", "flat_doorkey_ppo", "unlock_door", "solve_doorkey",
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_episodes(data_dir):
    episodes = []
    for label, policy in enumerate(POLICIES):
        path = data_dir / f"{policy}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing PPO dataset: {path}")
        with np.load(path, allow_pickle=False) as d:
            obs = d["observations"]
            lengths = d["episode_lengths"]
            ids = d["episode_ids"]
            seeds = d["episode_seeds"]
            successes = d["episode_successes"]
            assert obs.ndim == 2 and obs.shape[1] == 147
            assert len(ids) == len(obs)
            assert len(seeds) == len(successes) == len(lengths)
            assert len(obs) == int(lengths.sum()) + len(lengths)
            assert np.array_equal(np.bincount(ids, minlength=len(lengths)), lengths + 1)
            assert np.isfinite(obs).all()
            offset = 0
            for ep_id, length in enumerate(lengths):
                count = int(length) + 1
                seq = obs[offset:offset + count].copy()
                assert len(seq) == count and np.all(ids[offset:offset + count] == ep_id)
                episodes.append({
                    "policy": policy, "label": label, "episode_id": ep_id,
                    "seed": int(seeds[ep_id]), "success": bool(successes[ep_id]),
                    "observations": seq,
                })
                offset += count
            assert offset == len(obs)
    return episodes


def split_episodes(episodes, seed):
    rng = np.random.default_rng(seed)
    result = {"train": [], "val": [], "test": []}
    for label in range(len(POLICIES)):
        indices = np.array([i for i, e in enumerate(episodes) if e["label"] == label])
        if len(indices) < 7:
            raise ValueError("Need at least 7 episodes per policy for nonempty splits")
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        result["train"].extend(indices[:n_train].tolist())
        result["val"].extend(indices[n_train:n_train + n_val].tolist())
        result["test"].extend(indices[n_train + n_val:].tolist())
    return result


class WindowDataset(Dataset):
    """All prefix-ending windows, including short early-episode windows."""
    def __init__(self, episodes, episode_indices, seq_len):
        self.episodes = episodes
        self.seq_len = seq_len
        self.windows = []
        for ep_index in episode_indices:
            n_obs = len(episodes[ep_index]["observations"])
            for end in range(1, n_obs + 1):
                self.windows.append((ep_index, end))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        ep_index, end = self.windows[index]
        e = self.episodes[ep_index]
        seq = e["observations"][max(0, end - self.seq_len):end]
        return torch.from_numpy(seq.copy()).float(), e["label"]


def collate(batch):
    lengths = torch.tensor([len(seq) for seq, _ in batch], dtype=torch.long)
    padded = torch.zeros(len(batch), max(lengths).item(), 147, dtype=torch.float32)
    labels = torch.tensor([label for _, label in batch], dtype=torch.long)
    for i, (seq, _) in enumerate(batch):
        padded[i, :len(seq)] = seq
    return padded, lengths, labels


def balanced_sampler(dataset, episodes, seed):
    # Each policy has equal total weight, and each episode within a policy
    # has equal total weight regardless of its number of windows.
    counts_per_policy = np.bincount(
        [episodes[i]["label"] for i, _ in dataset.windows],
        minlength=len(POLICIES),
    )
    episodes_per_policy = np.bincount(
        [episodes[i]["label"] for i in {ep for ep, _ in dataset.windows}],
        minlength=len(POLICIES),
    )
    windows_per_episode = {}
    for ep, _ in dataset.windows:
        windows_per_episode[ep] = windows_per_episode.get(ep, 0) + 1
    weights = [
        1.0 / (episodes_per_policy[episodes[ep]["label"]] * windows_per_episode[ep])
        for ep, _ in dataset.windows
    ]
    assert np.all(episodes_per_policy > 0)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double),
        num_samples=len(dataset), replacement=True, generator=generator,
    )


def logits_for(model, x, lengths):
    packed = pack_padded_sequence(
        x, lengths.cpu(), batch_first=True, enforce_sorted=False,
    )
    _, hidden = model.gru(packed)
    return model.head(hidden[-1])


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = np.zeros(len(POLICIES), dtype=np.int64)
    totals = np.zeros(len(POLICIES), dtype=np.int64)
    with torch.no_grad():
        for x, lengths, y in loader:
            x, y = x.to(device), y.to(device)
            logits = logits_for(model, x, lengths)
            total_loss += nn.functional.cross_entropy(logits, y, reduction="sum").item()
            predictions = logits.argmax(dim=1)
            y_np = y.cpu().numpy()
            pred_np = predictions.cpu().numpy()
            for label in range(len(POLICIES)):
                mask = y_np == label
                totals[label] += int(mask.sum())
                correct[label] += int((pred_np[mask] == label).sum())
    return {
        "loss": total_loss / max(1, int(totals.sum())),
        "accuracy": float(correct.sum() / max(1, totals.sum())),
        "macro_accuracy": float(np.mean(correct / np.maximum(totals, 1))),
        "per_policy_accuracy": (correct / np.maximum(totals, 1)).tolist(),
        "counts": totals.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/gru/ppo_pilot"))
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("models/gru_ppo_pilot"))
    args = parser.parse_args()
    if args.seq_len < 1 or args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("seq-len, epochs, batch-size, and patience must be positive")
    set_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    episodes = load_episodes(args.data_dir)
    splits = split_episodes(episodes, args.seed)
    assert len(set(splits["train"]) & set(splits["val"])) == 0
    assert len(set(splits["train"]) & set(splits["test"])) == 0
    assert len(set(splits["val"]) & set(splits["test"])) == 0

    datasets = {name: WindowDataset(episodes, indices, args.seq_len) for name, indices in splits.items()}
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size,
                              sampler=balanced_sampler(datasets["train"], episodes, args.seed), collate_fn=collate)
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = GRUTaskIdentifier(obs_dim=147, hidden_size=64, num_tasks=len(POLICIES), lr=args.lr).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_loss, best_state, best_epoch, bad_epochs = float("inf"), None, 0, 0
    print(f"Device: {device} | sequence length: {args.seq_len}")
    for name in ("train", "val", "test"):
        print(f"{name}: {len(splits[name])} episodes, {len(datasets[name])} windows")
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for x, lengths, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(logits_for(model, x, lengths), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            seen += len(y)
        val = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | train_loss={loss_sum/max(1,seen):.4f} | "
              f"val_loss={val['loss']:.4f} | val_acc={val['accuracy']:.3f} | "
              f"val_macro={val['macro_accuracy']:.3f}")
        if val["loss"] < best_loss:
            best_loss, best_state, best_epoch, bad_epochs = val["loss"], copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping")
                break
    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"gru_seq{args.seq_len}_best.pt"
    split_path = args.output_dir / f"gru_seq{args.seq_len}_splits.json"
    torch.save({"model_state_dict": best_state, "obs_dim": 147, "hidden_size": 64,
                "num_tasks": len(POLICIES), "seq_len": args.seq_len,
                "policy_names": list(POLICIES), "best_epoch": best_epoch, "seed": args.seed}, model_path)
    split_info = {
        name: [{"policy": episodes[i]["policy"], "episode_id": int(episodes[i]["episode_id"]),
                "seed": int(episodes[i]["seed"]), "success": bool(episodes[i]["success"])}
               for i in indices]
        for name, indices in splits.items()
    }
    split_path.write_text(json.dumps(split_info, indent=2), encoding="utf-8")
    print("\n=== FINAL TEST ===")
    print(f"Best epoch: {best_epoch} | accuracy: {test['accuracy']:.4f} | macro: {test['macro_accuracy']:.4f}")
    for name, accuracy, count in zip(POLICIES, test["per_policy_accuracy"], test["counts"]):
        print(f"{name:22s}: accuracy={accuracy:.3f}, windows={count}")
    print(f"Saved model: {model_path}\nSaved splits: {split_path}")


if __name__ == "__main__":
    main()
