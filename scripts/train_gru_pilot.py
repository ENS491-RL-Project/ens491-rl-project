
from pathlib import Path
import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.task_detection.gru_identifier import GRUTaskIdentifier


DATA_PATH = Path("data/gru/pilot_dataset.npz")
MODEL_PATH = Path("models/gru_pilot_best.pt")

SEED = 42
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
PATIENCE = 6


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_splits(y, episode_ids, num_tasks):
    """Split by episode, not by overlapping sequence."""
    rng = np.random.default_rng(SEED)
    splits = {"train": [], "val": [], "test": []}

    for task_id in range(num_tasks):
        task_indices = np.flatnonzero(y == task_id)
        task_episode_ids = np.unique(episode_ids[task_indices])
        rng.shuffle(task_episode_ids)

        n = len(task_episode_ids)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)

        groups = {
            "train": task_episode_ids[:n_train],
            "val": task_episode_ids[n_train:n_train + n_val],
            "test": task_episode_ids[n_train + n_val:],
        }

        for split_name, ids in groups.items():
            selected = task_indices[
                np.isin(episode_ids[task_indices], ids)
            ]
            splits[split_name].extend(selected.tolist())

    return {
        name: np.asarray(indices, dtype=np.int64)
        for name, indices in splits.items()
    }


def predict_logits(model, sequences):
    """
    Batched version of the existing GRU architecture.
    sequences: (batch, N, 147)
    """
    _, hidden = model.gru(sequences)
    features = hidden[-1]
    return model.head(features)


def evaluate(model, loader, device, num_tasks):
    model.eval()
    correct = np.zeros(num_tasks, dtype=np.int64)
    total = np.zeros(num_tasks, dtype=np.int64)
    loss_sum = 0.0
    sample_count = 0

    criterion = nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = predict_logits(model, X_batch)
            loss_sum += criterion(logits, y_batch).item()

            predictions = logits.argmax(dim=1)

            for task_id in range(num_tasks):
                mask = y_batch == task_id
                total[task_id] += mask.sum().item()
                correct[task_id] += (
                    (predictions == y_batch) & mask
                ).sum().item()

            sample_count += len(y_batch)

    per_task_accuracy = correct / np.maximum(total, 1)
    overall_accuracy = correct.sum() / max(total.sum(), 1)
    macro_accuracy = per_task_accuracy.mean()

    return {
        "loss": loss_sum / sample_count,
        "accuracy": overall_accuracy,
        "macro_accuracy": macro_accuracy,
        "per_task_accuracy": per_task_accuracy,
    }


def main():
    set_seed(SEED)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("Device:", device)

    with np.load(DATA_PATH, allow_pickle=False) as data:
        X = data["X"]
        y = data["y"]
        episode_ids = data["episode_ids"]
        env_names = data["env_names"]

    num_tasks = len(env_names)

    assert X.ndim == 3
    assert X.shape[2] == 147
    assert set(np.unique(y)) == set(range(num_tasks))

    splits = make_splits(y, episode_ids, num_tasks)

    # Verify no episode appears in multiple splits.
    episode_sets = {
        name: set(episode_ids[idx].tolist())
        for name, idx in splits.items()
    }

    assert episode_sets["train"].isdisjoint(episode_sets["val"])
    assert episode_sets["train"].isdisjoint(episode_sets["test"])
    assert episode_sets["val"].isdisjoint(episode_sets["test"])

    for name, indices in splits.items():
        print(
            f"{name}: {len(indices)} sequences, "
            f"{len(episode_sets[name])} episodes"
        )

    # Persist the exact split for reproducibility.
    split_path = Path("data/gru/pilot_splits.npz")
    split_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(split_path, **{
        f"{name}_idx": indices
        for name, indices in splits.items()
    })

    # Build tensors and dataloaders.
    datasets = {}

    for name, indices in splits.items():
        datasets[name] = TensorDataset(
            torch.from_numpy(X[indices].copy()).float(),
            torch.from_numpy(y[indices].copy()).long(),
        )

    train_labels = y[splits["train"]]
    class_counts = np.bincount(
        train_labels, minlength=num_tasks
    )

    sample_weights = 1.0 / class_counts[train_labels]

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(train_labels),
        replacement=True,
    )

    train_loader = DataLoader(
        datasets["train"],
        batch_size=BATCH_SIZE,
        sampler=sampler,
    )

    val_loader = DataLoader(
        datasets["val"],
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        datasets["test"],
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = GRUTaskIdentifier(
        obs_dim=147,
        hidden_size=64,
        num_tasks=num_tasks,
        lr=LEARNING_RATE,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    print("\n=== TRAINING ===")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running_loss = 0.0
        seen = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = predict_logits(model, X_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(y_batch)
            seen += len(y_batch)

        train_loss = running_loss / seen
        val_metrics = evaluate(
            model, val_loader, device, num_tasks
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_accuracy={val_metrics['accuracy']:.3f} | "
            f"val_macro={val_metrics['macro_accuracy']:.3f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping.")
            break

    # Evaluate only the best validation checkpoint.
    model.load_state_dict(best_state)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": best_state,
            "obs_dim": 147,
            "hidden_size": 64,
            "num_tasks": num_tasks,
            "seq_len": X.shape[1],
            "env_names": env_names.tolist(),
            "best_epoch": best_epoch,
            "seed": SEED,
        },
        MODEL_PATH,
    )

    print("\n=== FINAL TEST ===")

    test_metrics = evaluate(
        model, test_loader, device, num_tasks
    )

    print("Best epoch:", best_epoch)
    print("Test accuracy:", round(test_metrics["accuracy"], 4))
    print(
        "Test macro accuracy:",
        round(test_metrics["macro_accuracy"], 4)
    )

    for task_id, accuracy in enumerate(
        test_metrics["per_task_accuracy"]
    ):
        print(
            f"Task {task_id} ({env_names[task_id]}): "
            f"{accuracy:.4f}"
        )

    print("\nSaved model:", MODEL_PATH)
    print("Saved splits:", split_path)


if __name__ == "__main__":
    main()
