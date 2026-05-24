"""
PN-4: Train Column {0,1} on Empty-8x8, freeze it, then train Column {0,2} on
FourRooms with a live lateral connection to {0,1}. Visualize lateral activations.

Usage:
    python scripts/train_pn4.py           # overnight full run
    python scripts/train_pn4.py --smoke   # quick pipeline validation (~40k steps total)

Verification (run with PYTHONNOUSERSITE to avoid user-site package conflicts):
    $env:PYTHONNOUSERSITE="1"
    python -m pytest tests/test_pn_trainer_iterative.py -v
    python scripts/train_pn4.py --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/train_pn4.py` from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import FlattenObservation
from minigrid.wrappers import ImgObsWrapper

from src.continual_learning.column import Column
from src.continual_learning.pn_trainer import ColumnTrainer


# ---------------------------------------------------------------------------
# Environment factory (matches the pattern used by ColumnTrainer._make_env)
# ---------------------------------------------------------------------------

def _make_env(env_id: str) -> gym.Env:
    from stable_baselines3.common.monitor import Monitor
    return Monitor(FlattenObservation(ImgObsWrapper(gym.make(env_id))))


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _pca_2d(matrix: np.ndarray) -> np.ndarray:
    """Project (N, D) to (N, 2) using SVD-based PCA. No sklearn required."""
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[:2].T


def _visualize(col1: Column, col2: Column, out_dir: Path, n_obs: int = 200) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env2 = _make_env("MiniGrid-FourRooms-v0")
    obs, _ = env2.reset()

    lat1_norms: list[float] = []
    lat2_norms: list[float] = []
    post_lat_h1_norms: list[float] = []
    post_lat_h2_norms: list[float] = []
    col1_h2_all: list[np.ndarray] = []
    col2_h2_all: list[np.ndarray] = []

    for _ in range(n_obs):
        out = col2.forward(obs)

        # col1_acts are populated as a side-effect of col2.forward() — same obs, no extra call needed.
        col1_acts = col1.get_activations()
        # col2_acts already include lateral contributions added element-wise (post-lateral).
        col2_acts = col2.get_activations()

        if col1_acts and col2_acts:
            lat1 = col2.policy.mlp_extractor.lat1
            lat2 = col2.policy.mlp_extractor.lat2

            with torch.no_grad():
                lat_h1 = lat1(col1_acts[0])
                lat_h2 = lat2(col1_acts[1])

            lat1_norms.append(lat_h1.norm().item())
            lat2_norms.append(lat_h2.norm().item())
            # col2's stored activations already have the lateral signal added in,
            # so "post-lateral" is the accurate label.
            post_lat_h1_norms.append(col2_acts[0].norm().item())
            post_lat_h2_norms.append(col2_acts[1].norm().item())

            col1_h2_all.append(col1_acts[1].squeeze().cpu().numpy())
            col2_h2_all.append(col2_acts[1].squeeze().cpu().numpy())

        obs, _, terminated, truncated, _ = env2.step(out.action)
        if terminated or truncated:
            obs, _ = env2.reset()

    env2.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: lateral contribution norm vs post-lateral activation norm ---
    x = np.arange(2)
    mean_lat = [np.mean(lat1_norms), np.mean(lat2_norms)]
    mean_post_lat = [np.mean(post_lat_h1_norms), np.mean(post_lat_h2_norms)]
    width = 0.35
    ax1.bar(x - width / 2, mean_lat, width, label="Lateral contribution")
    ax1.bar(x + width / 2, mean_post_lat, width, label="Post-lateral activation norm")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["h1", "h2"])
    ax1.set_ylabel("Mean L2 norm over 200 obs")
    ax1.set_title("Lateral Contribution vs Post-Lateral Activation Norm")
    ax1.legend()

    # --- Right: PCA of col1 h2 vs col2 h2 activations on the same obs ---
    if col1_h2_all and col2_h2_all:
        col1_h2 = np.stack(col1_h2_all)    # (N, 64)
        col2_h2 = np.stack(col2_h2_all)    # (N, 64)
        combined = np.concatenate([col1_h2, col2_h2], axis=0)  # (2N, 64)
        proj = _pca_2d(combined)            # (2N, 2)
        n = len(col1_h2_all)
        ax2.scatter(proj[:n, 0], proj[:n, 1], c="blue", alpha=0.4, s=10, label="col1 h2 (source)")
        ax2.scatter(proj[n:, 0], proj[n:, 1], c="red", alpha=0.4, s=10, label="col2 h2 (post-lateral)")
        ax2.set_title("Activation Space (PCA, h2)")
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.legend()

    fig.tight_layout()
    out_path = out_dir / "activations.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PN-4 progressive network training")
    parser.add_argument("--smoke", action="store_true", help="Quick pipeline check (20k steps per task)")
    args = parser.parse_args()

    if args.smoke:
        chunk_size = 5_000
        max_total_col1 = 20_000
        max_total_col2 = 20_000
    else:
        chunk_size = 20_000
        max_total_col1 = 500_000
        max_total_col2 = 2_000_000

    out_dir = Path("runs/pn4")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1: Column {0,1} on Empty-8x8
    # Threshold: 0.85 mean / 0.05 std (standard stabilization criterion)
    # ------------------------------------------------------------------
    print(f"\n=== Phase 1: Column {{0,1}} on MiniGrid-Empty-8x8-v0 ===")
    col1 = Column(n=0, m=1)
    trainer1 = ColumnTrainer(
        col1,
        "MiniGrid-Empty-8x8-v0",
        verbose=0,
        reward_threshold=0.85,
        std_threshold=0.05,
    )
    trainer1.train_until_stable(chunk_size=chunk_size, max_total=max_total_col1)
    col1.freeze()
    trainer1.save(out_dir / "col01")
    print(f"Col1 stable: {trainer1.is_stable()}  mean={trainer1.monitor.mean_reward:.3f}")

    # ------------------------------------------------------------------
    # Phase 2: Column {0,2} on FourRooms with lateral from {0,1}
    # Threshold: 0.75 mean / 0.05 std (relaxed for PN-4 milestone;
    # FourRooms requires ~2M steps to reach 0.85)
    # ------------------------------------------------------------------
    print(f"\n=== Phase 2: Column {{0,2}} on MiniGrid-FourRooms-v0 (lateral from {{0,1}}) ===")
    col2 = Column(n=0, m=2, lateral_source=col1)
    trainer2 = ColumnTrainer(
        col2,
        "MiniGrid-FourRooms-v0",
        verbose=0,
        reward_threshold=0.75,
        std_threshold=0.05,
    )
    trainer2.train_until_stable(chunk_size=chunk_size, max_total=max_total_col2)
    trainer2.save(out_dir / "col02")
    print(f"Col2 stable: {trainer2.is_stable()}  mean={trainer2.monitor.mean_reward:.3f}")

    # ------------------------------------------------------------------
    # Phase 3: Visualize lateral activation contributions
    # ------------------------------------------------------------------
    print(f"\n=== Phase 3: Visualization ===")
    _visualize(col1, col2, out_dir)


if __name__ == "__main__":
    main()
