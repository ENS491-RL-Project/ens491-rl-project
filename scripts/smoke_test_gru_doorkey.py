
from pathlib import Path

import numpy as np

from src.envs.obs import OBS_DIM
from src.envs.subgoal_env import NO_HOOK
from src.skills.registry import NodeRegistry
from src.training.node_trainer import (
    attach_sb3_primitive_policy,
    subgoal_env_factory,
)

from scripts.train_flat_baseline import build_flat_ppo


RUNS = Path("runs/v2")
OUTPUT = Path("data/gru/doorkey_policy_smoke.npz")
N_EPISODES = 3


def main():
    # 1. Build the exact node specification used in training.
    node = build_flat_ppo("doorkey")
    registry = NodeRegistry(RUNS)

    # 2. Load the frozen PPO checkpoint.
    # Build the correct PPO policy architecture.
    attach_sb3_primitive_policy(
        node,
        "MiniGrid-DoorKey-6x6-v0",
        NO_HOOK,
        normalize=True,
        device="cpu",
    )

    # Load the checkpoint from the baseline directory.
    registry.load_into(
        node,
        run_dir=RUNS / "baselines" / "0-B_flat_doorkey_ppo",
        require_frozen=True,
    )
    print("Loaded policy:", node.id)

    # 3. Create the same wrapped environment.
    env = subgoal_env_factory(
        node,
        "MiniGrid-DoorKey-6x6-v0",
        NO_HOOK,
        normalize=True,
        monitor=False,
    )()

    observations = []
    actions = []
    episode_ids = []
    episode_successes = []

    try:
        for episode in range(N_EPISODES):
            obs, info = env.reset(seed=10000 + episode)
            done = False
            steps = 0
            episode_observations = [obs.copy()]
            episode_actions = []
            success = False

            while not done:
                assert obs.shape == (OBS_DIM,)
                assert np.isfinite(obs).all()
                assert obs.min() >= 0.0
                assert obs.max() <= 1.0

                action = node.policy.act(obs, deterministic=True)
                next_obs, reward, terminated, truncated, info = env.step(action)

                episode_actions.append(action)
                episode_observations.append(next_obs.copy())

                obs = next_obs
                steps += 1
                done = terminated or truncated

                # SubgoalEnv uses the node's goal to determine success.
                if terminated:
                    success = True

                if steps >= node.timeout:
                    break

            observations.extend(episode_observations)
            actions.extend(episode_actions)
            episode_ids.extend([episode] * len(episode_observations))
            episode_successes.append(success)

            print(
                f"Episode {episode + 1}: "
                f"steps={steps}, success={success}, "
                f"observations={len(episode_observations)}"
            )

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            OUTPUT,
            observations=np.asarray(observations, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.int64),
            episode_ids=np.asarray(episode_ids, dtype=np.int64),
            episode_successes=np.asarray(episode_successes, dtype=bool),
        )

        print("\n=== SMOKE TEST SUMMARY ===")
        print("Observation shape:", np.asarray(observations).shape)
        print("Total actions:", len(actions))
        print("Success rate:", np.mean(episode_successes))
        print("Saved:", OUTPUT)

    finally:
        env.close()


if __name__ == "__main__":
    main()
