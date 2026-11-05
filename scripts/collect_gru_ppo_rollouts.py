
"""Collect raw observation trajectories from 11 frozen PPO policies."""

from pathlib import Path
import argparse

import numpy as np

from src.envs.obs import OBS_DIM
from src.envs.subgoal_env import NO_HOOK
from src.skills.executor import Executor
from src.skills.registry import NodeRegistry
from src.skills.state import build_state_view
from src.training.node_trainer import (
    attach_sb3_primitive_policy,
    load_frozen_primitive,
    subgoal_env_factory,
    execution_env_factory,
)

from scripts.train_primitives import specs, build_node
from scripts.train_choice_task import all_trainable_specs
from scripts.h7_nodes import h7_primitive_specs
from scripts.train_flat_baseline import build_flat_ppo
from scripts.smoke_test_gru_hierarchical import load_hierarchical


RUNS = Path("runs/v2")
OUTPUT_DIR = Path("data/gru/ppo_pilot")

PRIMITIVES = (
    "get_key",
    "open_door",
    "go_to_goal",
    "get_key_yellow",
    "get_key_blue",
    "open_door_tk",
    "drop_held",
)

FLAT_POLICIES = {
    "flat_unlock_ppo": (
        "unlock",
        "MiniGrid-Unlock-v0",
        "0-B_flat_unlock_ppo",
    ),
    "flat_doorkey_ppo": (
        "doorkey",
        "MiniGrid-DoorKey-6x6-v0",
        "0-B_flat_doorkey_ppo",
    ),
}

HIERARCHICAL = (
    "unlock_door",
    "solve_doorkey",
)

ALL_POLICIES = PRIMITIVES + tuple(FLAT_POLICIES) + HIERARCHICAL


def load_policy(name, registry):
    """Load a frozen policy and its matching environment."""

    if name in PRIMITIVES:
        all_specs = {
            **specs(),
            **all_trainable_specs(),
            **h7_primitive_specs(),
        }
        spec = all_specs[name]
        node = build_node(spec)
        env_id = spec["env_ids"][0]
        hook = spec["hook"]

        load_frozen_primitive(
            node, registry, env_id, hook,
            normalize=True, device="cpu",
        )

        env = subgoal_env_factory(
            node, env_id, hook,
            normalize=True, monitor=False,
        )()

        return node, env, "primitive", env_id

    if name in FLAT_POLICIES:
        task, env_id, checkpoint_dir = FLAT_POLICIES[name]

        node = build_flat_ppo(task)

        attach_sb3_primitive_policy(
            node, env_id, NO_HOOK,
            normalize=True, device="cpu",
        )

        registry.load_into(
            node,
            run_dir=RUNS / "baselines" / checkpoint_dir,
            require_frozen=True,
        )

        env = subgoal_env_factory(
            node, env_id, NO_HOOK,
            normalize=True, monitor=False,
        )()

        return node, env, "flat", env_id

    if name in HIERARCHICAL:
        node, env_id = load_hierarchical(name, registry)

        env = execution_env_factory(
            env_id, NO_HOOK, normalize=True,
        )()

        return node, env, "hierarchical", env_id

    raise ValueError(f"Unknown policy: {name}")


def collect_episode(node, env, kind, seed):
    """Run one episode and preserve its complete trajectory."""

    obs, _ = env.reset(seed=seed)

    observations = [np.asarray(obs, dtype=np.float32).copy()]
    actions = []
    rewards = []

    executor = Executor(node) if kind == "hierarchical" else None

    version = 0
    state = (
        build_state_view(env, version)
        if executor is not None
        else None
    )

    terminated = False
    truncated = False

    for step in range(node.timeout):
        assert obs.shape == (OBS_DIM,)
        assert np.isfinite(obs).all()

        if executor is not None:
            action = executor.act(obs, state)
            if action is None:
                break
        else:
            action = node.policy.act(obs, deterministic=True)

        action = int(action)

        next_obs, reward, terminated, truncated, _ = env.step(action)

        actions.append(action)
        rewards.append(float(reward))
        observations.append(
            np.asarray(next_obs, dtype=np.float32).copy()
        )

        obs = next_obs

        if executor is not None:
            version += 1
            state = build_state_view(env, version)
            executor.observe_reward(reward)

        if terminated or truncated:
            break

    if executor is not None:
        success = bool(node.goal(state))
    else:
        # SubgoalEnv sets terminated=True exactly when
        # the node's goal has been reached.
        success = bool(terminated)

    return {
        "observations": np.stack(observations),
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "success": success,
        "steps": len(actions),
    }


def collect_policy(name, registry, n_episodes, seed_base, output_dir):
    print(f"\n=== Collecting {name} ===")

    node, env, kind, env_id = load_policy(name, registry)

    all_obs = []
    all_actions = []
    all_rewards = []
    episode_ids = []
    episode_seeds = []
    episode_successes = []
    episode_lengths = []

    try:
        for episode in range(n_episodes):
            seed = seed_base + episode

            result = collect_episode(node, env, kind, seed)

            all_obs.extend(result["observations"])
            all_actions.extend(result["actions"])
            all_rewards.extend(result["rewards"])

            episode_ids.extend(
                [episode] * len(result["observations"])
            )
            episode_seeds.append(seed)
            episode_successes.append(result["success"])
            episode_lengths.append(result["steps"])

            print(
                f"  {episode + 1:03d}/{n_episodes}: "
                f"steps={result['steps']}, "
                f"success={result['success']}"
            )

    finally:
        env.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.npz"

    np.savez_compressed(
        output_path,
        observations=np.asarray(all_obs, dtype=np.float32),
        actions=np.asarray(all_actions, dtype=np.int64),
        rewards=np.asarray(all_rewards, dtype=np.float32),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        episode_seeds=np.asarray(episode_seeds, dtype=np.int64),
        episode_successes=np.asarray(episode_successes, dtype=bool),
        episode_lengths=np.asarray(episode_lengths, dtype=np.int64),
        policy_name=np.asarray(name),
        policy_kind=np.asarray(kind),
        env_id=np.asarray(env_id),
    )

    print(
        f"Saved: {output_path} | "
        f"success={np.mean(episode_successes):.1%} | "
        f"mean_steps={np.mean(episode_lengths):.2f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=20000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    parser.add_argument(
        "--policy",
        choices=ALL_POLICIES,
        default=None,
        help="Collect one policy; default collects all 11.",
    )
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be positive")

    registry = NodeRegistry(RUNS)

    selected = (
        (args.policy,) if args.policy else ALL_POLICIES
    )

    for name in selected:
        collect_policy(
            name,
            registry,
            args.episodes,
            args.seed_base,
            args.output_dir,
        )

    print("\nCollection completed.")


if __name__ == "__main__":
    main()
