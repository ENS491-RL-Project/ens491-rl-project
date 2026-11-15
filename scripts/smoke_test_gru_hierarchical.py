
from pathlib import Path

import numpy as np

from src.envs.obs import OBS_DIM
from src.envs.subgoal_env import NO_HOOK
from src.skills.executor import Executor
from src.skills.registry import NodeRegistry
from src.skills.state import build_state_view
from src.training.node_trainer import (
    execution_env_factory,
    load_frozen_composite,
)

from scripts.train_unlock_door import (
    build_t3,
    load_children as load_unlock_children,
)
from scripts.train_solve_doorkey import (
    build_t5,
    load_children as load_doorkey_children,
)
from scripts.train_flat_baseline import (
    build_flat_options,
    load_option_children,
)

RUNS = Path("runs/v2")
OUTPUT_DIR = Path("data/gru/hierarchical_smoke")
N_EPISODES = 3

UNLOCK = "MiniGrid-Unlock-v0"
DOORKEY = "MiniGrid-DoorKey-6x6-v0"


def load_hierarchical(name, registry):
    if name == "unlock_door":
        children = load_unlock_children(registry)
        node = build_t3(children)
        env_id = UNLOCK

    elif name == "solve_doorkey":
        unlock_door, go_to_goal = (
            load_doorkey_children(registry)
        )
        node = build_t5(unlock_door, go_to_goal)
        env_id = DOORKEY

    elif name == "flat_doorkey_options":
        children = load_option_children(
            "doorkey", registry
        )
        node = build_flat_options(
            children, task="doorkey"
        )
        env_id = DOORKEY

    else:
        raise ValueError(f"Unknown model: {name}")

    parent_registry = (
    NodeRegistry(RUNS / "baselines")
    if name == "flat_doorkey_options"
    else registry
    )

    load_frozen_composite(
        node,
        parent_registry,
        env_id,
        NO_HOOK,
        normalize=True,
        device="cpu",
    )

    return node, env_id


def smoke_test(name, registry):
    node, env_id = load_hierarchical(name, registry)

    env = execution_env_factory(
        env_id,
        NO_HOOK,
        normalize=True,
    )()

    observations = []
    actions = []
    episode_ids = []
    successes = []
    lengths = []

    try:
        for episode in range(N_EPISODES):
            obs, info = env.reset(seed=10000 + episode)
            executor = Executor(node)

            version = 0
            state = build_state_view(env, version)

            episode_obs = [obs.copy()]
            episode_actions = []

            success = False
            steps = 0

            while steps < node.timeout:
                assert obs.shape == (OBS_DIM,)
                assert np.isfinite(obs).all()

                action = executor.act(obs, state)

                if action is None:
                    break

                next_obs, reward, terminated, truncated, info = (
                    env.step(action)
                )

                episode_actions.append(action)
                episode_obs.append(next_obs.copy())

                obs = next_obs
                steps += 1
                version += 1

                state = build_state_view(env, version)
                executor.observe_reward(reward)

                if terminated or truncated:
                    break

            success = bool(node.goal(state))
            observations.extend(episode_obs)
            actions.extend(episode_actions)
            episode_ids.extend(
                [episode] * len(episode_obs)
            )
            successes.append(success)
            lengths.append(steps)

            print(
                f"  Episode {episode + 1}: "
                f"steps={steps}, success={success}"
            )

    finally:
        env.close()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"{name}.npz"

    np.savez_compressed(
        output,
        observations=np.asarray(
            observations, dtype=np.float32
        ),
        actions=np.asarray(
            actions, dtype=np.int64
        ),
        episode_ids=np.asarray(
            episode_ids, dtype=np.int64
        ),
        episode_successes=np.asarray(
            successes, dtype=bool
        ),
    )

    result = {
        "task": name,
        "success_rate": float(np.mean(successes)),
        "mean_length": float(np.mean(lengths)),
        "observations": len(observations),
    }

    print(f"  Success rate: {result['success_rate']:.1%}")
    print(f"  Mean length: {result['mean_length']:.2f}")
    print(f"  Saved: {output}")

    return result


def main():
    registry = NodeRegistry(RUNS)
    results = []

    for name in (
    "solve_doorkey",
    "flat_doorkey_options",
    ):
        print(f"\n=== Testing {name} ===")

        try:
            result = smoke_test(name, registry)
            results.append(result)
        except Exception as exc:
            print(
                f"FAILED {name}: "
                f"{type(exc).__name__}: {exc}"
            )

    print("\n=== SUMMARY ===")
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
