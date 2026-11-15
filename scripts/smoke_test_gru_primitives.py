
from pathlib import Path

import numpy as np

from src.envs.obs import OBS_DIM
from src.skills.registry import NodeRegistry
from src.training.node_trainer import (
    load_frozen_primitive,
    subgoal_env_factory,
)

from scripts.train_primitives import specs, build_node
from scripts.train_choice_task import all_trainable_specs
from scripts.h7_nodes import h7_primitive_specs

RUNS = Path("runs/v2")
OUTPUT_DIR = Path("data/gru/primitive_smoke")
N_EPISODES = 3

TASKS = (
    "get_key_yellow",
    "get_key_blue",
    "open_door_tk",
    "drop_held",
)


def smoke_test(name, registry):
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
        node,
        registry,
        env_id,
        hook,
        normalize=True,
        device="cpu",
    )

    env = subgoal_env_factory(
        node,
        env_id,
        hook,
        normalize=True,
        monitor=False,
    )()

    observations = []
    actions = []
    episode_ids = []
    successes = []
    lengths = []

    try:
        for episode in range(N_EPISODES):
            obs, info = env.reset(seed=10000 + episode)

            episode_obs = [obs.copy()]
            episode_actions = []

            success = False
            steps = 0

            while steps < node.timeout:
                assert obs.shape == (OBS_DIM,)
                assert np.isfinite(obs).all()
                assert 0.0 <= obs.min() <= obs.max() <= 1.0

                action = node.policy.act(
                    obs, deterministic=True
                )

                next_obs, reward, terminated, truncated, info = (
                    env.step(action)
                )

                episode_actions.append(action)
                episode_obs.append(next_obs.copy())

                obs = next_obs
                steps += 1

                if terminated:
                    success = True

                if terminated or truncated:
                    break

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

    print(
        f"  Success rate: {np.mean(successes):.1%}"
    )
    print(
        f"  Mean episode length: {np.mean(lengths):.2f}"
    )
    print(f"  Saved: {output}")

    return {
        "task": name,
        "success_rate": float(np.mean(successes)),
        "mean_length": float(np.mean(lengths)),
        "observations": len(observations),
    }


def main():
    registry = NodeRegistry(RUNS)
    results = []

    for name in TASKS:
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
