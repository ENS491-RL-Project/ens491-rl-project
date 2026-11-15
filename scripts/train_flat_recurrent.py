"""
W4(a): a flat PPO with memory (RecurrentPPO), on the same tasks as the memoryless flat PPO.

    {0,B} flat_<task>_rppo     same goal predicate, timeout, env, observation pipeline and PPO
                               hyperparameters as {0,B} flat_<task>_ppo; an LSTM in front of the heads.

Plan and pre-set readings: docs/plans/2026-11-15_W4a_recurrent_flat_plan.md.

Evaluation does not go through the `Executor`, which carries no hidden state: `evaluate` keeps the
LSTM state for one episode on the same `SubgoalEnv`. tests/test_flat_recurrent.py shows the two
paths agree episode by episode for a memoryless policy. A baseline is never a child.

    python scripts/train_flat_recurrent.py --task doorkey --smoke
    python scripts/train_flat_recurrent.py --task unlockpickup --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.envs.subgoal_env import NO_HOOK  # noqa: E402
from src.skills.registry import NodeRegistry  # noqa: E402
from src.training.device import describe, resolve_device  # noqa: E402
from src.training.evaluation import (  # noqa: E402
    EvalResult,
    EvaluationCallback,
    StabilityCriterion,
    heldout_seeds,
    stopping_seeds,
)
from src.training.node_trainer import (  # noqa: E402
    PPO_DEFAULTS,
    make_training_vec_env,
    set_torch_threads,
    subgoal_env_factory,
)
from src.training.run_store import record_training, track_model  # noqa: E402
from src.types import NodeStatus  # noqa: E402
from train_flat_baseline import BASELINE_RUNS, RUNS, TASKS, build_flat_ppo, curve, seed_dir  # noqa: E402

LSTM = dict(net_arch=[64, 64], lstm_hidden_size=64)     # fixed in the plan


def evaluate(predict, node, env_id, seeds) -> EvalResult:
    """Deterministic episodes on `SubgoalEnv`; `predict` is an SB3 `predict`, its state reset per episode."""
    env = subgoal_env_factory(node, env_id, NO_HOOK, monitor=False)()
    outcomes, lengths = [], []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            state, start, steps, reached, truncated = None, np.ones(1, dtype=bool), 0, False, False
            while not (reached or truncated):
                action, state = predict(obs, state=state, episode_start=start, deterministic=True)
                obs, _, reached, truncated, _ = env.step(int(action))
                start = np.zeros(1, dtype=bool)
                steps += 1
            outcomes.append(bool(reached))
            lengths.append(steps)
    finally:
        env.close()
    return EvalResult(n_episodes=len(seeds), successes=sum(outcomes),
                      mean_length=sum(lengths) / len(lengths), seed_base=seeds[0],
                      per_episode=outcomes, lengths=lengths, timeouts=len(seeds) - sum(outcomes))


@record_training
def train_recurrent(node, env_id, *, total_timesteps, eval_every, n_eval=50, n_envs=8,
                    seed=42, device="cpu", run_root=None, verbose=0):
    from sb3_contrib import RecurrentPPO

    dev = resolve_device(device)
    node.status = NodeStatus.TRAINING
    vec = make_training_vec_env(node, [env_id], NO_HOOK, n_envs=n_envs, seed=seed)
    model = RecurrentPPO("MlpLstmPolicy", vec, policy_kwargs=dict(LSTM), seed=seed, device=dev,
                         verbose=0, **PPO_DEFAULTS)
    node.policy = model.policy      # state_dict / parameters: all the archive and registry need
    track_model(model)

    criterion = StabilityCriterion()
    callback = EvaluationCallback(
        node,
        eval_fn=lambda: evaluate(model.predict, node, env_id, stopping_seeds(n_eval)),
        eval_every=eval_every,
        criterion=criterion,
        update_every=PPO_DEFAULTS["n_steps"] * n_envs,
        verbose=verbose,
    )
    t0 = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    wall = time.perf_counter() - t0

    report = {
        "node_id": node.id, "env_ids": [env_id], "timeout": node.timeout, "seed": seed,
        "device": dev, "n_envs": n_envs, "eval_every": eval_every,
        "algorithm": "RecurrentPPO", "policy": "MlpLstmPolicy", "policy_kwargs": LSTM,
        "ppo": dict(PPO_DEFAULTS), "budget": total_timesteps,
        "steps_trained": model.num_timesteps, "policy_updates": callback.train_calls,
        "wall_seconds": round(wall, 1),
        "steps_per_second": round(model.num_timesteps / wall, 1) if wall > 0 else None,
        "stable": criterion.is_stable, "criterion": criterion.as_dict(),
    }
    if criterion.is_stable:
        node.status = NodeStatus.STABLE
    return report, model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["doorkey", "unlockpickup"], required=True)   # the plan's two tasks
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--torch-threads", type=int, default=2)
    args = ap.parse_args()

    task = TASKS[args.task]
    env_id = task["env"]
    budget, eval_every = task["ppo"]
    n_eval, n_envs = 50, 8
    if args.smoke:
        budget, eval_every, n_eval, n_envs = 8_000, 2_000, 10, 2

    print(describe())
    print(f"torch threads: {set_torch_threads(args.torch_threads)}")
    node = build_flat_ppo(args.task, kind="rppo")
    registry = NodeRegistry(BASELINE_RUNS)
    run_dir = seed_dir(registry, node.id, args.seed)
    if not args.smoke:
        registry.ensure_can_save(node, run_dir=run_dir)     # fail now, not after training
    print(f"\n{node.id}  env={env_id} timeout={node.timeout} goal={node.goal.spec}"
          f"\n  seed={args.seed} budget={budget:,} env steps eval_every={eval_every:,}")

    report, model = train_recurrent(node, env_id, total_timesteps=budget, eval_every=eval_every,
                                    n_eval=n_eval, n_envs=n_envs, seed=args.seed,
                                    device=args.device, verbose=1)
    print(f"  steps={report['steps_trained']:,} wall={report['wall_seconds']}s "
          f"({report['steps_per_second']} steps/s) stable={report['stable']}")
    report["curve"] = curve(report, steps_per_x=1.0, eval_every=eval_every)
    heldout = evaluate(model.predict, node, env_id, heldout_seeds(n_eval))
    report["heldout"] = heldout.as_dict()
    print(f"  held-out: {heldout}")

    if not args.smoke:
        if report["stable"]:
            node.freeze()
        # Saved whether or not it is STABLE: a censored baseline must stay re-scorable.
        registry.save(node, extra_meta=report, run_dir=run_dir)
        print(f"  saved to runs/v2/baselines/{run_dir.name}/ ({node.status.value}; baseline, never a child)")

    name = f"flat_{args.task}_rppo{'_smoke' if args.smoke else ''}_s{args.seed}.json"
    (RUNS / name).write_text(json.dumps({"task": args.task, "seed": args.seed, "env_id": env_id,
                                         "report": report}, indent=2), encoding="utf-8")
    print(f"\nwrote runs/v2/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
