"""
H-6.4 / H-6.5: the C1 baselines for DoorKey-6x6.

Two of them, answering two different questions that the single phrase "flat
baseline" conflates:

    {0,B} flat_doorkey_ppo       plain PPO, no skills at all.  Do the frozen
                                 skills help at all?
    {1,B} flat_doorkey_options   one composite over the three primitives, no
                                 intermediate unlock_door.  Do the skills help,
                                 or does the *depth* help?

Everything that could flatter either side is held identical to
`{2,1} solve_doorkey`: the same goal predicate (`on_goal_tile`, read off the
StateView, never the env's reward), the same 360-step timeout, the same
observation pipeline, the same PPO hyperparameters, the same 8 envs, the same
stopping criterion on the same stopping seeds, and the same held-out seeds scored
through the same `Executor` -- `evaluate_node` scores a primitive and a composite
with one code path (v2 §5.3), so the comparison is not mediated by two different
evaluation harnesses.

Both nodes are **baselines, not curriculum nodes** (v2 §11, H-6 A.1). They are
written to `runs/v2/baselines/` so they can never be picked up as a child of a
real node.

`eval_every` is chosen from a measurement, not guessed: one evaluation is 50
single-env episodes capped at 360 steps, so evaluating a policy that fails
everything costs up to 18,000 env steps of Python inference, and at a fine
`eval_every` that would dominate the run. `--measure-eval` times it.

[H-7.2] `--task` extends the same harness to the rest of the curriculum, each
baseline matched to the node it is compared with (v2 §11, H-7 A.4):

    --task unlock        {0,B} flat_unlock_ppo            door open,        240,  1.5M / 50k
    --task unlockpickup  {0,B} flat_unlockpickup_ppo      carrying the box, 288,  3M   / 100k
                         {1,B} flat_unlockpickup_options  (get_key, open_door, drop_held, pickup_box),
                                                          288, max_decisions 12, 40,960 decisions / 512
                                                          (children per H-7 amendment 1)
    --task blocked       {0,B} flat_blocked_ppo           carrying the box, 576,  3M   / 100k

`doorkey` stays the default so the H-6 commands still reproduce. Budgets are
fixed in advance and never extended (H-7 plan, Part B).

Usage:
    python scripts/train_flat_baseline.py --which ppo --measure-eval
    python scripts/train_flat_baseline.py --which ppo --smoke
    python scripts/train_flat_baseline.py --which ppo     --seed 42
    python scripts/train_flat_baseline.py --which options --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.envs.subgoal_env import NO_HOOK  # noqa: E402
from src.skills.node import ALWAYS, SkillNode  # noqa: E402
from src.skills.predicates import carrying, door_open, on_goal_tile  # noqa: E402
from src.skills.registry import NodeRegistry  # noqa: E402
from src.training.c1_controls import ARM_LABELS, capture_trace, mask_divergence, run_arms  # noqa: E402
from src.training.device import describe  # noqa: E402
from src.training.evaluation import heldout_seeds  # noqa: E402
from src.training.node_trainer import (  # noqa: E402
    execution_env_factory,
    pooled_eval_fn,
    set_torch_threads,
    train_composite,
    train_primitive,
)
from train_primitives import DOORKEY, UNLOCK, build_node, specs  # noqa: E402
from train_solve_doorkey import load_children  # noqa: E402

RUNS = REPO / "runs" / "v2"
BASELINE_RUNS = RUNS / "baselines"

# Identical to {2,1} solve_doorkey: same goal, same ceiling (DoorKey-6x6's own
# max_steps is 360), so neither side gets more of the episode than the other.
TIMEOUT = 360
OPTIONS_MAX_DECISIONS = 12      # [A: F2] 3-4 children, room for a few wasted picks

UNLOCKPICKUP = "MiniGrid-UnlockPickup-v0"
BLOCKED = "MiniGrid-BlockedUnlockPickup-v0"

# [H-7.2] One row per task. Goal and timeout match the curriculum node the
# baseline is compared with: {1,1} unlock_door (240), {2,2} unlock_pickup (288),
# {3,1} blocked_pickup_b (576). `ppo` budgets are env steps; `options` budgets
# are SMDP decisions.
TASKS = {
    "doorkey": dict(env=DOORKEY, name="doorkey", goal=on_goal_tile, timeout=TIMEOUT,
                    ppo=(1_500_000, 50_000), options=(20_000, 1_000)),
    "unlock": dict(env=UNLOCK, name="unlock", goal=door_open, timeout=240,
                   ppo=(1_500_000, 50_000), options=None),
    "unlockpickup": dict(env=UNLOCKPICKUP, name="unlockpickup",
                         goal=lambda: carrying("box"), timeout=288,
                         ppo=(3_000_000, 100_000), options=(40_960, 512)),
    "blocked": dict(env=BLOCKED, name="blocked", goal=lambda: carrying("box"), timeout=576,
                    ppo=(3_000_000, 100_000), options=None),
}


def build_flat_ppo(task="doorkey"):
    t = TASKS[task]
    return SkillNode(
        "{0,B} flat_" + t["name"] + "_ppo",
        goal=t["goal"](),
        initiation=ALWAYS,          # the real task starts wherever the env starts
        timeout=t["timeout"],
        env_ids=[t["env"]],
        train_hook="none",
        m=1,
    )


def build_flat_options(children, task="doorkey"):
    t = TASKS[task]
    return SkillNode(
        "{1,B} flat_" + t["name"] + "_options",
        children=children,
        goal=t["goal"](),
        initiation=ALWAYS,
        timeout=t["timeout"],
        max_decisions=OPTIONS_MAX_DECISIONS,
        env_ids=[t["env"]],
        m=1,
    )


def load_option_children(task, registry, device="cpu"):
    if task == "doorkey":
        return load_three_primitives(registry, device=device)
    # unlockpickup (H-7 amendment 1): get_key, open_door, drop_held and
    # pickup_box -- the skills {2,2} unlock_pickup has, without the
    # intermediate unlock_door
    from h7_nodes import load_h7_primitive
    kids = load_three_primitives(registry, device=device, names=("get_key", "open_door"))
    kids.append(load_h7_primitive("drop_held", registry, device=device))
    kids.append(load_h7_primitive("pickup_box", registry, device=device))
    return kids


def load_three_primitives(registry, device="cpu", names=("get_key", "open_door", "go_to_goal")):
    from src.training.node_trainer import load_frozen_primitive
    from src.types import NodeStatus

    all_specs = specs()
    out = []
    for name in names:
        spec = all_specs[name]
        node = build_node(spec)
        load_frozen_primitive(node, registry, spec["env_ids"][0], spec["hook"],
                              device=device)
        if node.status is not NodeStatus.FROZEN:
            raise RuntimeError(f"{node.id} did not load as FROZEN")
        print(f"  loaded {node}")
        out.append(node)
    return out


def measure_eval_cost(n_eval: int, task: str = "doorkey") -> dict:
    """
    Time one evaluation of an untrained policy -- the worst case, because every
    episode runs to the full timeout.

    Printed as a suggested `eval_every`, not applied silently: `eval_every` sets
    the resolution of the success-rate-vs-env-steps curve, so the value used has
    to be a recorded decision.
    """
    from src.training.node_trainer import attach_sb3_primitive_policy

    env_id = TASKS[task]["env"]
    node = build_flat_ppo(task)
    attach_sb3_primitive_policy(node, env_id, NO_HOOK)
    t0 = time.perf_counter()
    res = pooled_eval_fn(node, [env_id], NO_HOOK, n_episodes=n_eval,
                         seed_base=heldout_seeds(1)[0])()
    dt = time.perf_counter() - t0
    env_steps = sum(res.lengths)
    print(f"  one evaluation ({n_eval} episodes, untrained): {dt:.1f}s, "
          f"{env_steps:,} env steps, SR {res.success_rate:.2f}")
    # Training throughput measured in H-4 was ~1,050 env steps/s. Keep total
    # evaluation wall time under ~40% of training wall time.
    return {"task": task, "env_id": env_id, "seconds": round(dt, 1), "env_steps": env_steps,
            "success_rate": res.success_rate}


def curve(report: dict, *, steps_per_x: float, eval_every: int) -> list[dict]:
    """
    The success-rate-vs-env-steps curve, with the x axis in env steps.

    A composite's `eval_every` counts SMDP decisions, so its x axis is scaled by
    the measured env steps per decision. Without that conversion the two
    baselines and the composite would be plotted on three incomparable axes --
    which is the whole difficulty of comparing an SMDP against a flat MDP.
    """
    out = []
    for i, h in enumerate(report["criterion"]["history"], start=1):
        out.append({
            "x_env_steps": int(round(i * eval_every * steps_per_x)),
            "x_native": i * eval_every,
            "success_rate": h["success_rate"],
            "wilson_lower": h["wilson_lower"],
            "mean_length": h["mean_length"],
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["ppo", "options"], required=True)
    ap.add_argument("--task", choices=list(TASKS), default="doorkey")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--measure-eval", action="store_true",
                    help="time one evaluation and exit")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--budget", type=int, default=None,
                    help="env steps (ppo) or SMDP decisions (options)")
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()

    task = TASKS[args.task]
    env_id = task["env"]
    if task[args.which] is None:
        ap.error(f"--which {args.which} is not defined for --task {args.task}")
    if args.budget is None:
        args.budget = task[args.which][0]
    if args.eval_every is None:
        args.eval_every = task[args.which][1]
    if args.smoke:
        args.n_eval = 10
        args.n_envs = 2
        args.budget = 8_000 if args.which == "ppo" else 600
        args.eval_every = 2_000 if args.which == "ppo" else 200

    print(describe())
    print(f"torch threads: {set_torch_threads(args.torch_threads)}")

    if args.measure_eval:
        cost = measure_eval_cost(args.n_eval, args.task)
        out = RUNS / ("flat_baseline_eval_cost.json" if args.task == "doorkey"
                      else f"flat_baseline_eval_cost_{args.task}.json")
        out.write_text(json.dumps(cost, indent=2), encoding="utf-8")
        print(f"wrote {out.relative_to(REPO)}")
        return 0

    registry = NodeRegistry(RUNS)
    baseline_registry = NodeRegistry(BASELINE_RUNS)
    result: dict = {"which": args.which, "task": args.task, "seed": args.seed,
                    "env_id": env_id, "budget": args.budget, "eval_every": args.eval_every}

    if args.which == "ppo":
        node = build_flat_ppo(args.task)
        print(f"\n{'=' * 72}\n{node.id}  n={node.n}  (no children: plain PPO)"
              f"\n  env={env_id} timeout={node.timeout} goal={node.goal.spec}"
              f"\n  seed={args.seed} budget={args.budget:,} env steps "
              f"eval_every={args.eval_every:,}\n{'=' * 72}")
        report, _model = train_primitive(
            node, [env_id], NO_HOOK,
            total_timesteps=args.budget, eval_every=args.eval_every,
            n_eval=args.n_eval, n_envs=args.n_envs, seed=args.seed,
            device=args.device, verbose=1,
        )
        print(f"  steps={report['steps_trained']:,} wall={report['wall_seconds']}s "
              f"({report['steps_per_second']} steps/s) stable={report['stable']}")
        report["env_steps_own"] = report["steps_trained"]
        report["env_steps_children"] = 0
        report["env_steps_total"] = report["steps_trained"]
        report["curve"] = curve(report, steps_per_x=1.0, eval_every=args.eval_every)
        learned = node.policy

    else:
        print("loading the frozen primitives:")
        children = load_option_children(args.task, registry, device=args.device)
        node = build_flat_options(children, args.task)
        print(f"\n{'=' * 72}\n{node.id}  n={node.n}  "
              f"children={[c.id for c in node.children]}"
              f"\n  env={env_id} timeout={node.timeout} "
              f"max_decisions={node.max_decisions}"
              f"\n  seed={args.seed} budget={args.budget:,} decisions\n{'=' * 72}")
        report, _model = train_composite(
            node, [env_id], NO_HOOK,
            total_timesteps=args.budget, eval_every=args.eval_every,
            n_eval=args.n_eval, n_envs=args.n_envs, seed=args.seed,
            device=args.device, verbose=1,
        )
        print(f"  decisions={report['decisions_trained']:,} "
              f"env_steps_own={report['env_steps_own']:,} "
              f"({report['env_steps_per_decision']}/decision) "
              f"wall={report['wall_seconds']}s stable={report['stable']}")
        report["curve"] = curve(
            report,
            steps_per_x=float(report["env_steps_per_decision"] or 1.0),
            eval_every=args.eval_every,
        )
        learned = node.policy

    # Held-out score, through the same evaluation path as every other node.
    heldout_fn = (
        pooled_eval_fn(node, [env_id], NO_HOOK, n_episodes=args.n_eval,
                       seed_base=heldout_seeds(1)[0])
        if node.is_primitive else
        pooled_eval_fn(node, [env_id], NO_HOOK, n_episodes=args.n_eval,
                       seed_base=heldout_seeds(1)[0],
                       env_factory_fn=execution_env_factory)
    )
    heldout = heldout_fn()
    report["heldout"] = heldout.as_dict()
    print(f"  held-out: {heldout}")

    if report["stable"]:
        node.freeze()
        baseline_registry.save(node, extra_meta=report)
        print(f"  saved to runs/v2/baselines/{baseline_registry.node_dir(node.id).name}/ "
              "(baseline, never a child)")
    result["report"] = report

    # The five C1 arms, for the options baseline only -- a primitive has no
    # children to mask, so the arms are undefined for it.
    if not node.is_primitive:
        seeds = heldout_seeds(args.n_eval)
        arms = run_arms(
            lambda: build_flat_options(node.children, args.task), env_id, NO_HOOK, seeds,
            learned_policy=learned, rng_seed=args.seed,
        )
        result["arms"] = {k: {"label": ARM_LABELS[k], **v.as_dict()}
                          for k, v in arms.items()}
        print(f"\n--- C1 arms, {args.n_eval} held-out episodes ---")
        for name, res in arms.items():
            print(f"  {ARM_LABELS[name]:<44} SR {res.success_rate:5.2f}  "
                  f"[{res.wilson_lower:.2f},{res.wilson_upper:.2f}]  "
                  f"mean env steps {res.mean_length:6.1f}")
        result["mask_divergence"] = mask_divergence(node, env_id, NO_HOOK, seeds[:20])
        result["trace"] = capture_trace(node, env_id, NO_HOOK, seeds[0])
        print(f"  max stack depth {result['trace']['max_stack_depth']}, "
              f"selected: {' -> '.join(result['trace']['selected_children'])}")

    suffix = "_smoke" if args.smoke else ""
    default = (f"flat_{args.which}{suffix}_s{args.seed}.json" if args.task == "doorkey"
               else f"flat_{args.task}_{args.which}{suffix}_s{args.seed}.json")
    name = args.out_name or default
    path = RUNS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
