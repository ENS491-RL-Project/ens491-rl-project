"""
H-4: train and freeze the three primitives T1, T2 and T4.

    T1 {0,1} get_key      Unlock + DoorKey-6x6   goal: carrying a key
    T2 {0,2} open_door    Unlock + DoorKey-6x6   goal: the door is open   (starts holding the key)
    T4 {0,3} go_to_goal   DoorKey-6x6            goal: on the goal tile   (starts with the door open)

Node ids follow v2 §11: `{0,3}` is go_to_goal and `{0,4}` is pickup_box.

[A: F29] T1 and T2 train on BOTH layouts with a randomized start pose. Unlock is
an 11x6 two-room grid and DoorKey-6x6 is 6x6, so a primitive trained only on
Unlock would very likely fail inside `{2,1} solve_doorkey`. This is on the
critical path, not an optional generality experiment.

[A: F13] Timeouts fit inside the env budgets: Unlock 288, DoorKey-6x6 360, and
get_key + open_door + go_to_goal = 120 + 60 + 120 = 300 <= 360.

Usage:
    python scripts/train_primitives.py --smoke      # tiny budget, prints steps/s
    python scripts/train_primitives.py              # full run
    python scripts/train_primitives.py --only get_key
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from src.envs.subgoal_env import (  # noqa: E402
    compose,
    give_key,
    open_door,
    randomize_agent_pose,
    randomize_carrying,
)
from src.skills.node import SkillNode  # noqa: E402
from src.skills.predicates import carrying, door_open, on_goal_tile  # noqa: E402
from src.skills.registry import NodeRegistry  # noqa: E402
from src.training.device import describe, resolve_device  # noqa: E402
from src.training.node_trainer import freeze_and_save, train_primitive  # noqa: E402

UNLOCK = "MiniGrid-Unlock-v0"
DOORKEY = "MiniGrid-DoorKey-6x6-v0"
RUNS = REPO / "runs" / "v2"


def specs():
    return {
        "get_key": dict(
            node_id="{0,1} get_key",
            m=1,
            goal=carrying("key"),
            timeout=120,
            env_ids=[UNLOCK, DOORKEY],
            hook=randomize_agent_pose(anchor="key"),
        ),
        "open_door": dict(
            node_id="{0,2} open_door",
            m=2,
            goal=door_open(),
            timeout=60,
            env_ids=[UNLOCK, DOORKEY],
            hook=compose(give_key(), randomize_agent_pose(anchor=None)),
        ),
        "go_to_goal": dict(
            node_id="{0,3} go_to_goal",
            m=3,
            goal=on_goal_tile(),
            timeout=120,
            env_ids=[DOORKEY],
            # randomize_carrying: unlock_door hands this node control while the
            # agent is still holding the key, and the held object is visible at
            # the agent's own observation cell [A: F29].
            hook=compose(open_door(), randomize_carrying(), randomize_agent_pose(anchor="goal")),
        ),
    }


def build_node(spec, normalize=True):
    return SkillNode(
        spec["node_id"],
        goal=spec["goal"],
        timeout=spec["timeout"],
        env_ids=spec["env_ids"],
        normalize_obs=normalize,
        m=spec["m"],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny budget; checks the loop")
    ap.add_argument("--only", nargs="*", default=None, help="subset of node names")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--budget", type=int, default=600_000)
    ap.add_argument("--eval-every", type=int, default=20_000)
    args = ap.parse_args()

    if args.smoke:
        args.budget, args.eval_every, args.n_envs = 8_000, 2_000, 4

    print(describe())
    print(f"device={resolve_device(args.device)} n_envs={args.n_envs} "
          f"budget={args.budget:,} eval_every={args.eval_every:,} seed={args.seed}")

    registry = NodeRegistry(RUNS)
    all_specs = specs()
    names = args.only or list(all_specs)
    reports = {}

    for name in names:
        spec = all_specs[name]
        node = build_node(spec)
        print(f"\n{'='*72}\n{node.id}  envs={spec['env_ids']}  "
              f"timeout={node.timeout}  hook={spec['hook'].spec}\n{'='*72}")

        t0 = time.perf_counter()
        report, model = train_primitive(
            node,
            spec["env_ids"],
            spec["hook"],
            total_timesteps=args.budget,
            eval_every=args.eval_every,
            n_envs=args.n_envs,
            seed=args.seed,
            device=args.device,
            verbose=1,
        )
        print(f"  steps={report['steps_trained']:,} "
              f"wall={report['wall_seconds']}s "
              f"throughput={report['steps_per_second']} steps/s "
              f"stable={report['stable']}")

        if report["stable"]:
            report = freeze_and_save(
                node, registry, report,
                env_ids=spec["env_ids"], reset_hook=spec["hook"],
            )
            print(f"  FROZEN. held-out: {report['heldout']}")
        else:
            print("  not stable within budget -- not frozen, nothing saved as FROZEN")
            registry.save(node, extra_meta=report)

        reports[name] = report
        # Written after every node, not once at the end: a long background run
        # that is cut short must still leave the finished nodes on disk.
        out = RUNS / ("primitives_smoke.json" if args.smoke else "primitives.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"  [{time.strftime('%H:%M:%S')}] wrote {out.relative_to(REPO)}", flush=True)

    out = RUNS / ("primitives_smoke.json" if args.smoke else "primitives.json")
    print(f"\nwrote {out.relative_to(REPO)}")

    stable = [n for n, r in reports.items() if r["stable"]]
    print(f"stable: {stable or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
