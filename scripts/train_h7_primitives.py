"""
H-7.3 / H-7.5: train the two H-7 primitives, `{0,4} pickup_box` and `{0,5} move_blocker`.

Specs live in `scripts/h7_nodes.py` (v2 §11 T6/T8, amended H-7 A.1/A.2). Same
PPO configuration and stopping rule as every other primitive: 600,000 env steps,
`eval_every` 20,000, SR >= 0.90 and Wilson LB >= 0.80 twice in a row on the fixed
stopping seeds, held-out score at freeze. No per-node stopping rule.

Seed 42 is canonical and is frozen into `runs/v2/0-4_pickup_box/` (or
`0-5_move_blocker/`). Other seeds run with `--no-freeze --out-name <name>_s43`:
the registry keys by node id, so a second seed that saved would overwrite the
frozen node. They exist only to put a range on the cost.

Usage:
    python scripts/train_h7_primitives.py --only pickup_box --smoke
    python scripts/train_h7_primitives.py --only pickup_box --seed 42
    python scripts/train_h7_primitives.py --only pickup_box --seed 43 --no-freeze --out-name pickup_box_s43
    python scripts/train_h7_primitives.py --only move_blocker --fallback   # Part B fallback hook
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

from src.skills.registry import NodeRegistry  # noqa: E402
from src.training.device import describe  # noqa: E402
from src.training.evaluation import heldout_seeds  # noqa: E402
from src.training.node_trainer import (  # noqa: E402
    WEAK_HELDOUT_LB,
    freeze_and_save,
    pooled_eval_fn,
    set_torch_threads,
    train_primitive,
)
from h7_nodes import RUNS, h7_primitive_specs, move_blocker_fallback_spec  # noqa: E402
from train_primitives import build_node  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", required=True, choices=["pickup_box", "move_blocker"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-freeze", action="store_true")
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--fallback", action="store_true",
                    help="move_blocker only: the empty-handed fallback hook of Part B")
    ap.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--budget", type=int, default=600_000)
    ap.add_argument("--eval-every", type=int, default=20_000)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--torch-threads", type=int, default=2)
    args = ap.parse_args()

    if args.smoke:
        args.budget, args.eval_every, args.n_envs, args.n_eval = 8_000, 4_096, 4, 10
    if args.fallback and args.only != "move_blocker":
        ap.error("--fallback applies to move_blocker only")

    print(describe())
    print(f"torch threads: {set_torch_threads(args.torch_threads)}")
    spec = move_blocker_fallback_spec() if args.fallback else h7_primitive_specs()[args.only]
    node = build_node(spec)
    print(f"\n{'=' * 72}\n{node.id}  envs={spec['env_ids']}  timeout={node.timeout}"
          f"\n  hook={spec['hook'].spec}\n  goal={node.goal.spec}  initiation={node.initiation.spec}"
          f"\n  seed={args.seed} budget={args.budget:,} eval_every={args.eval_every:,}"
          f"  freeze={not args.no_freeze and not args.smoke}\n{'=' * 72}")

    report, _model = train_primitive(
        node, spec["env_ids"], spec["hook"],
        total_timesteps=args.budget, eval_every=args.eval_every, n_eval=args.n_eval,
        n_envs=args.n_envs, seed=args.seed, device=args.device, verbose=1,
    )
    print(f"  steps={report['steps_trained']:,} wall={report['wall_seconds']}s "
          f"({report['steps_per_second']} steps/s) stable={report['stable']}")
    report["spec_hash"] = node.spec_hash()
    report["fallback_hook"] = bool(args.fallback)

    registry = NodeRegistry(RUNS)
    if report["stable"] and not args.no_freeze and not args.smoke:
        report = freeze_and_save(node, registry, report,
                                 env_ids=spec["env_ids"], reset_hook=spec["hook"],
                                 n_eval=args.n_eval)
        print(f"  FROZEN into {registry.node_dir(node.id).name}/  held-out: {report['heldout']}")
    else:
        # Scored on the held-out set all the same, so every seed has a comparable
        # number -- but never written to the registry.
        held = pooled_eval_fn(node, spec["env_ids"], spec["hook"], n_episodes=args.n_eval,
                              seed_base=heldout_seeds(1)[0])()
        report["heldout"] = held.as_dict()
        report["weak_heldout"] = bool(held.wilson_lower < WEAK_HELDOUT_LB)
        why = "smoke" if args.smoke else "--no-freeze" if args.no_freeze else "not stable"
        print(f"  not frozen ({why}); held-out: {held}")
        if not report["stable"] and not args.no_freeze and not args.smoke:
            registry.save(node, extra_meta=report)

    suffix = "_smoke" if args.smoke else ""
    name = args.out_name or f"{args.only}{'_fallback' if args.fallback else ''}_s{args.seed}"
    path = RUNS / f"{name}{suffix}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] wrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
