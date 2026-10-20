"""
H-7a.2: train the H-7a primitives, `{0,9} drop_held` and `{0,4} pickup_box`.

Specs live in `scripts/h7_nodes.py` (v2 §11 T6/T6b, H-7 amendment 1). Same PPO
configuration and stopping rule as every other primitive: SR >= 0.90 and Wilson
LB >= 0.80 twice in a row on the fixed stopping seeds, held-out score at freeze.
No per-node stopping rule. Budgets are the amendment's and are never extended:

    drop_held    100,000 env steps, eval_every 10,000
    pickup_box 1,000,000 env steps, eval_every 20,000

Seed 42 is canonical and is frozen into `runs/v2/0-9_drop_held/` and
`runs/v2/0-4_pickup_box/`. Other seeds run with `--no-freeze --out-name
<name>_s43`: the registry keys by node id, so a second seed that saved would
overwrite the frozen node. They exist only to put a range on the cost.

`{0,5} move_blocker` is H-7b and is not trainable from here until H-7b is
planned (amendment 1, §2).

Usage:
    python scripts/train_h7_primitives.py --only drop_held --smoke
    python scripts/train_h7_primitives.py --only drop_held --seed 42
    python scripts/train_h7_primitives.py --only pickup_box --seed 43 --no-freeze --out-name pickup_box_s43
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
from h7_nodes import H7A_BUDGETS, RUNS, h7_primitive_specs  # noqa: E402
from train_primitives import build_node  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", required=True, choices=sorted(H7A_BUDGETS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-freeze", action="store_true")
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--budget", type=int, default=None,
                    help="env steps; default the amendment's fixed budget for the node")
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--torch-threads", type=int, default=2)
    args = ap.parse_args()

    if args.budget is None:
        args.budget = H7A_BUDGETS[args.only]["budget"]
    if args.eval_every is None:
        args.eval_every = H7A_BUDGETS[args.only]["eval_every"]
    if args.smoke:
        args.budget, args.eval_every, args.n_envs, args.n_eval = 8_000, 4_096, 4, 10

    print(describe())
    print(f"torch threads: {set_torch_threads(args.torch_threads)}")
    spec = h7_primitive_specs()[args.only]
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
    name = args.out_name or f"{args.only}_s{args.seed}"
    path = RUNS / f"{name}{suffix}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] wrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
