"""
H-5: train and freeze the first composite, {1,1} unlock_door = (get_key, open_door).

The first real test of the hierarchy: a composite trained as an SMDP, where one
decision is a whole child execution.

Under goal-only masking [A: F1] both children are offered at every decision, so
the composite has to learn that `open_door` fails when it is not holding the
key -- that failure is the signal. The two C1 controls are run here, where they
are cheap, so the headline number is never reported alone:

  (a) random over the available children -- how far does masking alone get?
  (b) the same training run with initiation masking switched on -- how much does
      the stricter mask contribute?

Usage:
    python scripts/train_unlock_door.py --smoke
    python scripts/train_unlock_door.py

[H-7a.3] Re-measuring the node's own cost under the composite setting H-6.8
introduced, without touching the frozen checkpoint:

    python scripts/train_unlock_door.py --no-freeze --n-steps 64 --eval-every 512 \
        --budget 51200 --skip-ablation --seed 43 --out-name unlock_door_h7_s43.json

`--no-freeze` matters: the registry keys by node id, so a stable run that saved
would overwrite the frozen `{1,1} unlock_door` every later node was built on.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.envs.subgoal_env import NO_HOOK  # noqa: E402
from src.skills.node import SkillNode  # noqa: E402
from src.skills.predicates import door_locked, door_open  # noqa: E402
from src.skills.registry import NodeRegistry  # noqa: E402
from src.training.device import describe  # noqa: E402
from src.training.evaluation import evaluate_node, heldout_seeds, stopping_seeds  # noqa: E402
from src.training.node_trainer import (  # noqa: E402
    RandomOverAvailable,
    execution_env_factory,
    load_frozen_primitive,
    pooled_eval_fn,
    set_torch_threads,
    train_composite,
)
from src.types import NodeStatus  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
from train_primitives import build_node, specs  # noqa: E402

UNLOCK = "MiniGrid-Unlock-v0"
RUNS = REPO / "runs" / "v2"

# [A: F13] 120 (get_key) <= 240 <= 288 (Unlock's own max_steps).
T3_TIMEOUT = 240
T3_MAX_DECISIONS = 8


def load_children(registry, device="cpu"):
    """Rebuild T1 and T2 exactly as H-4 declared them, then load their weights."""
    all_specs = specs()
    kids = []
    for name in ("get_key", "open_door"):
        spec = all_specs[name]
        node = build_node(spec)
        load_frozen_primitive(node, registry, spec["env_ids"][0], spec["hook"],
                              device=device)
        if node.status is not NodeStatus.FROZEN:
            raise RuntimeError(f"{node.id} did not load as FROZEN")
        print(f"  loaded {node} spec_hash={node.spec_hash()}")
        kids.append(node)
    return kids


def build_t3(children):
    return SkillNode(
        "{1,1} unlock_door",
        children=children,
        goal=door_open(),
        # [H-6 A.3] v2 §2 requires initial conditions on every node, not only on
        # primitives, and H-5 measured what the ALWAYS default costs: the strict
        # mask then differed from the goal-only mask in 0 of 304 decisions and
        # the C1 masking ablation on this node was vacuous. There is nothing to
        # unlock unless the door is locked.
        #
        # Declaring it changes this node's spec_hash, so the H-5 checkpoint is
        # correctly refused by the registry (F18) and the node is retrained
        # (H-6.2). `initiation` does not mask under goal-only training, so the
        # trajectory is expected to be identical.
        initiation=door_locked(),
        timeout=T3_TIMEOUT,
        max_decisions=T3_MAX_DECISIONS,
        env_ids=[UNLOCK],
        m=1,
    )


def control_random(children, n_eval, seed=0):
    """C1 control (a): uniform over the available children."""
    node = build_t3(children)
    node.policy = RandomOverAvailable(seed=seed)
    return pooled_eval_fn(
        node, [UNLOCK], NO_HOOK, n_episodes=n_eval,
        seed_base=heldout_seeds(1)[0], env_factory_fn=execution_env_factory,
    )()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--budget", type=int, default=60_000, help="SMDP decisions")
    ap.add_argument("--eval-every", type=int, default=2_000)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--out-name", default=None,
                    help="output file name; defaults to unlock_door[_smoke].json")
    ap.add_argument("--no-freeze", action="store_true",
                    help="train and report but never write to the registry")
    ap.add_argument("--n-steps", type=int, default=None,
                    help="MaskablePPO n_steps (and batch_size); default COMPOSITE_N_STEPS")
    args = ap.parse_args()

    if args.smoke:
        args.budget, args.eval_every, args.n_envs, args.n_eval = 600, 200, 2, 10

    print(describe())
    print(f"torch threads: {set_torch_threads(args.torch_threads)}")
    registry = NodeRegistry(RUNS)
    print("loading frozen children:")
    children = load_children(registry, device=args.device)

    out: dict = {}

    # ---- C1 control (a): random over available children --------------
    rnd = control_random(children, args.n_eval)
    out["control_random_over_available"] = rnd.as_dict()
    print(f"\nC1 control (a) random-over-available: {rnd}")

    # ---- the composite ------------------------------------------------
    node = build_t3(children)
    print(f"\n{'='*72}\n{node.id}  n={node.n}  children={[c.id for c in node.children]}"
          f"\n  timeout={node.timeout} max_decisions={node.max_decisions}\n{'='*72}")

    overrides = ({} if args.n_steps is None else
                 {"n_steps": args.n_steps, "batch_size": args.n_steps})
    report, _model = train_composite(
        node, [UNLOCK], NO_HOOK,
        total_timesteps=args.budget, eval_every=args.eval_every,
        n_eval=args.n_eval, n_envs=args.n_envs, seed=args.seed,
        device=args.device, ppo_overrides=overrides, verbose=1,
    )
    print(f"  decisions={report['decisions_trained']:,} wall={report['wall_seconds']}s "
          f"({report['decisions_per_second']} decisions/s) stable={report['stable']}")

    if report["stable"]:
        node.freeze()
        heldout = pooled_eval_fn(
            node, [UNLOCK], NO_HOOK, n_episodes=args.n_eval,
            seed_base=heldout_seeds(1)[0], env_factory_fn=execution_env_factory,
        )()
        report["heldout"] = heldout.as_dict()
        if args.no_freeze or args.smoke:
            print(f"  stable, not saved (--no-freeze/--smoke). held-out: {heldout}")
        else:
            registry.save(node, extra_meta=report)
            print(f"  FROZEN. held-out: {heldout}")
    else:
        print("  not stable within budget -- not frozen")
    out["composite"] = report

    # ---- C1 control (b): initiation masking on -------------------------
    if not args.skip_ablation:
        print(f"\n{'='*72}\nC1 control (b): the same run with initiation masking ON"
              f"\n{'='*72}")
        abl_node = build_t3(children)
        abl_report, _ = train_composite(
            abl_node, [UNLOCK], NO_HOOK,
            total_timesteps=args.budget, eval_every=args.eval_every,
            n_eval=args.n_eval, n_envs=args.n_envs, seed=args.seed,
            device=args.device, initiation_masking=True, ppo_overrides=overrides,
            verbose=1,
        )
        out["ablation_initiation_masking"] = abl_report
        print(f"  decisions={abl_report['decisions_trained']:,} "
              f"stable={abl_report['stable']}")

    default_name = "unlock_door_smoke.json" if args.smoke else "unlock_door.json"
    path = RUNS / (args.out_name or default_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path.relative_to(REPO)}")

    print("\n--- C1 summary ---")
    print(f"  random over available : SR {rnd.success_rate:.3f}")
    comp = out["composite"]
    sr = comp.get("heldout", {}).get("success_rate")
    print(f"  learned composite     : SR {sr if sr is not None else 'not stable'}"
          f" ({comp['decisions_trained']:,} decisions)")
    if "ablation_initiation_masking" in out:
        a = out["ablation_initiation_masking"]
        print(f"  + initiation masking  : stable={a['stable']} "
              f"({a['decisions_trained']:,} decisions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
