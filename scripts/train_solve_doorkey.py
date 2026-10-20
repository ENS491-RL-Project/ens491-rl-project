"""
H-6.3: train and freeze T5 `{2,1} solve_doorkey` = composite(unlock_door, go_to_goal).

The first depth-2 node, and the first one whose own child is itself a composite:
one episode drives the stack `solve_doorkey -> unlock_door -> get_key`, which is
the mixed-level DAG v2 §11 claims the curriculum exercises. The claim is checked
here, not asserted -- `--trace` stores one episode's execution trace with the
depth each frame reached.

Three seeds, because a single seed says nothing about variance and H-4 already
showed evaluations at SR exactly 0.900 that the Wilson bound correctly refused.
Seed 42 is the canonical T5 and the only one frozen into the registry; the other
two are evidence about variance, not separate skills.

All five C1 arms are reported together (`src/training/c1_controls.py`). The
prediction written into the plan before this ran: the initiation-masked random
control **matches** the learned composite, because solve_doorkey's subgoal order
is strict just as unlock_door's was. If that happens it is the predicted null
result (assumption A2), not a failure -- and it is why H-6.6 exists.

Usage:
    python scripts/train_solve_doorkey.py --smoke
    python scripts/train_solve_doorkey.py --seed 42
    python scripts/train_solve_doorkey.py --seed 43 --no-freeze
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.envs.subgoal_env import NO_HOOK  # noqa: E402
from src.skills.node import SkillNode  # noqa: E402
from src.skills.predicates import door_locked, on_goal_tile  # noqa: E402
from src.skills.registry import NodeRegistry  # noqa: E402
from src.training.c1_controls import (  # noqa: E402
    ARM_LABELS,
    capture_trace,
    mask_divergence,
    run_arms,
)
from src.training.device import describe  # noqa: E402
from src.training.evaluation import heldout_seeds  # noqa: E402
from src.training.node_trainer import (  # noqa: E402
    execution_env_factory,
    load_frozen_composite,
    load_frozen_primitive,
    pooled_eval_fn,
    set_torch_threads,
    train_composite,
)
from src.types import NodeStatus  # noqa: E402
from train_primitives import DOORKEY, UNLOCK, build_node, specs  # noqa: E402
from train_unlock_door import build_t3  # noqa: E402

RUNS = REPO / "runs" / "v2"

# [A: F13] children are 240 (unlock_door) and 120 (go_to_goal), so the parent
# must outlive 240; DoorKey-6x6's own max_steps is 360, which is the ceiling a
# root timeout may not exceed. 360 is therefore the only value that gives the
# node the whole episode without breaking either half of the invariant.
T5_TIMEOUT = 360
# [A: F2] Mandatory. Under goal-only masking the parent may re-select a child
# that just timed out, and go_to_goal with the door shut burns its full 120
# steps, so three such picks would exhaust the episode on their own.
T5_MAX_DECISIONS = 8


def load_children(registry, device="cpu"):
    """
    Children before parents: `spec_hash` recurses into the children's hashes, so
    `{1,1} unlock_door` cannot be verified until its own two primitives are
    loaded [A: F18].
    """
    all_specs = specs()
    prims = {}
    for name in ("get_key", "open_door", "go_to_goal"):
        spec = all_specs[name]
        node = build_node(spec)
        load_frozen_primitive(node, registry, spec["env_ids"][0], spec["hook"],
                              device=device)
        if node.status is not NodeStatus.FROZEN:
            raise RuntimeError(f"{node.id} did not load as FROZEN")
        prims[name] = node
        print(f"  loaded {node}  spec_hash={node.spec_hash()}")

    unlock_door = build_t3([prims["get_key"], prims["open_door"]])
    load_frozen_composite(unlock_door, registry, UNLOCK, NO_HOOK, device=device)
    print(f"  loaded {unlock_door}  spec_hash={unlock_door.spec_hash()}")
    return unlock_door, prims["go_to_goal"]


def build_t5(unlock_door, go_to_goal):
    return SkillNode(
        "{2,1} solve_doorkey",
        children=[unlock_door, go_to_goal],
        # [A: F10] The node's OWN goal, read off the StateView -- never the env's
        # reward. On DoorKey the two coincide, which is exactly what makes the
        # comparison against flat PPO in H-6.4 fair.
        goal=on_goal_tile(),
        # [H-6 A.3] There is nothing to solve if the door is already open.
        initiation=door_locked(),
        timeout=T5_TIMEOUT,
        max_decisions=T5_MAX_DECISIONS,
        env_ids=[DOORKEY],
        m=1,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--budget", type=int, default=20_000, help="SMDP decisions")
    ap.add_argument("--eval-every", type=int, default=1_000)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--no-freeze", action="store_true",
                    help="train and report but do not write to the registry")
    ap.add_argument("--skip-strict", action="store_true",
                    help="skip the separately trained strict-mask arm")
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()

    if args.smoke:
        args.budget, args.eval_every, args.n_envs, args.n_eval = 600, 200, 2, 10

    print(describe())
    print(f"torch threads: {set_torch_threads(args.torch_threads)}")
    registry = NodeRegistry(RUNS)
    print("loading the frozen children:")
    unlock_door, go_to_goal = load_children(registry, device=args.device)

    node = build_t5(unlock_door, go_to_goal)
    print(f"\n{'=' * 72}\n{node.id}  n={node.n}  "
          f"children={[c.id for c in node.children]} (levels {[c.n for c in node.children]})"
          f"\n  env={DOORKEY} timeout={node.timeout} max_decisions={node.max_decisions}"
          f"\n  seed={args.seed} budget={args.budget:,} decisions\n{'=' * 72}")

    out: dict = {"seed": args.seed, "node_id": node.id, "env_id": DOORKEY}

    report, model = train_composite(
        node, [DOORKEY], NO_HOOK,
        total_timesteps=args.budget, eval_every=args.eval_every,
        n_eval=args.n_eval, n_envs=args.n_envs, seed=args.seed,
        device=args.device, verbose=1,
    )
    print(f"  decisions={report['decisions_trained']:,} "
          f"env_steps_own={report['env_steps_own']:,} "
          f"({report['env_steps_per_decision']}/decision) "
          f"wall={report['wall_seconds']}s stable={report['stable']}")

    learned_policy = node.policy
    if report["stable"]:
        node.freeze()
        heldout = pooled_eval_fn(
            node, [DOORKEY], NO_HOOK, n_episodes=args.n_eval,
            seed_base=heldout_seeds(1)[0], env_factory_fn=execution_env_factory,
        )()
        report["heldout"] = heldout.as_dict()
        print(f"  held-out: {heldout}")
        if not args.no_freeze:
            registry.save(node, extra_meta=report)
            print(f"  FROZEN and saved to {registry.node_dir(node.id).name}/")
    else:
        print("  not stable within budget -- not frozen")
    out["composite"] = report

    # ---- the separately trained strict-mask arm ------------------------
    strict_policy = None
    if not args.skip_strict:
        print(f"\n{'=' * 72}\nC1 arm: the same run trained with initiation masking ON"
              f"\n{'=' * 72}")
        abl = build_t5(unlock_door, go_to_goal)
        abl_report, _ = train_composite(
            abl, [DOORKEY], NO_HOOK,
            total_timesteps=args.budget, eval_every=args.eval_every,
            n_eval=args.n_eval, n_envs=args.n_envs, seed=args.seed,
            device=args.device, initiation_masking=True, verbose=1,
        )
        print(f"  decisions={abl_report['decisions_trained']:,} "
              f"env_steps_own={abl_report['env_steps_own']:,} "
              f"stable={abl_report['stable']}")
        out["trained_with_strict_mask"] = abl_report
        strict_policy = abl.policy

    # ---- all five arms on the same held-out seeds ----------------------
    seeds = heldout_seeds(args.n_eval)
    arms = run_arms(
        lambda: build_t5(unlock_door, go_to_goal), DOORKEY, NO_HOOK, seeds,
        learned_policy=learned_policy,
        learned_strict_policy=strict_policy or learned_policy,
        rng_seed=args.seed,
    )
    out["arms"] = {k: {"label": ARM_LABELS[k], **v.as_dict()} for k, v in arms.items()}

    print(f"\n--- C1 arms, {args.n_eval} held-out episodes "
          f"(seed base {seeds[0]}) ---")
    for name, res in arms.items():
        print(f"  {ARM_LABELS[name]:<44} SR {res.success_rate:5.2f}  "
              f"[{res.wilson_lower:.2f},{res.wilson_upper:.2f}]  "
              f"mean env steps {res.mean_length:6.1f}")

    # ---- mask divergence ----------------------------------------------
    div = mask_divergence(node, DOORKEY, NO_HOOK, seeds[:20])
    out["mask_divergence"] = div
    print(f"\nmask divergence over {div['episodes']} episodes of the learned policy:")
    print(f"  root decisions                                      {div['decisions']}")
    print(f"  where the strict mask differs from goal-only        {div['masks_differ']}")
    print(f"  of those, where it would have blocked the choice    "
          f"{div['would_have_blocked_learned_choice']}")

    # ---- depth check (plan assumption A6) ------------------------------
    trace = capture_trace(node, DOORKEY, NO_HOOK, seeds[0])
    out["trace"] = trace
    print(f"\ndepth check (A6): max stack depth {trace['max_stack_depth']}, "
          f"levels selected {trace['levels_selected']}, "
          f"success={trace['success']} in {trace['env_steps']} env steps")
    print(f"  selected: {' -> '.join(trace['selected_children'])}")
    if trace["max_stack_depth"] < 3:
        print("  A6 CONTRADICTED: the stack never reached depth 3. The executor is "
              "not composing; stop and report.")

    name = args.out_name or (
        f"solve_doorkey_smoke_s{args.seed}.json" if args.smoke
        else f"solve_doorkey_s{args.seed}.json"
    )
    path = RUNS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
