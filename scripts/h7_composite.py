"""
H-7.4 / H-7.6: one composite experiment, the same way every time.

`train_unlock_pickup.py` and `train_blocked_pickup.py` differ only in which node
they build and which env it runs in; everything that makes the result
interpretable is here, once, so the three composites of H-7 are reported
identically (plan success criterion 2):

  composite                 trained under the goal-only mask, held-out score
  trained_with_strict_mask  the same node trained under the strict mask, scored
                            under the strict mask
  arms                      the five C1 arms on the held-out seeds
  mask_divergence           of the goal-only policy
  trace                     one held-out episode, with the stack depth reached
  chain_ceiling             product of the children's F29 handoff rows (v2 §11,
                            H-7 A.6) -- a composite cannot exceed it
  selection_sequences       the mechanism measure, both trained conditions

`n_steps` 64 (`COMPOSITE_N_STEPS`), budget 40,960 decisions = 80 updates,
`eval_every` 512 = one update (v2 §5.2/§5.3).
"""

from __future__ import annotations

import argparse
import json
import time

from src.envs.subgoal_env import NO_HOOK
from src.skills.registry import NodeRegistry
from src.training.c1_controls import (
    ARM_LABELS,
    capture_trace,
    mask_divergence,
    run_arms,
    selection_sequences,
)
from src.training.device import describe
from src.training.evaluation import heldout_seeds
from src.training.node_trainer import (
    WEAK_HELDOUT_LB,
    execution_env_factory,
    pooled_eval_fn,
    set_torch_threads,
    train_composite,
)

from h7_nodes import REPO, RUNS

BUDGET = 40_960
EVAL_EVERY = 512


def common_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--budget", type=int, default=BUDGET, help="SMDP decisions")
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--no-freeze", action="store_true")
    ap.add_argument("--skip-strict", action="store_true")
    ap.add_argument("--out-name", default=None)
    return ap


def heldout(node, env_id, n, *, strict=False):
    return pooled_eval_fn(node, [env_id], NO_HOOK, n_episodes=n,
                          seed_base=heldout_seeds(1)[0],
                          env_factory_fn=execution_env_factory,
                          initiation_masking=strict)()


def deepest_trace(node, env_id, seeds, tries=10):
    """The first successful held-out episode's trace (seeds[0] if none succeeds)."""
    first = None
    for s in seeds[:tries]:
        t = capture_trace(node, env_id, NO_HOOK, s)
        first = first or t
        if t["success"]:
            return t
    return first


def run(args, build, env_id, *, chain_ceiling: dict, default_name: str,
        expected_depth: int) -> dict:
    if args.smoke:
        args.budget, args.eval_every, args.n_envs, args.n_eval = 1_024, 512, 8, 10
    print(describe())
    print(f"torch threads: {set_torch_threads(args.torch_threads)}")
    registry = NodeRegistry(RUNS)

    node = build()
    print(f"\n{'=' * 72}\n{node.id}  n={node.n}  children={[c.id for c in node.children]} "
          f"(levels {[c.n for c in node.children]})\n  env={env_id} timeout={node.timeout} "
          f"max_decisions={node.max_decisions}\n  seed={args.seed} budget={args.budget:,} "
          f"decisions eval_every={args.eval_every}\n{'=' * 72}")
    out: dict = {"seed": args.seed, "node_id": node.id, "env_id": env_id,
                 "spec_hash": node.spec_hash(), "chain_ceiling": chain_ceiling}

    report, _ = train_composite(
        node, [env_id], NO_HOOK, total_timesteps=args.budget, eval_every=args.eval_every,
        n_eval=args.n_eval, n_envs=args.n_envs, seed=args.seed, device=args.device,
        verbose=1,
    )
    print(f"  decisions={report['decisions_trained']:,} updates={report['policy_updates']} "
          f"env_steps_own={report['env_steps_own']:,} "
          f"children={report['env_steps_children']:,} stable={report['stable']}")
    if report["stable"]:
        node.freeze()
    h = heldout(node, env_id, args.n_eval)
    report["heldout"] = h.as_dict()
    report["weak_heldout"] = bool(h.wilson_lower < WEAK_HELDOUT_LB)
    print(f"  held-out: {h}")
    if report["stable"] and not args.no_freeze and not args.smoke:
        registry.save(node, extra_meta=report)
        print(f"  FROZEN and saved to {registry.node_dir(node.id).name}/")
    out["composite"] = report

    strict_policy = None
    if not args.skip_strict:
        print(f"\n{'=' * 72}\nC1 arm: the same node trained with initiation masking ON\n{'=' * 72}")
        abl = build()
        abl_report, _ = train_composite(
            abl, [env_id], NO_HOOK, total_timesteps=args.budget,
            eval_every=args.eval_every, n_eval=args.n_eval, n_envs=args.n_envs,
            seed=args.seed, device=args.device, initiation_masking=True, verbose=1,
        )
        ah = heldout(abl, env_id, args.n_eval, strict=True)
        abl_report["heldout_strict_scored"] = ah.as_dict()
        print(f"  decisions={abl_report['decisions_trained']:,} "
              f"env_steps_own={abl_report['env_steps_own']:,} stable={abl_report['stable']}"
              f"  held-out (strict mask): {ah}")
        out["trained_with_strict_mask"] = abl_report
        strict_policy = abl.policy

    seeds = heldout_seeds(args.n_eval)
    arms = run_arms(build, env_id, NO_HOOK, seeds, learned_policy=node.policy,
                    learned_strict_policy=strict_policy or node.policy, rng_seed=args.seed)
    out["arms"] = {k: {"label": ARM_LABELS[k], **v.as_dict()} for k, v in arms.items()}
    out["arms_note"] = ("learned_strict_mask is the separately strict-trained policy"
                        if strict_policy is not None else
                        "learned_strict_mask is the goal-only policy scored under the strict mask")
    print(f"\n--- C1 arms, {args.n_eval} held-out episodes ---")
    for k, v in arms.items():
        print(f"  {ARM_LABELS[k]:<44} SR {v.success_rate:5.2f}  "
              f"[{v.wilson_lower:.2f},{v.wilson_upper:.2f}]  mean env steps {v.mean_length:6.1f}")

    out["mask_divergence"] = mask_divergence(node, env_id, NO_HOOK, seeds[:20])
    d = out["mask_divergence"]
    print(f"\nmask divergence: {d['masks_differ']}/{d['decisions']} decisions differ, "
          f"{d['would_have_blocked_learned_choice']} would have blocked the learned choice")

    out["selection_sequences"] = {
        "learned": selection_sequences(node, env_id, NO_HOOK, seeds),
    }
    if strict_policy is not None:
        probe = build()
        probe.policy = strict_policy
        out["selection_sequences"]["learned_strict_mask"] = selection_sequences(
            probe, env_id, NO_HOOK, seeds, initiation_masking=True)
    for k, v in out["selection_sequences"].items():
        print(f"\nselection sequences ({k}):")
        for r in v["sequences"]:
            print(f"  {r['count']:>3}  ({r['successes']} ok)  {r['sequence']}")

    trace = deepest_trace(node, env_id, seeds)
    out["trace"] = trace
    out["depth_check"] = {"expected": expected_depth, "reached": trace["max_stack_depth"],
                          "passed": trace["max_stack_depth"] >= expected_depth,
                          "levels_selected": trace["levels_selected"]}
    print(f"\ndepth check: max stack depth {trace['max_stack_depth']} (expected {expected_depth}), "
          f"levels {trace['levels_selected']}, seed {trace['seed']}, success={trace['success']}")
    print(f"  selected: {' -> '.join(trace['selected_children'])}")
    print(f"chain ceiling: {chain_ceiling}")

    suffix = "_smoke" if args.smoke else ""
    path = RUNS / f"{args.out_name or default_name}{suffix}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[{time.strftime('%H:%M:%S')}] wrote {path.relative_to(REPO)}")
    return out


def f29(stage: str, start_state: str) -> float:
    """A success rate from `runs/v2/h7_preflight.json`, by stage and row label."""
    doc = json.loads((RUNS / "h7_preflight.json").read_text(encoding="utf-8"))
    for r in doc[stage]["rows"]:
        if r["start_state"] == start_state:
            return r["success_rate"]
    raise KeyError(f"{stage}: no row '{start_state}'")
