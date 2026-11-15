"""
W1b-1 steps 2-3 (`docs/plans/2026-11-15_W1b-1_unlock_pickup_plan.md`, revision 3):
`{2,2} unlock_pickup` (T7) over the canonical `unlock_door`, `drop_held` and the `pickup_box`
frozen in step 1 under decision 0014 (`Proposed`). Every node argmax; decision 0004 unchanged.

This experiment runs while 0014 is `Proposed`, with Emir's go-ahead, under `scripts/` only.
Nothing is saved to the registry; no file under `src/` changes; no output is overwritten.

    --gate              ceiling gate, evaluation only: the scripted sequence unlock_door ->
                        drop_held -> pickup_box, argmax, on the stopping seeds 10000-10049 and the
                        held-out seeds 20000-20049. Fewer than 46/50 on the stopping set: W1b-1
                        fails with reading "chain" and nothing is trained.
                        Writes runs/v2/w1b_unlock_pickup_gate.json
    --smoke             h7_composite.run at its smoke size, then the fresh row on 10 seeds;
                        both smoke files are deleted afterwards; prints decisions/s and env
                        steps per decision with a wall-time estimate for one full seed
    --seed S            step 3: train (h7_composite.run, --no-freeze --skip-strict forced),
                        then score the fresh row in the same process on 54000-54499
                        -> runs/v2/w1b_unlock_pickup_s<S>.json and ..._s<S>_fresh.json
    --seed S --rerun-of RUN_ID
                        the plan's rule for a process that died between training and the fresh
                        row: same command, output ..._s<S>_rerun.json (+ _fresh); counts only if
                        its stopping history and policy_sha256 equal RUN_ID's exactly
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
from src.skills.registry import NodeRegistry, SpecMismatch  # noqa: E402
from src.skills.state import build_state_view  # noqa: E402
from src.training.evaluation import heldout_seeds, stopping_seeds, wilson_lower  # noqa: E402
from src.training.node_trainer import (  # noqa: E402
    attach_sb3_primitive_policy,
    execution_env_factory,
    pooled_eval_fn,
    set_torch_threads,
)
from src.training.run_store import policy_hash  # noqa: E402
from src.types import NodeStatus  # noqa: E402
from eval_h7_preflight import roll  # noqa: E402
from eval_w1a import runs_root  # noqa: E402
from h7_composite import common_args, run  # noqa: E402
from h7_nodes import RUNS, UNLOCKPICKUP, build_unlock_pickup, load_h7_primitive, load_unlock_graph  # noqa: E402
from pickup_box_spec3 import spec3  # noqa: E402
from train_primitives import build_node  # noqa: E402

FREEZE_RECORD = RUNS / "w1b_pickup_box_freeze.json"
CHILD_DIR = REPO / "runs" / "w1b" / "0-4_pickup_box_spec3_s45"
GATE_OUT = RUNS / "w1b_unlock_pickup_gate.json"
FRESH_SEEDS = list(range(54_000, 54_500))
GATE_MIN = 46                       # of 50, on the stopping set
TRAINING_SEEDS = (43, 44, 45)
SMOKE_FRESH_N = 10
# Recorded rows (plan, revision 3): the chain leaves the handoff state in 488/500 (53000-53499),
# and the child succeeds argmax from it in 485/488. Not f29(); h7_preflight.json is not read.
CHAIN_HANDOFF = (488, 500)
CHILD_ARGMAX_HANDOFF = (485, 488)


# --------------------------------------------------------------------------
# children
# --------------------------------------------------------------------------

def load_children(registry, device="cpu"):
    """Canonical unlock_door and drop_held; pickup_box from step 1's ignored checkpoint."""
    record = json.loads(FREEZE_RECORD.read_text(encoding="utf-8"))
    _, unlock_door = load_unlock_graph(registry, device=device)
    drop_held = load_h7_primitive("drop_held", registry, device=device)
    spec, _stats, _, _ = spec3(registry)
    pickup_box = build_node(spec)
    attach_sb3_primitive_policy(pickup_box, UNLOCKPICKUP, spec["hook"], device=device)
    registry.load_into(pickup_box, run_dir=CHILD_DIR)          # require_frozen=True
    sha = policy_hash(pickup_box.policy)
    if pickup_box.spec_hash() != record["spec_hash"]:
        raise SpecMismatch(f"pickup_box spec_hash {pickup_box.spec_hash()} != {record['spec_hash']}")
    if sha != record["policy_sha256"]:
        raise RuntimeError(f"pickup_box weights hash to {sha}, step 1 recorded {record['policy_sha256']}")
    for c in (unlock_door, drop_held, pickup_box):
        if c.status is not NodeStatus.FROZEN:
            raise RuntimeError(f"{c.id} is {c.status.value}, not frozen")
    print(f"  loaded {pickup_box}  spec_hash={pickup_box.spec_hash()}  sha={sha[:16]} "
          f"(0014 Proposed, from {CHILD_DIR.relative_to(REPO).as_posix()})")
    child_ids = {c.id: {"spec_hash": c.spec_hash(), "policy_sha256": policy_hash(c.policy)}
                 for c in (unlock_door, drop_held, pickup_box)}
    return unlock_door, drop_held, pickup_box, child_ids


def chain_ceiling() -> dict:
    (h, n), (k, m) = CHAIN_HANDOFF, CHILD_ARGMAX_HANDOFF
    return {"path": ["{1,1} unlock_door", "{0,9} drop_held", "{0,4} pickup_box (spec 3, seed 45)"],
            "rows": {"chain leaves the handoff state (53000-53499)": f"{h}/{n}",
                     "pickup_box argmax from the handoff (53000-53499)": f"{k}/{m}"},
            "value": round(h / n * k / m, 4),
            "source": "recorded rows (runs/v2/w1a_spec3_sampled.json, seed 45), not f29()"}


# --------------------------------------------------------------------------
# ceiling gate (evaluation only)
# --------------------------------------------------------------------------

def scripted(nodes, seeds) -> dict:
    env = execution_env_factory(UNLOCKPICKUP, NO_HOOK)()
    eps = []
    try:
        for s in seeds:
            obs, _ = env.reset(seed=s)
            u = env.unwrapped
            obs = roll(nodes[0], env, obs)
            obs = roll(nodes[1], env, obs)
            sv = build_state_view(env, 0)
            door = sv.door()
            handoff = bool(door is not None and door.is_open and sv.carrying_type is None)
            obs = roll(nodes[2], env, obs)
            sv = build_state_view(env, 0)
            ok = sv.carrying_type == "box"
            eps.append([s, int(handoff), int(ok), int(u.step_count)])
    finally:
        env.close()
    k = sum(e[2] for e in eps)
    return {"seeds": f"{seeds[0]}-{seeds[-1]}", "episodes": len(eps), "successes": k,
            "handoff_reached": sum(e[1] for e in eps),
            "wilson_lower": round(wilson_lower(k, len(eps)), 4),
            "mean_env_steps_successes": (round(sum(e[3] for e in eps if e[2]) / k, 2) if k else None),
            "per_episode": eps}


def gate(args) -> int:
    if GATE_OUT.exists():
        raise SystemExit(f"{GATE_OUT.relative_to(REPO)} exists; it is never overwritten")
    registry = NodeRegistry(RUNS)
    t0 = time.perf_counter()
    unlock_door, drop_held, pickup_box, child_ids = load_children(registry, args.device)
    t7 = build_unlock_pickup(unlock_door, drop_held, pickup_box)
    print(f"T7 {t7.id} spec_hash={t7.spec_hash()}")
    nodes = (unlock_door, drop_held, pickup_box)
    stop = scripted(nodes, stopping_seeds(50))
    held = scripted(nodes, heldout_seeds(50))
    passed = stop["successes"] >= GATE_MIN
    out = {"plan": "docs/plans/2026-11-15_W1b-1_unlock_pickup_plan.md (revision 3), step 2",
           "decision": "docs/kb/decisions/0014-freeze-without-live-stable.md (Proposed)",
           "t7_spec_hash": t7.spec_hash(), "children": child_ids, "chain_ceiling": chain_ceiling(),
           "rule": f"scripted sequence, argmax; stopping set >= {GATE_MIN}/50 to train, else reading 'chain'",
           "stopping_10000_10049": stop, "heldout_20000_20049": held, "passed": passed,
           "reading": None if passed else "chain",
           "per_episode_columns": ["env_seed", "handoff_reached", "success", "env_steps"],
           "wall_seconds": round(time.perf_counter() - t0, 1)}
    GATE_OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"ceiling gate: stopping {stop['successes']}/50 (handoff {stop['handoff_reached']}/50), "
          f"held-out {held['successes']}/50 (handoff {held['handoff_reached']}/50) -> "
          f"{'PASSED' if passed else 'FAILED: W1b-1 fails with reading chain'}")
    print(f"wrote {GATE_OUT.relative_to(REPO)} ({out['wall_seconds']}s)")
    return 0 if passed else 1


# --------------------------------------------------------------------------
# training + fresh row (steps 2 smoke, 3)
# --------------------------------------------------------------------------

def manifest(run_id: str) -> dict:
    return json.loads((runs_root() / run_id / "run.json").read_text(encoding="utf-8"))


def train(args) -> int:
    smoke = args.smoke
    if not smoke:
        if args.seed not in TRAINING_SEEDS:
            raise SystemExit(f"training seeds are {TRAINING_SEEDS}")
        g = json.loads(GATE_OUT.read_text(encoding="utf-8")) if GATE_OUT.exists() else None
        if not g or not g["passed"]:
            raise SystemExit("the ceiling gate has not passed (run --gate first); nothing is trained")
    name = f"w1b_unlock_pickup_s{args.seed}" + ("_rerun" if args.rerun_of else "")
    out_path = RUNS / f"{name}{'_smoke' if smoke else ''}.json"
    fresh_path = RUNS / f"{name}_fresh{'_smoke' if smoke else ''}.json"
    for p in (out_path, fresh_path):
        if p.exists():
            raise SystemExit(f"{p.relative_to(REPO)} exists; it is never overwritten")
    if args.rerun_of:
        first = manifest(args.rerun_of)          # fail now if the first run is not on this machine

    # Forced by the plan, whatever was passed.
    args.no_freeze, args.skip_strict, args.out_name = True, True, name
    registry = NodeRegistry(RUNS)
    print("loading the frozen children:")
    unlock_door, drop_held, pickup_box, child_ids = load_children(registry, args.device)
    kept = []

    def build():
        n = build_unlock_pickup(unlock_door, drop_held, pickup_box)
        kept.append(n)
        return n

    t0 = time.perf_counter()
    try:
        out = run(args, build, UNLOCKPICKUP, chain_ceiling=chain_ceiling(),
                  default_name=name, expected_depth=3)
        train_wall = time.perf_counter() - t0
        node = kept[0]                           # the node run() trained
        rep = out["composite"]
        sha = policy_hash(node.policy)
        man = manifest(rep["run_id"])
        sha_ok = man["policy_sha256"] == sha

        n_fresh = SMOKE_FRESH_N if smoke else len(FRESH_SEEDS)
        t1 = time.perf_counter()
        fr = pooled_eval_fn(node, [UNLOCKPICKUP], NO_HOOK, n_episodes=n_fresh,
                            seed_base=FRESH_SEEDS[0], env_factory_fn=execution_env_factory)()
        fresh = {"plan": "docs/plans/2026-11-15_W1b-1_unlock_pickup_plan.md (revision 3), step 3",
                 "seed": args.seed, "node_id": node.id, "t7_spec_hash": node.spec_hash(),
                 "composite_run_id": rep["run_id"], "composite_policy_sha256": sha,
                 "policy_sha256_equals_manifest": sha_ok, "children": child_ids,
                 "stable": rep["stable"], "decisions_to_stable": rep["decisions_to_stable"],
                 "fresh_seeds": f"{FRESH_SEEDS[0]}-{FRESH_SEEDS[n_fresh - 1]}", "execution_mode": "argmax",
                 "fresh": fr.as_dict(), "fresh_wall_seconds": round(time.perf_counter() - t1, 1)}
        if args.rerun_of:
            same_hist = (first.get("state") is not None and
                         json.loads((runs_root() / args.rerun_of / "report.json").read_text(encoding="utf-8"))
                         ["criterion"]["history"] == rep["criterion"]["history"])
            same_sha = first["policy_sha256"] == sha
            fresh["rerun_of"] = {"run_id": args.rerun_of, "stopping_history_equal": same_hist,
                                 "policy_sha256_equal": same_sha, "counts": same_hist and same_sha}
        fresh_path.write_text(json.dumps(fresh, indent=1), encoding="utf-8")
        print(f"\nfresh row ({fresh['fresh_seeds']}): {fr}  timeouts {fr.timeouts}")
        print(f"policy_sha256 {sha[:16]} equals run-store manifest: {sha_ok}")
        print(f"wrote {fresh_path.relative_to(REPO)}")
        if smoke:
            d, w = rep["decisions_trained"], rep["wall_seconds"]
            print(f"\nspeed (smoke, {d} decisions, n_eval 10): {rep['decisions_per_second']} decisions/s "
                  f"(training wall incl. stopping evaluations {w}s), env steps per decision "
                  f"{rep['env_steps_per_decision']}; whole run() {train_wall:.1f}s")
            print(f"estimate, one full seed (40,960 decisions, 80 evaluations x 50 episodes): "
                  f"~{40_960 * w / d / 60 * 1.0:.0f} min training at the smoke rate, plus "
                  f"stopping evaluations scaled 10 -> 50 episodes and 500 fresh episodes")
        return 0 if sha_ok else 1
    finally:
        if smoke:
            for p in (out_path, fresh_path):
                if p.exists():
                    p.unlink()
                    print(f"deleted {p.relative_to(REPO)}")


def main() -> int:
    ap = common_args(argparse.ArgumentParser())
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--rerun-of", default=None, metavar="RUN_ID")
    args = ap.parse_args()
    set_torch_threads(args.torch_threads)
    if args.gate:
        return gate(args)
    if not args.smoke and args.seed == 42:
        raise SystemExit("pass --seed 43, 44 or 45")
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
