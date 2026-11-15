"""
W1b-1 step 1 (`docs/plans/2026-11-15_W1b-1_unlock_pickup_plan.md`, revision 3): promote one
`{0,4} pickup_box` spec-3 model to FROZEN under decision 0014 (`Proposed`).

This experiment runs while 0014 is `Proposed`, with Emir's go-ahead, under `scripts/` only.
No file under `src/` changes. The checkpoint goes to the git-ignored `runs/w1b/`; its identity
and the gates go to the tracked `runs/v2/w1b_pickup_box_freeze.json`. Nothing is trained.

    P2   the rule over all ten spec-3 seeds, from both W1a spec-3 result files:
         (a) W1a's test passed and (b) argmax handoff Wilson 95 % lower bound >= 0.95 on
         >= 475 valid episodes; the smallest seed meeting both must be 45
    P1   seed 45's handoff row on 53000-53499 reproduces the record episode for episode,
         argmax and sampled
    --   STABLE set here (0014), freeze(), registry.save(run_dir=runs/w1b/...)
    P3   loaded back with attach_sb3_primitive_policy + registry.load_into(run_dir=...):
         FROZEN, same policy_sha256 and spec_hash, 50 handoff episodes identical to the record

    python scripts/w1b_freeze_pickup_box.py --smoke    10 seeds per gate, temporary dir, nothing kept
    python scripts/w1b_freeze_pickup_box.py            the real run
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.envs.subgoal_env import NO_HOOK  # noqa: E402
from src.skills.executor import DETERMINISTIC, STOCHASTIC  # noqa: E402
from src.skills.registry import NodeRegistry  # noqa: E402
from src.training.evaluation import wilson_lower  # noqa: E402
from src.training.node_trainer import (  # noqa: E402
    attach_sb3_primitive_policy,
    execution_env_factory,
    set_torch_threads,
)
from src.training.run_store import policy_hash  # noqa: E402
from src.types import NodeStatus  # noqa: E402
from eval_h7_preflight import chain  # noqa: E402
from eval_w1a import SAMPLING_SEED, run_mode, runs_root  # noqa: E402
from eval_w1a_spec3 import HANDOFF_SEEDS, SPEC3_HASH, load_spec3_model  # noqa: E402
from h7_nodes import RUNS, UNLOCKPICKUP  # noqa: E402
from pickup_box_spec3 import spec3  # noqa: E402
from train_primitives import build_node  # noqa: E402

SEED = 45
PARENT = "{2,2} unlock_pickup"
RESULT_FILES = {RUNS / "w1a_spec3_sampled.json": (43, 44, 45, 46, 47),
                RUNS / "w1a_spec3_sampled_ext.json": (48, 49, 50, 51, 52)}
LB_MIN, MIN_VALID = 0.95, 475
CKPT_DIR = REPO / "runs" / "w1b" / "0-4_pickup_box_spec3_s45"
OUT = RUNS / "w1b_pickup_box_freeze.json"
SPEC3_TRAINING_RECORD = RUNS / "pickup_box_s3_s45.json"
P3_N = 50
SMOKE_N = 10


def recorded_models() -> dict:
    out = {}
    for path, seeds in RESULT_FILES.items():
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc["spec3_hash"] != SPEC3_HASH:
            raise RuntimeError(f"{path.name}: spec3_hash {doc['spec3_hash']} != {SPEC3_HASH}")
        for s in seeds:
            out[s] = {"file": path.relative_to(REPO).as_posix(), **doc["models"][str(s)]}
    return out


def gate_p2(models: dict) -> dict:
    """The rule over all ten seeds; the smallest seed meeting (a) and (b) must be SEED."""
    rows = {}
    for s in sorted(models):
        m = models[s]
        a_hand = m["test"]["handoff"]["argmax"]
        k, n = a_hand["successes"], a_hand["valid"]
        lb = wilson_lower(k, n)
        if round(lb, 4) != a_hand["wilson_lower"]:
            raise RuntimeError(f"seed {s}: recomputed lower bound {lb:.4f} != recorded {a_hand['wilson_lower']}")
        a = m["test"]["passes"] is True
        b = bool(n >= MIN_VALID and lb >= LB_MIN)
        rows[str(s)] = {"file": m["file"], "a_w1a_test_passed": a,
                        "b_argmax_handoff": f"{k}/{n}", "b_wilson_lower": round(lb, 4),
                        "b_met": b, "meets_rule": a and b}
    meeting = [int(s) for s, r in rows.items() if r["meets_rule"]]
    chosen = min(meeting) if meeting else None
    return {"rule": ("(a) W1a's test passed (test.passes) and (b) argmax handoff Wilson 95 % lower "
                     f"bound >= {LB_MIN} on >= {MIN_VALID} valid episodes; smallest seed meeting both"),
            "seeds": rows, "seeds_meeting_rule": meeting, "chosen": chosen,
            "passed": chosen == SEED}


def recorded_handoff_episodes(models: dict) -> dict:
    eps = models[SEED]["test"]["handoff"]["episodes"]
    return {(e[0], e[1]): tuple(e[2:]) for e in eps}


def as_key(r: dict):
    return (r["seed"], r["sampling_seed"]), (int(r["valid"]), int(r["success"]), r["steps"],
                                             r["repeat_step"], r["repeat_period"])


def compare(node, prep, seeds, rec: dict, *, sampled: bool) -> dict:
    factory = execution_env_factory(UNLOCKPICKUP, NO_HOOK)
    got = run_mode(node, factory, seeds, mode=DETERMINISTIC, prepare=prep)
    if sampled:
        got += run_mode(node, factory, seeds, mode=STOCHASTIC, sampling_seed=SAMPLING_SEED, prepare=prep)
    mine = dict(as_key(r) for r in got)
    diff = [k for k in mine if mine[k] != rec.get(k)]
    def count(ss):
        return {"valid": sum(v[0] for k, v in mine.items() if k[1] == ss),
                "successes": sum(v[0] and v[1] for k, v in mine.items() if k[1] == ss)}
    out = {"seeds": f"{seeds[0]}-{seeds[-1]}", "episodes_compared": len(mine),
           "episodes_differing": len(diff), "argmax_now": count(None),
           "first_differences": [{"key": list(k), "record": rec.get(k), "now": mine[k]} for k in diff[:3]]}
    if sampled:
        out["sampled_now"] = count(SAMPLING_SEED)
    out["passed"] = not diff
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--torch-threads", type=int, default=2)
    args = ap.parse_args()
    t0 = time.perf_counter()
    set_torch_threads(args.torch_threads)
    n1, n3 = (SMOKE_N, SMOKE_N) if args.smoke else (len(HANDOFF_SEEDS), P3_N)

    registry = NodeRegistry(RUNS)
    tmp = tempfile.TemporaryDirectory() if args.smoke else None
    ckpt = Path(tmp.name) / CKPT_DIR.name if args.smoke else CKPT_DIR
    if not args.smoke:
        if OUT.exists():
            raise SystemExit(f"{OUT.relative_to(REPO)} exists; it is never overwritten")

    try:
        # P2 first: it is cheap and decides which model is loaded at all.
        models = recorded_models()
        p2 = gate_p2(models)
        print(f"P2: seeds meeting the rule {p2['seeds_meeting_rule']}, chosen {p2['chosen']} "
              f"-> {'PASSED' if p2['passed'] else 'FAILED'}")
        if not p2["passed"]:
            return 1

        spec, _stats, (t3, dh), _ = spec3(registry)
        node, info = load_spec3_model(SEED, spec)        # spec, manifest, sha, held-out reproduce
        registry.ensure_can_save(node, run_dir=ckpt)      # before any further work
        sha = info["policy_sha256"]
        print(f"loaded seed {SEED}: run {info['run_id']} sha {sha[:16]} spec_hash {node.spec_hash()}")

        prep = chain(t3, dh)
        rec = recorded_handoff_episodes(models)
        p1 = compare(node, prep, HANDOFF_SEEDS[:n1], rec, sampled=True)
        if not args.smoke:
            rec_a = models[SEED]["test"]["handoff"]["argmax"]["successes"]
            rec_s = models[SEED]["test"]["handoff"]["sampled"]["successes"]
            p1["record_successes"] = {"argmax": rec_a, "sampled": rec_s}
            p1["passed"] = p1["passed"] and p1["argmax_now"]["successes"] == rec_a \
                and p1["sampled_now"]["successes"] == rec_s and p1["episodes_compared"] == 1000
        print(f"P1: {p1['episodes_compared']} episodes, {p1['episodes_differing']} differ; argmax "
              f"{p1['argmax_now']['successes']}/{p1['argmax_now']['valid']}, sampled "
              f"{p1['sampled_now']['successes']}/{p1['sampled_now']['valid']} "
              f"-> {'PASSED' if p1['passed'] else 'FAILED'}")
        if not p1["passed"]:
            print(json.dumps(p1["first_differences"], indent=1))
            return 1

        run_dir = runs_root() / info["run_id"]
        steps_trained = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))["steps_trained"]
        chain_steps = json.loads(SPEC3_TRAINING_RECORD.read_text(encoding="utf-8"))[
            "chain_env_steps_in_resets"]["during_training"]["training_envs"]["chain_steps"]
        test = models[SEED]["test"]
        table = {
            "a_w1a_test_sampled": {"own_hook": f"{test['own_hook']['sampled']['successes']}/"
                                               f"{test['own_hook']['sampled']['valid']}",
                                   "handoff": f"{test['handoff']['sampled']['successes']}/"
                                              f"{test['handoff']['sampled']['valid']}",
                                   "passed": test["passes"]},
            "b_argmax_handoff": {"successes": f"{test['handoff']['argmax']['successes']}/"
                                              f"{test['handoff']['argmax']['valid']}",
                                 "wilson_lower": test["handoff"]["argmax"]["wilson_lower"]},
            "argmax_own_hook_known_limit": {
                "successes": f"{test['own_hook']['argmax']['successes']}/{test['own_hook']['argmax']['valid']}",
                "failures_with_state_repeat": test["own_hook"]["argmax"]["failures_with_state_repeat"],
                "failures": test["own_hook"]["argmax"]["failures"]},
            "decision_0004_during_training": "not met; budget exhausted (v2 §9: ABANDONED)",
            "ranges": {"own_hook": test["own_hook"]["seeds_range"], "handoff": test["handoff"]["seeds_range"]},
        }
        meta = {"source_run_id": info["run_id"], "source_policy_sha256": sha,
                "steps_trained": steps_trained, "chain_env_steps_in_training_resets": chain_steps,
                "met_0004": False, "frozen_under": "0014 (Proposed)", "execution_mode": "argmax",
                "accepted_for_parent": PARENT, "acceptance": table,
                "plan": "docs/plans/2026-11-15_W1b-1_unlock_pickup_plan.md (revision 3), step 1"}

        node.status = NodeStatus.STABLE   # decision 0014 (Proposed): frozen without live STABLE
        node.freeze()
        registry.save(node, extra_meta=meta, run_dir=ckpt)
        print(f"FROZEN and saved to {ckpt if args.smoke else ckpt.relative_to(REPO)}")

        back = build_node(spec)
        attach_sb3_primitive_policy(back, UNLOCKPICKUP, spec["hook"])
        registry.load_into(back, run_dir=ckpt)
        sha_back = policy_hash(back.policy)
        p3 = compare(back, prep, HANDOFF_SEEDS[:n3], rec, sampled=False)
        p3.update({"status": back.status.value, "policy_sha256_equal": sha_back == sha,
                   "spec_hash_equal": back.spec_hash() == node.spec_hash() == SPEC3_HASH})
        p3["passed"] = bool(p3["passed"] and back.status is NodeStatus.FROZEN
                            and p3["policy_sha256_equal"] and p3["spec_hash_equal"])
        print(f"P3: status {p3['status']}, sha equal {p3['policy_sha256_equal']}, spec_hash equal "
              f"{p3['spec_hash_equal']}, {p3['episodes_compared']} argmax episodes, "
              f"{p3['episodes_differing']} differ -> {'PASSED' if p3['passed'] else 'FAILED'}")
        if not p3["passed"]:
            print(f"checkpoint left at {ckpt}; tracked file not written")
            return 1

        out = {"plan": meta["plan"], "decision": "docs/kb/decisions/0014-freeze-without-live-stable.md (Proposed)",
               "node_id": node.id, "spec_hash": node.spec_hash(), "seed": SEED,
               "source_run_id": info["run_id"], "policy_sha256": sha,
               "checkpoint": CKPT_DIR.relative_to(REPO).as_posix() + " (git-ignored)",
               "meta": meta, "load_check": info["load_check_heldout_mixed_hook"],
               "gate_p2_rule": p2, "gate_p1_reproduces_record": p1, "gate_p3_load_back": p3,
               "note": "nothing trained; runs/v2/0-4_pickup_box/ (spec 2) untouched",
               "wall_seconds": round(time.perf_counter() - t0, 1)}
        if args.smoke:
            print(f"smoke run, {out['wall_seconds']}s; nothing kept")
            return 0
        OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"wrote {OUT.relative_to(REPO)} ({out['wall_seconds']}s)")
        return 0
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
