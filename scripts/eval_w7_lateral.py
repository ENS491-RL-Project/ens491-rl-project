"""
W7 (C-2) final score: each final model once on 500 fresh env seeds 60000-60499.

Plan: docs/plans/2026-11-15_W7_pn_lateral_plan.md, Design ("Final score") and step 4.
Argmax, DoorKey-6x6 under go_to_goal's own hook, through the Executor (v2 §5.3).
Reported, not a criterion.

Before a model is scored:
- its lateral source is rebuilt (arm B: the frozen `{0,2} open_door` from the registry;
  arm C: re-drawn from its torch seed) and its hash must equal the one recorded in the
  run's result JSON and in `run.json`;
- the loaded weights' hash must equal `run.json`'s `policy_sha256`.
A mismatch is an error. The state dict is then copied to
`runs/v2/w7/models/<arm>_s<seed>.pt`, because `runs/training/` is not tracked.

    python scripts/eval_w7_lateral.py --arm B        # writes runs/v2/w7/scores/B.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import torch  # noqa: E402

from train_w7_lateral import (  # noqa: E402
    ARMS,
    DOORKEY,
    OUT,
    TRAINING_SEEDS,
    make_holder,
    node_trainer_build_node,
    specs,
)
from src.training.evaluation import evaluate_node  # noqa: E402
from src.training.node_trainer import (  # noqa: E402
    attach_sb3_primitive_policy,
    set_torch_threads,
    subgoal_env_factory,
)
from src.training.run_store import _json_value, policy_hash  # noqa: E402

SCORE_SEEDS = list(range(60_000, 60_500))


def score_one(arm: str, seed: int) -> dict:
    result = json.loads((OUT / "runs" / f"{arm}_s{seed}.json").read_text(encoding="utf-8"))
    run_dir = Path(result["report"]["run_dir"])
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if manifest["state"] not in ("stable", "budget_exhausted") or not (run_dir / "complete.json").exists():
        raise RuntimeError(f"{arm} s{seed}: run {run_dir.name} is not a complete archive")

    spec = specs()["go_to_goal"]
    holder, lateral_sources = make_holder(arm, seed)
    source_check = None
    if holder is not None:
        recorded = result["source"]["policy_sha256"]
        in_manifest = manifest["config"]["lateral_source"]["policy_sha256"]
        if not (holder.policy_sha256 == recorded == in_manifest):
            raise RuntimeError(f"{arm} s{seed}: lateral source hash {holder.policy_sha256} "
                               f"!= recorded {recorded} / run.json {in_manifest}")
        source_check = {"policy_sha256": holder.policy_sha256, "matches_recorded": True,
                        "rebuilt_from": "registry {0,2} open_door" if arm == "B"
                        else f"torch seed {holder.torch_seed}"}

    node = node_trainer_build_node(spec, node_id=f"{spec['node_id']} w7-{arm}",
                                   lateral_sources=lateral_sources)
    if _json_value(node.spec()) != manifest["node_spec"]:
        raise RuntimeError(f"{arm} s{seed}: node spec differs from the run's run.json")
    attach_sb3_primitive_policy(node, DOORKEY, spec["hook"], lateral_source=holder)
    sd_path = run_dir / "policy_state_dict.pt"
    node.policy.load_state_dict(torch.load(sd_path, map_location="cpu", weights_only=True))
    if policy_hash(node.policy) != manifest["policy_sha256"]:
        raise RuntimeError(f"{arm} s{seed}: loaded weights differ from run.json's policy_sha256")

    models = OUT / "models"
    models.mkdir(parents=True, exist_ok=True)
    copy = models / f"{arm}_s{seed}.pt"
    shutil.copyfile(sd_path, copy)

    t0 = time.perf_counter()
    res = evaluate_node(node, subgoal_env_factory(node, DOORKEY, spec["hook"], monitor=False),
                        seeds=SCORE_SEEDS)
    wall = time.perf_counter() - t0
    if holder is not None and holder.current_hash() != holder.policy_sha256:
        raise RuntimeError(f"{arm} s{seed}: lateral source changed during scoring")
    print(f"  {arm} s{seed}: {res}  ({wall:.0f}s)", flush=True)
    return {
        "arm": arm,
        "seed": seed,
        "run_id": manifest["run_id"],
        "stable": result["stable"],
        "metric_steps": result["metric_steps"],
        "policy_sha256": manifest["policy_sha256"],
        "model_file": copy.relative_to(REPO).as_posix(),
        "model_file_sha256": hashlib.sha256(copy.read_bytes()).hexdigest(),
        "source_check": source_check,
        "score": {k: v for k, v in res.as_dict().items() if k != "per_episode"},
        "failed_env_seeds": [s for s, ok in zip(SCORE_SEEDS, res.per_episode) if not ok],
        "wall_seconds": round(wall, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    set_torch_threads(args.threads)

    rows = [score_one(args.arm, s) for s in TRAINING_SEEDS]
    out = OUT / "scores" / f"{args.arm}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment": "W7 (C-2) final score",
        "env": DOORKEY,
        "hook": specs()["go_to_goal"]["hook"].spec,
        "env_seeds": f"{SCORE_SEEDS[0]}-{SCORE_SEEDS[-1]}",
        "execution_mode": "deterministic (argmax)",
        "arm": args.arm,
        "models": rows,
    }, indent=2), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
