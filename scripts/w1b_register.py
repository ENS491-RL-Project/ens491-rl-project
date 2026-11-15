"""
Register W1b-1's nodes in the canonical registry after decision 0014 was accepted (2026-11-15).

    {0,4} pickup_box      spec 3 (0011), training seed 45, frozen under 0014 in W1b-1 step 1;
                          copied from the git-ignored runs/w1b/ checkpoint, weights unchanged
    {2,2} unlock_pickup   training seed 43 (the smallest passing seed; all three act identically),
                          STABLE under 0004 in W1b-1 step 3; weights from its run-store archive

Before this script runs, spec 2's seed-42 checkpoint (TRAINING) is moved out of the registry to
`runs/v2/attempts/0-4_pickup_box_spec2_s42/` (`h7_nodes.SPEC2_S42_DIR`). Nothing is trained and no
existing checkpoint is overwritten. Checks, all of which must pass before anything is written to
`runs/v2/w1b_register.json`:

    child     loads FROZEN from runs/w1b/, policy_sha256 and spec_hash equal step 1's record
    parent    run-store weights hash to the recorded policy_sha256 (manifest and fresh file);
              held-out 50 reproduces the recorded 49/50 and mean length
    load back h7_nodes.load_unlock_pickup loads the whole graph FROZEN from the registry; T7's
              spec_hash is W1b-1's; held-out 50 reproduces again

    python scripts/w1b_register.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import torch  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from src.envs.subgoal_env import NO_HOOK  # noqa: E402
from src.skills.registry import NodeRegistry  # noqa: E402
from src.training.evaluation import heldout_seeds  # noqa: E402
from src.training.node_trainer import (  # noqa: E402
    CompositePolicyAdapter,
    execution_env_factory,
    option_env_factory,
    pooled_eval_fn,
    set_torch_threads,
)
from src.training.run_store import policy_hash  # noqa: E402
from src.types import NodeStatus  # noqa: E402
from eval_w1a import runs_root  # noqa: E402
from h7_nodes import RUNS, UNLOCKPICKUP, build_unlock_pickup, load_unlock_pickup  # noqa: E402
from w1b_unlock_pickup import CHILD_DIR, FREEZE_RECORD, load_children  # noqa: E402

DECISION = REPO / "docs" / "kb" / "decisions" / "0014-freeze-without-live-stable.md"
PARENT_SEED = 43
PARENT_RECORD = RUNS / f"w1b_unlock_pickup_s{PARENT_SEED}.json"
PARENT_FRESH = RUNS / f"w1b_unlock_pickup_s{PARENT_SEED}_fresh.json"
OUT = RUNS / "w1b_register.json"
T7_SPEC_HASH = "3012137ea5300693"


def heldout(node):
    return pooled_eval_fn(node, [UNLOCKPICKUP], NO_HOOK, n_episodes=50,
                          seed_base=heldout_seeds(1)[0], env_factory_fn=execution_env_factory)()


def main() -> int:
    t0 = time.perf_counter()
    set_torch_threads(2)
    if not re.search(r"^Status: Accepted", DECISION.read_text(encoding="utf-8"), re.M):
        raise SystemExit("0014 is not Accepted; nothing is registered")
    if OUT.exists():
        raise SystemExit(f"{OUT.relative_to(REPO)} exists; it is never overwritten")
    registry = NodeRegistry(RUNS)
    freeze_rec = json.loads(FREEZE_RECORD.read_text(encoding="utf-8"))
    rec = json.loads(PARENT_RECORD.read_text(encoding="utf-8"))["composite"]
    fresh = json.loads(PARENT_FRESH.read_text(encoding="utf-8"))
    out = {"decisions": ["0011 (Accepted)", "0014 (Accepted 2026-11-15)"]}

    # -- child ------------------------------------------------------------
    unlock_door, drop_held, pickup_box, child_ids = load_children(registry)   # checks sha, FROZEN
    registry.ensure_can_save(pickup_box)
    registration = {"from": CHILD_DIR.relative_to(REPO).as_posix(), "date": "2026-11-15",
                    "decision": "0014 Accepted 2026-11-15 (frozen while Proposed, W1b-1 step 1)",
                    "record": FREEZE_RECORD.relative_to(REPO).as_posix()}
    d = registry.save(pickup_box, extra_meta={"registration": registration})
    out["pickup_box"] = {"dir": d.relative_to(REPO).as_posix(), "spec_hash": pickup_box.spec_hash(),
                         "policy_sha256": policy_hash(pickup_box.policy),
                         "equals_step1_record": policy_hash(pickup_box.policy) == freeze_rec["policy_sha256"]}
    print(f"registered {pickup_box.id} -> {out['pickup_box']['dir']}")

    # -- parent -----------------------------------------------------------
    t7 = build_unlock_pickup(unlock_door, drop_held, pickup_box)
    if t7.spec_hash() != T7_SPEC_HASH:
        raise RuntimeError(f"T7 spec_hash {t7.spec_hash()} != {T7_SPEC_HASH}")
    registry.ensure_can_save(t7)
    from sb3_contrib import MaskablePPO
    vec = DummyVecEnv([option_env_factory(t7, UNLOCKPICKUP, NO_HOOK, monitor=False)])
    try:
        t7.policy = CompositePolicyAdapter(MaskablePPO("MlpPolicy", vec, device="cpu", verbose=0).policy)
    finally:
        vec.close()
    run_dir = runs_root() / rec["run_id"]
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    t7.policy.load_state_dict(torch.load(run_dir / "policy_state_dict.pt", map_location="cpu",
                                         weights_only=True))
    sha = policy_hash(t7.policy)
    if not (sha == manifest["policy_sha256"] == fresh["composite_policy_sha256"]):
        raise RuntimeError(f"T7 weights hash to {sha}; manifest {manifest['policy_sha256']}, "
                           f"fresh file {fresh['composite_policy_sha256']}")
    if not rec["stable"]:
        raise RuntimeError("the seed-43 run is not STABLE")
    h = heldout(t7)
    want = rec["heldout"]
    if (h.successes, round(h.mean_length, 2)) != (want["successes"], want["mean_length"]):
        raise RuntimeError(f"held-out {h.successes}, {h.mean_length} != recorded "
                           f"{want['successes']}, {want['mean_length']}")
    t7.status = NodeStatus.STABLE          # met 0004 in training (W1b-1 step 3, criterion in meta)
    t7.freeze()
    meta = {k: v for k, v in rec.items() if k not in ("run_dir",)}
    meta["fresh_54000_54499"] = fresh["fresh"]
    meta["registration"] = {"from": f"run store {rec['run_id']}", "date": "2026-11-15",
                            "seed": PARENT_SEED,
                            "why_this_seed": "smallest passing training seed (plan); seeds 43-45 act identically",
                            "record": PARENT_RECORD.relative_to(REPO).as_posix()}
    d = registry.save(t7, extra_meta=meta)
    out["unlock_pickup"] = {"dir": d.relative_to(REPO).as_posix(), "spec_hash": t7.spec_hash(),
                            "run_id": rec["run_id"], "policy_sha256": sha,
                            "heldout_reproduced": h.as_dict()}
    print(f"registered {t7.id} -> {out['unlock_pickup']['dir']}  held-out {h}")

    # -- load back --------------------------------------------------------
    _, _, _, pb2, t7b = load_unlock_pickup(NodeRegistry(RUNS), verbose=False)
    h2 = heldout(t7b)
    out["load_back"] = {
        "all_frozen": all(n.status is NodeStatus.FROZEN for n in (pb2, t7b, *t7b.children)),
        "t7_spec_hash_equal": t7b.spec_hash() == T7_SPEC_HASH,
        "pickup_box_sha_equal": policy_hash(pb2.policy) == freeze_rec["policy_sha256"],
        "t7_sha_equal": policy_hash(t7b.policy) == sha,
        "heldout": h2.as_dict(),
        "heldout_equal": (h2.successes, round(h2.mean_length, 2)) == (want["successes"], want["mean_length"])}
    ok = all(v for k, v in out["load_back"].items() if k != "heldout") and out["pickup_box"]["equals_step1_record"]
    out["passed"] = ok
    out["wall_seconds"] = round(time.perf_counter() - t0, 1)
    print(f"load back: {json.dumps({k: v for k, v in out['load_back'].items() if k != 'heldout'})} -> "
          f"{'PASSED' if ok else 'FAILED'}")
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)} ({out['wall_seconds']}s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
