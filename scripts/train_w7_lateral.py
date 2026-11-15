"""
W7 (C-2): does a lateral from a frozen primitive make a new primitive learn faster?

Plan: docs/plans/2026-11-15_W7_pn_lateral_plan.md (decision 0013).

Target `{0,3} go_to_goal` (DoorKey-6x6, its own spec and hook from
`train_primitives.specs()`), trained by the unchanged `train_primitive` in one of
three arms:

    A  no lateral
    B  lateral from the frozen `{0,2} open_door`
    C  lateral from an untrained network of the same shape, drawn from torch seed
       900000 + training seed. Not a node, never registered, never frozen through
       `freeze()`; an experimental control only (decision 0013).

Nothing is frozen or saved to the registry. Each run writes one JSON under
`runs/v2/w7/` and is archived by the run store under `runs/training/`.

Usage:
    python scripts/train_w7_lateral.py --arm B --seed 1000            # one real run
    python scripts/train_w7_lateral.py --arm C --seed 1000 --smoke     # short check run
    python scripts/train_w7_lateral.py --arm A --seed 42 --check-h4    # H-4 reproduction check
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

import torch  # noqa: E402

import src.training.node_trainer as node_trainer  # noqa: E402
from train_primitives import DOORKEY, UNLOCK, specs  # noqa: E402
from src.skills.node import ALWAYS, SkillNode  # noqa: E402
from src.skills.registry import NodeRegistry  # noqa: E402
from src.training.device import describe  # noqa: E402
from src.training.node_trainer import (  # noqa: E402
    PPO_DEFAULTS,
    attach_sb3_primitive_policy,
    load_frozen_primitive,
    set_torch_threads,
    train_primitive,
)
from src.training.run_store import policy_hash  # noqa: E402
from src.types import NodeStatus  # noqa: E402

RUNS_V2 = REPO / "runs" / "v2"
OUT = RUNS_V2 / "w7"

ARMS = ("A", "B", "C")
TRAINING_SEEDS = tuple(range(1000, 2000, 100))     # 1000, 1100, ..., 1900
BUDGET = 401_408                                    # 196 rollouts of 2,048 = 98 x 4,096
EVAL_EVERY = 4_096                                  # two policy updates
N_ENVS = 8
ARM_C_TORCH_SEED_BASE = 900_000

SMOKE_BUDGET = 16_384                               # 8 rollouts, 4 evaluations
H4_CHECK = dict(seed=42, eval_every=20_000, budget=800_000)   # as H-4 ran it


# --------------------------------------------------------------------------
# Lateral source holder
# --------------------------------------------------------------------------

class LateralSourceHolder:
    """
    What `LateralMlpExtractor` reads as its source: anything with `.policy.mlp_extractor`.

    A plain object on purpose, not an `nn.Module`. `ColumnPolicy` stores the source
    before `nn.Module.__init__` runs, so a module holder fails at construction; and a
    registered module would be re-initialised by SB3's orthogonal init and handed to
    the optimizer. As a plain attribute the source stays outside the target's
    `state_dict()`, `parameters()` and optimizer.

    `as_dict()` is what the run store writes into `run.json`'s config, so the run names
    its source. The hash is taken here, before the PPO object exists.
    """

    def __init__(self, sb3_policy, *, kind: str, node_id: str | None = None,
                 spec_hash: str | None = None, torch_seed: int | None = None) -> None:
        self.policy = sb3_policy
        self.kind = kind
        self.node_id = node_id
        self.spec_hash = spec_hash
        self.torch_seed = torch_seed
        self.policy_sha256 = policy_hash(sb3_policy)

    def current_hash(self) -> str:
        return policy_hash(self.policy)

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "node_id": self.node_id,
            "spec_hash": self.spec_hash,
            "torch_seed": self.torch_seed,
            "policy_sha256": self.policy_sha256,
        }


def load_open_door() -> SkillNode:
    """The frozen `{0,2} open_door` from the registry, spec hash verified."""
    spec = specs()["open_door"]
    node = node_trainer_build_node(spec)
    load_frozen_primitive(node, NodeRegistry(RUNS_V2), UNLOCK, spec["hook"])
    if node.status is not NodeStatus.FROZEN:
        raise RuntimeError(f"{node.id} did not load as FROZEN")
    return node


def untrained_source(seed: int):
    """Arm C's source: a fresh, never-trained go_to_goal-shaped policy (plan, Design)."""
    torch_seed = ARM_C_TORCH_SEED_BASE + seed
    spec = specs()["go_to_goal"]
    throwaway = node_trainer_build_node(spec, node_id="{0,3} go_to_goal w7-C-source")
    torch.manual_seed(torch_seed)
    attach_sb3_primitive_policy(throwaway, DOORKEY, spec["hook"])
    sb3_policy = throwaway.policy.sb3_policy
    for p in sb3_policy.parameters():
        p.requires_grad_(False)
    return sb3_policy, torch_seed


def make_holder(arm: str, seed: int):
    """(holder or None, lateral_sources for the node)."""
    if arm == "A":
        return None, ()
    if arm == "B":
        src = load_open_door()
        holder = LateralSourceHolder(src.policy.sb3_policy, kind="frozen_node",
                                     node_id=src.id, spec_hash=src.spec_hash())
        return holder, (src,)
    if arm == "C":
        sb3_policy, torch_seed = untrained_source(seed)
        holder = LateralSourceHolder(sb3_policy, kind="untrained_network",
                                     torch_seed=torch_seed)
        return holder, ()
    raise ValueError(f"unknown arm {arm!r}")


def node_trainer_build_node(spec, *, node_id=None, lateral_sources=()) -> SkillNode:
    """`train_primitives.build_node`, with an id and lateral sources (it takes neither)."""
    return SkillNode(
        node_id or spec["node_id"],
        goal=spec["goal"],
        initiation=spec.get("initiation", ALWAYS),
        timeout=spec["timeout"],
        env_ids=spec["env_ids"],
        normalize_obs=True,
        train_hook=spec["hook"].spec,
        lateral_sources=lateral_sources,
        m=spec["m"],
    )


# --------------------------------------------------------------------------
# Snapshot right after the PPO object is built
# --------------------------------------------------------------------------

def _lateral_state(policy) -> dict:
    ex = policy.mlp_extractor
    if getattr(ex, "lateral_source_column", None) is None:
        return {}
    return {name: getattr(ex, name).weight.detach().clone() for name in ("lat1", "lat2")}


def _hash_tensors(tensors: dict) -> str | None:
    if not tensors:
        return None
    import hashlib
    h = hashlib.sha256()
    for name, t in sorted(tensors.items()):
        h.update(name.encode())
        h.update(t.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class _Snapshot:
    """
    Records the target's lateral weights and the source's hash between PPO
    construction and `learn()`, by wrapping `track_model`, which `train_primitive`
    calls right after building the model. A script-side wrapper; nothing under
    `src/` changes and the training path is untouched.
    """

    def __init__(self, holder: LateralSourceHolder | None) -> None:
        self.holder = holder
        self.lateral_init: dict = {}
        self.source_hash_after_build: str | None = None
        self.target_param_count: int | None = None
        self.source_in_target_state_dict: bool | None = None
        self.source_params_in_optimizer: int | None = None
        self._orig = node_trainer.track_model

    def __enter__(self):
        def wrapped(model):
            pol = model.policy
            self.lateral_init = _lateral_state(pol)
            self.target_param_count = sum(p.numel() for p in pol.parameters())
            if self.holder is not None:
                self.source_hash_after_build = self.holder.current_hash()
                src_ids = {id(p) for p in self.holder.policy.parameters()}
                opt_ids = {id(p) for g in pol.optimizer.param_groups for p in g["params"]}
                self.source_params_in_optimizer = len(src_ids & opt_ids)
                self.source_in_target_state_dict = any(
                    "lateral_source" in k for k in pol.state_dict())
            return self._orig(model)

        node_trainer.track_model = wrapped
        return self

    def __exit__(self, *exc):
        node_trainer.track_model = self._orig
        return False


# --------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------

def run_one(arm: str, seed: int, *, budget: int, eval_every: int, out_path: Path,
            label: str, threads: int, verbose: int) -> dict:
    spec = specs()["go_to_goal"]
    holder, lateral_sources = make_holder(arm, seed)
    node = node_trainer_build_node(spec, node_id=f"{spec['node_id']} w7-{arm}",
                                   lateral_sources=lateral_sources)
    source_hash_before = holder.policy_sha256 if holder is not None else None

    print(f"\n{'=' * 72}\nW7 arm {arm}  seed {seed}  {label}  node {node.id!r}\n"
          f"budget {budget:,}  eval_every {eval_every:,}  n_envs {N_ENVS}  threads {threads}\n"
          f"source {holder.as_dict() if holder else None}\n{'=' * 72}", flush=True)

    t0 = time.perf_counter()
    with _Snapshot(holder) as snap:
        report, model = train_primitive(
            node, spec["env_ids"], spec["hook"],
            total_timesteps=budget, eval_every=eval_every, n_envs=N_ENVS,
            seed=seed, device="cpu", lateral_source=holder, verbose=verbose,
        )
    wall_total = time.perf_counter() - t0

    lateral_final = _lateral_state(model.policy)
    lateral = None
    if lateral_final:
        lateral = {
            "init_sha256": _hash_tensors(snap.lateral_init),
            "final_sha256": _hash_tensors(lateral_final),
            "changed": {k: bool(not torch.equal(snap.lateral_init[k], lateral_final[k]))
                        for k in lateral_final},
            "max_abs_change": {k: float((lateral_final[k] - snap.lateral_init[k]).abs().max())
                               for k in lateral_final},
            "weights": int(sum(t.numel() for t in lateral_final.values())),
        }
    source = None
    if holder is not None:
        after = holder.current_hash()
        source = {
            **holder.as_dict(),
            "sha256_before_ppo": source_hash_before,
            "sha256_after_ppo_build": snap.source_hash_after_build,
            "sha256_after_training": after,
            "unchanged": source_hash_before == snap.source_hash_after_build == after,
            "params_in_target_optimizer": snap.source_params_in_optimizer,
            "in_target_state_dict": snap.source_in_target_state_dict,
        }

    history = report["criterion"]["history"]
    result = {
        "experiment": "W7 (C-2)",
        "plan": "docs/plans/2026-11-15_W7_pn_lateral_plan.md",
        "label": label,
        "arm": arm,
        "seed": seed,
        "budget": budget,
        "eval_every": eval_every,
        "threads": threads,
        "stable": report["stable"],
        "steps_trained": report["steps_trained"],
        "steps_to_stable": report["steps_trained"] if report["stable"] else None,
        "metric_steps": report["steps_trained"] if report["stable"] else budget,
        "censored": not report["stable"],
        "n_evaluations": len(history),
        "success_series": [[eval_every * (i + 1), h["success_rate"]] for i, h in enumerate(history)],
        "target_param_count": snap.target_param_count,
        "lateral": lateral,
        "source": source,
        "wall_seconds_total": round(wall_total, 1),
        "report": report,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  stable={result['stable']} steps={result['steps_trained']:,} "
          f"evals={result['n_evaluations']} wall={report['wall_seconds']}s "
          f"({report['steps_per_second']} steps/s)\n  wrote {out_path.relative_to(REPO)}",
          flush=True)
    if source is not None and not source["unchanged"]:
        raise RuntimeError(f"lateral source changed during the run: {source}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--seed", type=int, required=True)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true",
                      help=f"budget {SMOKE_BUDGET:,}, same eval_every; output under w7/smoke/")
    mode.add_argument("--check-h4", action="store_true",
                      help="arm A, seed 42, eval_every 20,000, budget 800,000 (H-4's settings)")
    ap.add_argument("--threads", type=int, default=1,
                    help="torch intra-op threads; 0 leaves torch's default")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", type=int, default=1)
    args = ap.parse_args()

    if args.smoke:
        budget, eval_every, label = SMOKE_BUDGET, EVAL_EVERY, "smoke"
        out = OUT / "smoke" / f"{args.arm}_s{args.seed}.json"
    elif args.check_h4:
        if args.arm != "A" or args.seed != H4_CHECK["seed"]:
            ap.error("--check-h4 is arm A at seed 42")
        budget, eval_every, label = H4_CHECK["budget"], H4_CHECK["eval_every"], "check_h4"
        out = OUT / "checks" / f"A_s42_h4_check_threads{args.threads}.json"
    else:
        if args.seed not in TRAINING_SEEDS:
            ap.error(f"training seeds are fixed by the plan: {TRAINING_SEEDS}")
        budget, eval_every, label = BUDGET, EVAL_EVERY, "main"
        out = OUT / "runs" / f"{args.arm}_s{args.seed}.json"
    if out.exists() and not args.overwrite:
        print(f"{out.relative_to(REPO)} exists; refusing to overwrite (pass --overwrite)")
        return 2

    threads = set_torch_threads(args.threads) if args.threads > 0 else torch.get_num_threads()
    print(describe())
    print(f"PPO {PPO_DEFAULTS}")
    run_one(args.arm, args.seed, budget=budget, eval_every=eval_every, out_path=out,
            label=label, threads=threads, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
