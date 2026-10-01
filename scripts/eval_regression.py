"""
H-6.7: re-evaluate every FROZEN node and check it reproduces its recorded score.

[A: F26] This is a **bug guard, not evidence.** Freezing clears `requires_grad`
and nothing ever writes to a frozen node's weights, so retention is true by
construction and a "100% retention" result says nothing about continual
learning. What it does catch is the failure mode that would quietly invalidate
every downstream number: a checkpoint that loads into the wrong network shape, a
`spec_hash` that matches while the reset hook or observation pipeline has
drifted, or a registry that silently skips the state dict.

So the assertion is not "the score is still good" but "the score is still
*identical*". Any difference at all is a bug in the loading path.

Run after every task that trains or freezes anything.

Usage:  python scripts/eval_regression.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.envs.subgoal_env import NO_HOOK, two_keys  # noqa: E402
from src.skills.registry import NodeRegistry  # noqa: E402
from src.training.evaluation import heldout_seeds  # noqa: E402
from src.training.node_trainer import (  # noqa: E402
    execution_env_factory,
    load_frozen_composite,
    load_frozen_primitive,
    pooled_eval_fn,
)
from src.types import NodeStatus  # noqa: E402
from train_choice_task import COLORS, build_t11, choice_specs  # noqa: E402
from train_primitives import DOORKEY, UNLOCK, build_node, specs  # noqa: E402
from train_solve_doorkey import build_t5  # noqa: E402
from train_unlock_door import build_t3  # noqa: E402

RUNS = REPO / "runs" / "v2"
N_EVAL = 50


def primitive(registry, name, spec):
    node = build_node(spec)
    load_frozen_primitive(node, registry, spec["env_ids"][0], spec["hook"])
    return node, spec["env_ids"], spec["hook"]


def main() -> int:
    registry = NodeRegistry(RUNS)
    rows = []
    nodes = {}

    # ---- primitives ---------------------------------------------------
    all_specs = dict(specs())
    all_specs.update(choice_specs())
    for name, spec in all_specs.items():
        try:
            node, env_ids, hook = primitive(registry, name, spec)
        except Exception as exc:
            print(f"SKIP {name}: {type(exc).__name__}: {exc}")
            continue
        nodes[name] = node
        res = pooled_eval_fn(node, env_ids, hook, n_episodes=N_EVAL,
                             seed_base=heldout_seeds(1)[0])()
        rows.append((node, res, node.meta.get("heldout", {}).get("success_rate")))

    # ---- composites, children first ------------------------------------
    composites = []
    if {"get_key", "open_door"} <= set(nodes):
        t3 = build_t3([nodes["get_key"], nodes["open_door"]])
        composites.append((t3, [UNLOCK], NO_HOOK))
        if "go_to_goal" in nodes:
            composites.append((build_t5(t3, nodes["go_to_goal"]), [DOORKEY], NO_HOOK))
    if {"get_key_yellow", "get_key_blue", "open_door"} <= set(nodes):
        composites.append((
            build_t11([nodes["get_key_yellow"], nodes["get_key_blue"],
                       nodes["open_door"]]),
            [DOORKEY], two_keys(COLORS),
        ))

    for node, env_ids, hook in composites:
        try:
            load_frozen_composite(node, registry, env_ids[0], hook)
        except Exception as exc:
            print(f"SKIP {node.id}: {type(exc).__name__}: {exc}")
            continue
        res = pooled_eval_fn(node, env_ids, hook, n_episodes=N_EVAL,
                             seed_base=heldout_seeds(1)[0],
                             env_factory_fn=execution_env_factory)()
        rows.append((node, res, node.meta.get("heldout", {}).get("success_rate")))

    # ---- report --------------------------------------------------------
    print(f"\n{'node':<24}{'n':>3}{'recorded':>10}{'now':>8}{'delta':>8}   status")
    print("-" * 64)
    out, drift = [], []
    for node, res, recorded in rows:
        now = round(res.success_rate, 4)
        delta = None if recorded is None else round(now - recorded, 4)
        ok = recorded is None or abs(now - recorded) < 1e-9
        status = "no record" if recorded is None else ("ok" if ok else "DRIFT")
        print(f"{node.id:<24}{node.n:>3}"
              f"{'-' if recorded is None else f'{recorded:>10.3f}'}"
              f"{now:>8.3f}{'-' if delta is None else f'{delta:>8.3f}'}   {status}")
        out.append({"node": node.id, "n": node.n, "status": node.status.value,
                    "spec_hash": node.spec_hash(), "recorded": recorded,
                    "now": now, "delta": delta, **res.as_dict()})
        if not ok:
            drift.append((node.id, recorded, now))

    path = RUNS / "regression.json"
    path.write_text(json.dumps({"n_eval": N_EVAL, "rows": out}, indent=2),
                    encoding="utf-8")
    print(f"\nwrote {path.relative_to(REPO)}")

    if drift:
        print("\nREGRESSION FAILED -- a frozen node's score changed. Frozen weights "
              "cannot change, so this is a bug in the loading path, not drift:")
        for node_id, was, now in drift:
            print(f"  {node_id}: recorded {was:.3f}, now {now:.3f}")
        return 1
    print(f"\nREGRESSION PASSED: {len(rows)} frozen nodes reproduce their recorded "
          "scores exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
