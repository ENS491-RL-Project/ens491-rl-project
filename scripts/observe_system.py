"""
System observation: the registered roots on fresh seeds, with a label on every failure.

Plan (rows, seeds, labels): docs/plans/2026-11-15_system_observation_plan.md.
Evaluation only. Output: runs/v2/system_observation.json.

    python scripts/observe_system.py --smoke     10 seeds per row, nothing written
    python scripts/observe_system.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.envs.subgoal_env import NO_HOOK  # noqa: E402
from src.skills.executor import Executor  # noqa: E402
from src.skills.registry import NodeRegistry  # noqa: E402
from src.skills.state import build_state_view  # noqa: E402
from src.training.evaluation import heldout_seeds, wilson_lower, wilson_upper  # noqa: E402
from src.training.node_trainer import (  # noqa: E402
    execution_env_factory,
    load_frozen_composite,
    load_frozen_primitive,
    set_torch_threads,
    subgoal_env_factory,
)
from diagnose_pickup_box import _full_state  # noqa: E402
from h7_nodes import UNLOCKPICKUP, load_unlock_pickup  # noqa: E402
from train_flat_baseline import BASELINE_RUNS, build_flat_ppo  # noqa: E402
from train_primitives import DOORKEY, UNLOCK, build_node, specs  # noqa: E402
from train_solve_doorkey import build_t5  # noqa: E402

RUNS = REPO / "runs" / "v2"


def episode(node, env, seed) -> dict:
    """One argmax episode as `evaluation.run_episode` runs it, plus what failed."""
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    version, sv = 0, build_state_view(env, 0)
    ex = Executor(node)
    seen, loop_node, acting, ended, steps = {_full_state(u)}, None, None, False, 0
    while steps < node.timeout:
        action = ex.act(obs, sv)
        if action is None:
            break
        acting = ex.stack[-1].node.id
        obs, _, terminated, truncated, _ = env.step(action)
        steps += 1
        version += 1
        sv = build_state_view(env, version)
        ex.observe_reward(0.0)
        state = _full_state(u)
        if loop_node is None and state in seen:
            loop_node = acting
        seen.add(state)
        if terminated or truncated:
            ended = True
            break
    success = bool(node.goal(sv))
    label = None
    if not success:
        label = ("env_ended" if ended and steps < node.timeout
                 else f"loop:{loop_node}" if loop_node else f"timeout:{acting}")
    return {"seed": seed, "success": success, "steps": steps, "label": label,
            "selected": ex.selected_children(),
            "child_timeouts": sum(e.event == "CLOSE" and e.outcome == "timeout" and e.depth > 1
                                  for e in ex.trace)}


def run_row(node, env_factory, seeds) -> dict:
    env = env_factory()
    try:
        eps = [episode(node, env, s) for s in seeds]
    finally:
        env.close()
    ok = [e for e in eps if e["success"]]
    bad = [e for e in eps if not e["success"]]
    labels = Counter(e["label"] for e in bad)
    return {
        "node": node.id, "n": len(eps), "successes": len(ok),
        "success_rate": round(len(ok) / len(eps), 4),
        "wilson": [round(wilson_lower(len(ok), len(eps)), 4), round(wilson_upper(len(ok), len(eps)), 4)],
        "steps_on_success": {"mean": round(statistics.mean(e["steps"] for e in ok), 1),
                             "median": statistics.median(e["steps"] for e in ok)} if ok else None,
        "failures": dict(labels.most_common()),
        "example_seeds": {k: [e["seed"] for e in bad if e["label"] == k][:5] for k in labels},
        "sequences_on_success": Counter(" > ".join(e["selected"]) for e in ok).most_common(3),
        "sequences_on_failure": Counter(" > ".join(e["selected"]) for e in bad).most_common(3),
        "mean_child_timeouts": round(statistics.mean(e["child_timeouts"] for e in eps), 3),
        "episodes": eps,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--torch-threads", type=int, default=2)
    args = ap.parse_args()
    set_torch_threads(args.torch_threads)

    registry = NodeRegistry(RUNS)
    _, unlock_door, _, _, unlock_pickup = load_unlock_pickup(registry, verbose=False)
    spec = specs()["go_to_goal"]
    go = build_node(spec)
    load_frozen_primitive(go, registry, spec["env_ids"][0], spec["hook"])
    solve_doorkey = build_t5(unlock_door, go)
    load_frozen_composite(solve_doorkey, registry, DOORKEY, NO_HOOK)
    flat = {}
    for task, env_id in (("unlock", UNLOCK), ("doorkey", DOORKEY)):
        flat[task] = build_flat_ppo(task)
        load_frozen_primitive(flat[task], NodeRegistry(BASELINE_RUNS), env_id, NO_HOOK)

    # Self-check: the loading path must reproduce the regression sweep on the held-out seeds.
    recorded = {r["node"]: r["successes"] for r in
                json.loads((RUNS / "regression.json").read_text(encoding="utf-8"))["rows"]}
    for node, env_id in ((unlock_door, UNLOCK), (solve_doorkey, DOORKEY), (unlock_pickup, UNLOCKPICKUP)):
        got = run_row(node, execution_env_factory(env_id), heldout_seeds(50))["successes"]
        if got != recorded[node.id]:
            raise RuntimeError(f"{node.id}: {got}/50 on the held-out seeds, regression.json has {recorded[node.id]}")
    print("self-check: the three roots reproduce regression.json on the held-out seeds")

    def hier(node, env_id):
        return node, execution_env_factory(env_id)

    def flat_row(task, env_id):
        return flat[task], subgoal_env_factory(flat[task], env_id, NO_HOOK, monitor=False)

    rows = {   # name: (node, env factory, seed base, n, kind)
        "unlock": (*hier(unlock_door, UNLOCK), 70000, 500, "home"),
        "doorkey": (*hier(solve_doorkey, DOORKEY), 71000, 500, "home"),
        "unlockpickup": (*hier(unlock_pickup, UNLOCKPICKUP), 72000, 500, "home"),
        "doorkey5": (*hier(solve_doorkey, "MiniGrid-DoorKey-5x5-v0"), 73000, 200, "zero-shot"),
        "doorkey8": (*hier(solve_doorkey, "MiniGrid-DoorKey-8x8-v0"), 74000, 200, "zero-shot"),
        "doorkey16": (*hier(solve_doorkey, "MiniGrid-DoorKey-16x16-v0"), 75000, 200, "zero-shot"),
        "flat_unlock": (*flat_row("unlock", UNLOCK), 70000, 500, "reference"),
        "flat_doorkey": (*flat_row("doorkey", DOORKEY), 71000, 500, "reference"),
    }
    out = {}
    for name, (node, factory, base, n, kind) in rows.items():
        r = run_row(node, factory, range(base, base + (10 if args.smoke else n)))
        out[name] = {"kind": kind, "seeds": [base, base + r["n"] - 1], **r}
        s = r["steps_on_success"]
        print(f"{name:<13} {kind:<10} {r['successes']:>3}/{r['n']:<3} [{r['wilson'][0]:.3f}, {r['wilson'][1]:.3f}]  "
              f"steps {s['mean'] if s else '-':>6} / {s['median'] if s else '-':>5}  {r['failures']}", flush=True)

    if not args.smoke:
        path = RUNS / "system_observation.json"
        path.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"\nwrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
