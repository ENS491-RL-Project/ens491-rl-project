"""
H-4 exit gate: score each frozen primitive from PARENT-REACHABLE states.

[A: F29] A primitive trained on `SubgoalEnv` only ever starts from states its own
reset hook produces. A parent hands it whatever state the previous child left
behind. A primitive that scores 1.00 from its hook distribution and 0.30 from the
states its parent actually produces is not ready to be a child, and the failure
would surface later as "the composite will not learn" -- which is much more
expensive to diagnose.

The handoffs that actually occur inside `{1,1} unlock_door` and
`{2,1} solve_doorkey`:

  get_key     after the composite tried open_door first and it failed
              (goal-only masking means that is a legal, and likely, first move)
  open_door   after get_key succeeded -- agent somewhere else, now holding a key
  go_to_goal  after the door has been unlocked and opened by the pair above

Usage:  python scripts/eval_parent_reachable.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.envs.subgoal_env import NO_HOOK, open_door as open_door_hook  # noqa: E402
from src.skills.executor import Executor  # noqa: E402
from src.skills.registry import NodeRegistry  # noqa: E402
from src.skills.state import build_state_view  # noqa: E402
from src.training.evaluation import evaluate_node, heldout_seeds  # noqa: E402
from src.training.node_trainer import execution_env_factory, load_frozen_primitive  # noqa: E402
from train_primitives import DOORKEY, UNLOCK, build_node, specs  # noqa: E402

RUNS = REPO / "runs" / "v2"
N_EVAL = 50


def roll(node, env, obs, max_steps=None):
    """Run one frozen node to its own end. Returns the observation it leaves."""
    version = 0
    sv = build_state_view(env, version)
    ex = Executor(node)
    limit = max_steps or node.timeout
    for _ in range(limit):
        a = ex.act(obs, sv)
        if a is None:
            break
        obs, _, term, trunc, _ = env.step(a)
        version += 1
        sv = build_state_view(env, version)
        ex.observe_reward(0.0)
        if term or trunc:
            break
    return obs


def prepare_with(*nodes):
    """A `prepare` hook that rolls these frozen nodes forward in order."""
    def _prepare(env, obs):
        for n in nodes:
            obs = roll(n, env, obs)
        return obs
    return _prepare


def main() -> int:
    registry = NodeRegistry(RUNS)
    all_specs = specs()

    loaded = {}
    for name in ("get_key", "open_door", "go_to_goal"):
        spec = all_specs[name]
        node = build_node(spec)
        try:
            load_frozen_primitive(node, registry, spec["env_ids"][0], spec["hook"])
        except Exception as exc:
            print(f"SKIP {name}: {exc}")
            continue
        loaded[name] = (node, spec)
        print(f"loaded {node}")

    if len(loaded) < 3:
        print("\nnot all three primitives are frozen yet")
        return 1

    get_key, _ = loaded["get_key"]
    open_door, _ = loaded["open_door"]
    go_to_goal, _ = loaded["go_to_goal"]

    # Only states a parent in the curriculum can actually produce count towards
    # the gate. Two kinds are excluded:
    #   - states where the goal is unreachable by construction (the door is still
    #     shut), where failing IS correct and is the signal the parent learns
    #     from under goal-only masking [A: F1];
    #   - artificial probes, kept as diagnostics because they say something about
    #     the policy, but not gating.
    cases = [
        # node,     env,      hook,            prepare,                   label, gate
        (get_key, UNLOCK, NO_HOOK, None,
         "episode start", True),
        (get_key, UNLOCK, NO_HOOK, prepare_with(open_door),
         "after a failed open_door", True),
        (open_door, UNLOCK, NO_HOOK, prepare_with(get_key),
         "after get_key succeeded", True),
        (open_door, DOORKEY, NO_HOOK, prepare_with(get_key),
         "after get_key succeeded", True),
        (go_to_goal, DOORKEY, NO_HOOK, prepare_with(get_key, open_door),
         "after the door was unlocked", True),
        # Diagnostic, not gating: no parent produces "door open, nothing
        # carried" -- unlock_door always leaves the agent holding the key. This
        # measures sensitivity to the start pose alone.
        (go_to_goal, DOORKEY, open_door_hook(), None,
         "[probe] door open, empty-handed", False),
        # Expected failure, not gating: solve_doorkey may pick go_to_goal first,
        # with the door still shut. The goal is unreachable; timing out is right.
        (go_to_goal, DOORKEY, NO_HOOK, None,
         "[expected fail] door still shut", False),
    ]

    print(f"\n{'node':<22}{'env':<22}{'start state':<34}{'SR':>6}  {'len':>6}   gate")
    print("-" * 100)
    results = []
    for node, env_id, hook, prepare, label, gate in cases:
        res = evaluate_node(
            node,
            execution_env_factory(env_id, hook),
            seeds=heldout_seeds(N_EVAL),
            prepare=prepare,
        )
        print(f"{node.id:<22}{env_id.replace('MiniGrid-',''):<22}{label:<34}"
              f"{res.success_rate:>6.2f}  {res.mean_length:>6.1f}   "
              f"{'GATE' if gate else '-'}")
        results.append({
            "node": node.id, "env": env_id, "start_state": label, "gate": gate,
            **res.as_dict(),
        })

    out = RUNS / "parent_reachable_eval.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(REPO)}")

    weak = [r for r in results if r["gate"] and r["success_rate"] < 0.90]
    if weak:
        print("\nBELOW 0.90 from a parent-reachable state:")
        for r in weak:
            print(f"  {r['node']} on {r['env']} from '{r['start_state']}': "
                  f"SR {r['success_rate']:.2f}")
        print("This is the F29 risk. The plan's response: widen the layout and "
              "start-pose mix, or train the primitive inside the parent env.")
    else:
        print("\nGATE PASSED: every primitive clears 0.90 from every start state a "
              "parent can actually hand it.")
        for r in results:
            if not r["gate"]:
                print(f"  (non-gating) {r['node']} '{r['start_state']}': "
                      f"SR {r['success_rate']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
