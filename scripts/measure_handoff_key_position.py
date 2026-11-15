"""
Where does the dropped key lie at the real `pickup_box` handoff? Evaluation only.

Run during the review of `docs/plans/2026-11-15_W1a_spec3_plan.md` (revision 1
said the key lies next to the door; this measured it). UnlockPickup from the
env's reset, `{1,1} unlock_door` -> `{0,9} drop_held` (FROZEN, argmax), env seeds
51000-51499 (W1a's handoff row). The chain is deterministic and independent of
`pickup_box`, so these are exactly the start states W1a and its handoff
diagnostic scored. Per valid state (door open, hand empty): is the key
orthogonally adjacent to the door (`key_off_door`'s rejection criterion), on the
door's approach cell, in front of the agent; where is the agent and which way
does it face. Joined with the diagnostic's unchanged sampled outcomes.

    python scripts/measure_handoff_key_position.py  ->  runs/v2/w1a_handoff_key_position.json (never overwritten)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.envs.subgoal_env import NO_HOOK  # noqa: E402
from src.skills.registry import NodeRegistry  # noqa: E402
from src.skills.state import build_state_view  # noqa: E402
from src.training.evaluation import wilson_lower, wilson_upper  # noqa: E402
from src.training.node_trainer import execution_env_factory  # noqa: E402
from diagnose_pickup_box import _find  # noqa: E402
from eval_h7_preflight import chain  # noqa: E402
from h7_nodes import RUNS, UNLOCKPICKUP, load_h7_primitive, load_unlock_graph  # noqa: E402

OUT = RUNS / "w1a_handoff_key_position.json"
SEEDS = range(51_000, 51_500)
FAILING, PASSING = (46, 47, 50, 52), (43, 44, 45, 48, 49, 51)
DIRS = ["east", "south", "west", "north"]


def adj(a, b) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"{OUT.relative_to(REPO)} exists; it is never overwritten")
    registry = NodeRegistry(RUNS)
    _, t3 = load_unlock_graph(registry, verbose=False)
    dh = load_h7_primitive("drop_held", registry, verbose=False)
    prep = chain(t3, dh)
    env = execution_env_factory(UNLOCKPICKUP, NO_HOOK)()
    states = {}
    try:
        for s in SEEDS:
            obs, _ = env.reset(seed=s)
            prep(env, obs)
            u = env.unwrapped
            sv = build_state_view(env, 0)
            d = sv.door()
            door, _ = _find(u, "door")
            key, _ = _find(u, "key")
            agent = (int(u.agent_pos[0]), int(u.agent_pos[1]))
            front = tuple(int(v) for v in u.front_pos)
            approach = (door[0] - 1, door[1])
            states[s] = {
                "valid": bool(d is not None and d.is_open and sv.carrying_type is None),
                "key_on_grid": key is not None,
                "key_adjacent_to_door": key is not None and adj(key, door),
                "key_on_approach_cell": key is not None and key == approach,
                "key_in_front_of_agent": key is not None and key == front,
                "agent_on_approach_cell": agent == approach,
                "agent_facing": DIRS[int(u.agent_dir)],
                "key_relative_to_door": None if key is None else [key[0] - door[0], key[1] - door[1]],
                "steps_used_by_chain": int(u.step_count),
            }
    finally:
        env.close()

    valid = [s for s in SEEDS if states[s]["valid"]]
    n = len(valid)

    def count(field):
        k = sum(bool(states[s][field]) for s in valid)
        return {"n": k, "share": round(k / n, 4)}

    facing: dict[str, int] = {}
    rel: dict[str, int] = {}
    for s in valid:
        facing[states[s]["agent_facing"]] = facing.get(states[s]["agent_facing"], 0) + 1
        r = str(tuple(states[s]["key_relative_to_door"]))
        rel[r] = rel.get(r, 0) + 1
    summary = {f: count(f) for f in ("key_on_grid", "key_adjacent_to_door", "key_on_approach_cell",
                                      "key_in_front_of_agent", "agent_on_approach_cell")}
    summary["agent_facing"] = dict(sorted(facing.items()))
    summary["key_relative_to_door"] = dict(sorted(rel.items(), key=lambda kv: -kv[1]))
    steps = [states[s]["steps_used_by_chain"] for s in valid]
    summary["steps_used_by_chain"] = {"mean": round(sum(steps) / n, 2), "max": max(steps)}

    diag = json.loads((RUNS / "w1a_handoff_diagnostic.json").read_text(encoding="utf-8"))
    split = {}
    for name, group in (("failing", FAILING), ("passing", PASSING)):
        for flag in (True, False):
            k = m = 0
            for seed in group:
                for e in diag["models"][str(seed)]["episodes"]["unchanged"]:
                    if e[1] and states[e[0]]["valid"] and states[e[0]]["key_adjacent_to_door"] == flag:
                        m += 1
                        k += e[2]
            split[f"{name}, key adjacent to door = {flag}"] = {
                "episodes": m, "successes": k, "success_rate": round(k / m, 4) if m else None,
                "wilson": [round(wilson_lower(k, m), 4), round(wilson_upper(k, m), 4)] if m else None}

    out = {"script": "scripts/measure_handoff_key_position.py", "env": UNLOCKPICKUP,
           "seeds": "51000-51499", "chain": [t3.id, dh.id], "valid_states": n,
           "summary": summary, "success_by_key_adjacency_unchanged_sampled": split,
           "note": "evaluation only; describes the start states, tests nothing",
           "states": {str(s): v for s, v in states.items()}}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"valid_states": n, "summary": summary}, indent=1))
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
