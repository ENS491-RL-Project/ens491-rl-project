"""
H-7.3 follow-up: where does the unstable `{0,4} pickup_box` (seed 42, saved as
TRAINING, never frozen) succeed and fail?

A2 failed -- no seed stable within 600,000 env steps -- and the plan stops there.
This is not a remedy; it splits the held-out score of the seed-42 checkpoint by
env and by what the agent holds at the start, plus the real handoff (after
`{1,1} unlock_door` ran on UnlockPickup), so the decision on what to change
starts from a measurement.

    python scripts/diagnose_pickup_box.py           ->  runs/v2/pickup_box_diagnosis.json
    python scripts/diagnose_pickup_box.py --spec2   ->  runs/v2/pickup_box_spec2_diagnosis.json
    python scripts/diagnose_pickup_box.py --trace   ->  runs/v2/pickup_box_spec2_trace.json

[H-7 amendment 1] The attempt-1 spec was withdrawn and its checkpoint moved to
`runs/v2/attempts/0-4_pickup_box_attempt1/`; it is loaded from there with the
withdrawn spec, so this diagnosis still reproduces.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

from src.envs.subgoal_env import (  # noqa: E402
    NO_HOOK,
    ResetHook,
    UnsuitableLayout,
    compose,
    give_key,
    key_off_door,
    open_door,
    reachable_cells,
    relocate_blocker,
)
from src.skills.registry import NodeRegistry  # noqa: E402
from src.training.evaluation import evaluate_node, heldout_seeds  # noqa: E402
from src.training.node_trainer import attach_sb3_primitive_policy, execution_env_factory  # noqa: E402
from eval_h7_preflight import chain  # noqa: E402
from h7_nodes import (  # noqa: E402
    ATTEMPT1_DIR,
    BLOCKED,
    RUNS,
    UNLOCKPICKUP,
    load_unlock_graph,
    pickup_box_attempt1_spec,
)
from train_primitives import build_node  # noqa: E402

N = 50


def main() -> int:
    registry = NodeRegistry(RUNS)
    spec = pickup_box_attempt1_spec()
    node = build_node(spec)
    attach_sb3_primitive_policy(node, UNLOCKPICKUP, spec["hook"])
    registry.load_into(node, run_dir=ATTEMPT1_DIR, require_frozen=False)   # TRAINING, not frozen
    _, t3 = load_unlock_graph(registry, verbose=False)

    cases = []
    for env_id in (UNLOCKPICKUP, BLOCKED):
        for hand, extra in (("nothing", []), ("key", [give_key()])):
            hook = compose(relocate_blocker(), open_door(), *extra)
            cases.append((env_id, hook, None, f"door open, holding {hand}"))
    cases.append((UNLOCKPICKUP, NO_HOOK, chain(t3), "after unlock_door ran (the real handoff)"))

    rows = []
    for env_id, hook, prep, label in cases:
        r = evaluate_node(node, execution_env_factory(env_id, hook),
                          seeds=heldout_seeds(N), prepare=prep)
        rows.append({"env": env_id, "start_state": label, "hook": hook.spec, **r.as_dict()})
        print(f"  {env_id.replace('MiniGrid-', ''):<26}{label:<42}SR {r.success_rate:.2f}"
              f"  mean length {r.mean_length:.1f}")
    out = {"node": node.id, "checkpoint": "seed 42, TRAINING (not stable at 600,000)",
           "spec_hash": node.spec_hash(), "rows": rows}
    path = RUNS / "pickup_box_diagnosis.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path.relative_to(REPO)}")
    return 0


def pose_in_room(side: str) -> ResetHook:
    """Probe only: a random free pose on one side of the door ("left" = the parent's side)."""

    def fn(env):
        u = env.unwrapped
        door_pos = next((x, y) for x in range(u.width) for y in range(u.height)
                        if u.grid.get(x, y) is not None and u.grid.get(x, y).type == "door")
        start = (int(u.agent_pos[0]), int(u.agent_pos[1]))
        cells = [c for c in reachable_cells(u, start) if u.grid.get(*c) is None
                 and ((c[0] < door_pos[0]) if side == "left" else (c[0] > door_pos[0]))]
        if not cells:      # the box stands right behind the door: no right-room pose
            raise UnsuitableLayout(f"probe_pose_in_room:{side}: no free cell")
        c = cells[u._rand_int(0, len(cells))]
        u.agent_pos = np.array(c)
        u.agent_dir = u._rand_int(0, 4)

    return ResetHook(f"probe_pose_in_room:{side}", fn)


def main_spec2() -> int:
    """
    [H-7a.2, after the A2b stop] The same question for amendment 1's pickup_box:
    the seed-42 checkpoint (TRAINING, not stable at 1,000,000) under its own
    reset distribution, split by the room the agent starts in, against the real
    handoff. Writes `runs/v2/pickup_box_spec2_diagnosis.json`; the attempt-1 file
    is left as it is.
    """
    from h7_nodes import h7_primitive_specs, load_h7_primitive

    registry = NodeRegistry(RUNS)
    spec = h7_primitive_specs()["pickup_box"]
    node = build_node(spec)
    attach_sb3_primitive_policy(node, UNLOCKPICKUP, spec["hook"])
    registry.load_into(node, require_frozen=False)                       # TRAINING, not frozen
    _, t3 = load_unlock_graph(registry, verbose=False)
    dh = load_h7_primitive("drop_held", registry, verbose=False)

    base = [open_door(), key_off_door()]
    cases = [
        (spec["hook"], None, "own hook (pose over both rooms)"),
        (compose(*base, pose_in_room("left")), None, "own hook, pose in the LEFT room (parent's side)"),
        (compose(*base, pose_in_room("right")), None, "own hook, pose in the RIGHT room (box's side)"),
        (NO_HOOK, chain(t3, dh), "after unlock_door, drop_held (the real handoff)"),
    ]
    rows = []
    for hook, prep, label in cases:
        r = evaluate_node(node, execution_env_factory(UNLOCKPICKUP, hook),
                          seeds=heldout_seeds(N), prepare=prep)
        rows.append({"env": UNLOCKPICKUP, "start_state": label, "hook": hook.spec, **r.as_dict()})
        print(f"  {label:<52}SR {r.success_rate:.2f} (LB {r.wilson_lower:.3f})"
              f"  mean length {r.mean_length:.1f}")
    out = {"node": node.id, "spec": "H-7 amendment 1",
           "checkpoint": "seed 42, TRAINING (not stable at 1,001,472)",
           "spec_hash": node.spec_hash(), "rows": rows,
           "note": "diagnostic; a reading of one checkpoint, not a tested cause"}
    path = RUNS / "pickup_box_spec2_diagnosis.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path.relative_to(REPO)}")
    return 0


# --------------------------------------------------------------------------
# [H-7a close-out, H-7a.5] --trace: where and how the spec-2 checkpoint fails
# --------------------------------------------------------------------------

TRACE_SAMPLING_SEEDS = (0, 1, 2)
# The readings, fixed by the close-out before the run, for condition H's failures.
LOOP_MIN, KEY_MIN, NAV_MIN, NAV_LOOP_MAX = 0.80, 0.50, 0.50, 0.50


def _find(u, kind: str):
    for x in range(u.width):
        for y in range(u.height):
            o = u.grid.get(x, y)
            if o is not None and o.type == kind:
                return (x, y), o
    return None, None


def _held(u):
    return None if u.carrying is None else u.carrying.type


def _room(x: int, door_x: int) -> str:
    return "left" if x < door_x else ("doorway" if x == door_x else "right")


def _full_state(u) -> tuple:
    """Everything the observation can depend on: the whole grid (every object's
    position, and the door's open/locked state), the agent's pose, what it holds."""
    held = None if u.carrying is None else (u.carrying.type, u.carrying.color)
    return (u.grid.encode().tobytes(), int(u.agent_pos[0]), int(u.agent_pos[1]),
            int(u.agent_dir), held)


def trace_episode(node, env, seed: int, *, mode: str, sampling_seed, prepare) -> dict:
    """
    One episode of `node` through the Executor, as `evaluation.run_episode` runs
    it (same termination, same step limit), with per-step bookkeeping.
    """
    import torch
    from minigrid.core.constants import OBJECT_TO_IDX
    from src.skills.executor import DETERMINISTIC, Executor
    from src.skills.state import build_state_view

    box_idx = OBJECT_TO_IDX["box"]
    u = env.unwrapped
    obs, _ = env.reset(seed=seed)
    if prepare is not None:
        obs = prepare(env, obs)
    if sampling_seed is not None:
        # One torch seed per (sampling seed, env seed): SB3 samples with torch.
        torch.manual_seed(1_000_003 * int(sampling_seed) + int(seed))
    door_pos, door = _find(u, "door")
    box_pos, _ = _find(u, "box")
    door_x = door_pos[0]
    start = {"pos": [int(u.agent_pos[0]), int(u.agent_pos[1])], "dir": int(u.agent_dir),
             "room": _room(int(u.agent_pos[0]), door_x), "holding": _held(u),
             "door_open": bool(door.is_open), "door_locked": bool(door.is_locked)}
    box_behind_door = box_pos == (door_x + 1, door_pos[1])

    def box_in_view() -> bool:
        return bool((u.gen_obs()["image"][:, :, 0] == box_idx).any())

    version = 0
    sv = build_state_view(env, version)
    ex = Executor(node, execution_mode=mode)
    seen = {_full_state(u): 0}
    repeat_step = period = None
    entered_right = start["room"] == "right"
    box_seen = box_in_view()
    key_pickups = drops = 0
    actions = [0] * 7
    steps = 0
    while steps < node.timeout:
        action = ex.act(obs, sv)
        if action is None:
            break
        before = _held(u)
        obs, _, terminated, truncated, _ = env.step(action)
        steps += 1
        actions[int(action)] += 1
        after = _held(u)
        key_pickups += int(after == "key" and before != "key")
        drops += int(before is not None and after is None)
        entered_right |= int(u.agent_pos[0]) > door_x
        box_seen |= box_in_view()
        if mode == DETERMINISTIC and repeat_step is None:
            s = _full_state(u)
            if s in seen:
                repeat_step, period = steps, steps - seen[s]
            else:
                seen[s] = steps
        version += 1
        sv = build_state_view(env, version)
        ex.observe_reward(0.0)
        if terminated or truncated:
            break
    success = bool(node.goal(sv))
    if repeat_step is not None and success:
        raise RuntimeError(f"seed {seed}: a deterministic state repeat, yet the episode succeeded")
    return {
        "seed": seed, "mode": mode, "sampling_seed": sampling_seed,
        "success": success, "steps": steps, "start": start,
        "end_room": _room(int(u.agent_pos[0]), door_x), "end_holding": _held(u),
        "ever_entered_right_room": bool(entered_right), "box_ever_in_view": bool(box_seen),
        "key_pickups": key_pickups, "drops": drops,
        "action_counts": dict(zip(["left", "right", "forward", "pickup", "drop", "toggle", "done"],
                                  actions)),
        "box_behind_door": bool(box_behind_door),
        "state_repeat_step": repeat_step, "state_repeat_period": period,
    }


def summarise_trace(records: list[dict], deterministic: bool) -> dict:
    from src.training.evaluation import wilson_lower

    n = len(records)
    k = sum(r["success"] for r in records)
    fails = [r for r in records if not r["success"]]
    counts = {
        "i_state_repeat": (sum(r["state_repeat_step"] is not None for r in fails)
                           if deterministic else None),
        "ii_key_pickup": sum(r["key_pickups"] >= 1 for r in fails),
        "iii_never_entered_right": sum(not r["ever_entered_right_room"] for r in fails),
        "iv_box_behind_door": sum(r["box_behind_door"] for r in fails),
        "v_box_never_seen": sum(not r["box_ever_in_view"] for r in fails),
    }
    acts = {a: sum(r["action_counts"][a] for r in fails) for a in records[0]["action_counts"]}
    tot = sum(acts.values()) or 1
    ends = {}
    for r in fails:
        key = f"{r['end_room']}/{r['end_holding']}"
        ends[key] = ends.get(key, 0) + 1
    periods = {}
    for r in fails:
        if r["state_repeat_period"] is not None:
            periods[str(r["state_repeat_period"])] = periods.get(str(r["state_repeat_period"]), 0) + 1
    return {
        "n": n, "successes": k, "success_rate": round(k / n, 4),
        "wilson_lower": round(wilson_lower(k, n), 4),
        "wilson_upper": round(1 - wilson_lower(n - k, n), 4),
        "failures": len(fails), "failure_counts": counts,
        "failure_action_shares": {a: round(v / tot, 4) for a, v in acts.items()},
        "failure_end_room_holding": dict(sorted(ends.items())),
        "failure_repeat_periods": dict(sorted(periods.items(), key=lambda kv: int(kv[0]))),
        "successes_box_behind_door": sum(r["box_behind_door"] for r in records if r["success"]),
    }


def readings(det: dict, stoch: dict) -> dict:
    """The close-out's pre-set readings, on condition H's deterministic failures."""
    f = det["failures"]
    c = det["failure_counts"]
    share = {k: (round(v / f, 4) if f else None) for k, v in c.items()}
    loop = bool(f) and c["i_state_repeat"] / f >= LOOP_MIN
    key = bool(f) and c["ii_key_pickup"] / f >= KEY_MIN
    nav = (bool(f) and c["iii_never_entered_right"] / f >= NAV_MIN
           and c["i_state_repeat"] / f < NAV_LOOP_MAX)
    held = [name for name, ok in (("loop", loop), ("key", key), ("navigation", nav)) if ok]
    out = {"judged_on": "condition H, deterministic failures", "failures": f, "shares": share,
           "loop": loop, "key": key, "navigation": nav,
           "reading": held[0] if len(held) == 1 else ("none" if not held else "+".join(held)),
           "rules": {"loop": "(i) in >= 80% of the deterministic failures",
                     "key": "(ii) in >= 50%",
                     "navigation": "(iii) in >= 50% and (i) in < 50%",
                     "else": "no reading"}}
    if loop:
        out["stochastic_150_success_rate"] = stoch["success_rate"]
        out["stochastic_reaches_0.90"] = stoch["success_rate"] >= 0.90
    return out


def main_trace() -> int:
    """
    [H-7a close-out, H-7a.5] The seed-42 spec-2 checkpoint (TRAINING, never
    frozen) on the 50 held-out seeds under two start conditions -- H: after
    `unlock_door` then `drop_held` (both FROZEN, run deterministically); L: its
    own hook with the pose in the left room -- each deterministically and
    stochastically with sampling seeds 0, 1, 2. Evaluation only: no training, no
    checkpoint is written. Writes `runs/v2/pickup_box_spec2_trace.json`.
    """
    from h7_nodes import h7_primitive_specs, load_h7_primitive
    from src.skills.executor import DETERMINISTIC, STOCHASTIC

    registry = NodeRegistry(RUNS)
    spec = h7_primitive_specs()["pickup_box"]
    node = build_node(spec)
    attach_sb3_primitive_policy(node, UNLOCKPICKUP, spec["hook"])
    registry.load_into(node, require_frozen=False)                       # TRAINING, not frozen
    _, t3 = load_unlock_graph(registry, verbose=False)
    dh = load_h7_primitive("drop_held", registry, verbose=False)

    conditions = {
        "H": (NO_HOOK, chain(t3, dh), "after unlock_door then drop_held (the real handoff)"),
        "L": (compose(open_door(), key_off_door(), pose_in_room("left")), None,
              "own hook, pose in the left room"),
    }
    seeds = heldout_seeds(N)
    records, summary = [], {}
    for cname, (hook, prep, label) in conditions.items():
        env = execution_env_factory(UNLOCKPICKUP, hook)()
        try:
            det = [trace_episode(node, env, s, mode=DETERMINISTIC, sampling_seed=None,
                                 prepare=prep) for s in seeds]
            sto = [trace_episode(node, env, s, mode=STOCHASTIC, sampling_seed=k, prepare=prep)
                   for k in TRACE_SAMPLING_SEEDS for s in seeds]
        finally:
            env.close()
        for r in det + sto:
            r["condition"] = cname
        records += det + sto
        summary[cname] = {"label": label, "hook": hook.spec,
                          "deterministic": summarise_trace(det, True),
                          "stochastic": summarise_trace(sto, False),
                          "stochastic_by_sampling_seed": {
                              str(k): sum(r["success"] for r in sto if r["sampling_seed"] == k)
                              for k in TRACE_SAMPLING_SEEDS}}
        d, s = summary[cname]["deterministic"], summary[cname]["stochastic"]
        print(f"  {cname} {label}\n    deterministic SR {d['success_rate']:.2f} "
              f"[{d['wilson_lower']:.3f}, {d['wilson_upper']:.3f}]  failures {d['failures']}  "
              f"{d['failure_counts']}\n    stochastic    SR {s['success_rate']:.2f} "
              f"[{s['wilson_lower']:.3f}, {s['wilson_upper']:.3f}]  failures {s['failures']}  "
              f"{s['failure_counts']}")
    read = readings(summary["H"]["deterministic"], summary["H"]["stochastic"])
    print(f"  reading for condition H: {read['reading']}  {read['shares']}")
    out = {"node": node.id, "spec": "H-7 amendment 1 (specification 2)",
           "checkpoint": "seed 42, TRAINING (not stable at 1,001,472); never frozen",
           "spec_hash": node.spec_hash(), "env": UNLOCKPICKUP,
           "seeds": f"held-out {seeds[0]}-{seeds[-1]}", "sampling_seeds": list(TRACE_SAMPLING_SEEDS),
           "torch_seed_per_episode": "1_000_003 * sampling_seed + env_seed",
           "n_records": len(records), "summary": summary, "readings_condition_H": read,
           "note": ("diagnostic of one checkpoint; a stochastic success rate is not a gate and "
                    "freezes nothing (v2 §5.3 evaluates deterministically)"),
           "episodes": records}
    path = RUNS / "pickup_box_spec2_trace.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    if "--trace" in sys.argv:
        raise SystemExit(main_trace())
    raise SystemExit(main_spec2() if "--spec2" in sys.argv else main())
