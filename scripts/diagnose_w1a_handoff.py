"""
W1a handoff diagnostic (`docs/plans/2026-11-15_W1a_handoff_diagnostic_plan.md`):
why do some `pickup_box` models time out at the handoff? Evaluation only.

Nothing is trained, frozen or registered, no rule changes, and no existing file
under `runs/v2/` is written. Models, hashes and seeding come from
`scripts/eval_w1a.py`; the per-step bookkeeping from `diagnose_pickup_box.py`'s
trace (H-7a.5).

Episodes: the W1a handoff row, sampled: UnlockPickup from the env's reset,
unlock_door -> drop_held (FROZEN, argmax), then pickup_box sampled with
torch.manual_seed(1_000_003 * 0 + env_seed), env seeds 51000-51499.

    gate   the unchanged condition must reproduce W1a's sampled handoff episodes
           (validity, success, steps) for every model, so the per-seed success
           counts of w1a_pickup_box_sampled.json and _ext.json exactly
    trace  per episode: success, steps, end room, held at the end, key pick-ups,
           drops, steps holding the key, right room entered (and when), box in
           view (and when); summarised for failures and successes per seed
    K      counterfactual: the dropped key is taken off the grid before
           pickup_box starts (same models, env seeds, sampling seed)
    T      counterfactual: pickup_box may run 240 steps instead of 120 (the
           env's 288-step limit still applies); same weights, a node copy whose
           only difference is the timeout

K and T are diagnostics, never a gate, and not results for pickup_box.

    python scripts/diagnose_w1a_handoff.py --parts gate --out <path>        step 1
    python scripts/diagnose_w1a_handoff.py --parts gate,trace --out <path>  step 2
    python scripts/diagnose_w1a_handoff.py --parts gate,trace,K,T           all  ->  runs/v2/w1a_handoff_diagnostic.json
    python scripts/diagnose_w1a_handoff.py --smoke                          10 seeds, all parts, nothing written
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

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from minigrid.core.constants import OBJECT_TO_IDX  # noqa: E402

from src.envs.subgoal_env import NO_HOOK  # noqa: E402
from src.skills.executor import STOCHASTIC, Executor  # noqa: E402
from src.skills.registry import NodeRegistry  # noqa: E402
from src.skills.state import build_state_view  # noqa: E402
from src.training.evaluation import wilson_lower, wilson_upper  # noqa: E402
from src.training.node_trainer import execution_env_factory, set_torch_threads  # noqa: E402
from diagnose_pickup_box import _find, _held, _room  # noqa: E402   H-7a.5's trace helpers
from eval_h7_preflight import chain  # noqa: E402
from eval_w1a import (  # noqa: E402
    EXT_SEEDS,
    HANDOFF_SEEDS,
    OUT as W1A_OUT,
    OUT_EXT as W1A_OUT_EXT,
    SAMPLING_SEED,
    TEST_SEEDS,
    load_run_store_model,
    load_seed42,
    torch_seed,
)
from h7_nodes import RUNS, UNLOCKPICKUP, h7_primitive_specs, load_h7_primitive, load_unlock_graph  # noqa: E402
from train_primitives import build_node  # noqa: E402

OUT = RUNS / "w1a_handoff_diagnostic.json"
PARTS = ("gate", "trace", "K", "T")
SMOKE_N = 10
T_TIMEOUT = 240
BOX = OBJECT_TO_IDX["box"]

# Fixed by the plan before any number of this diagnostic was seen.
FAILING = (46, 47, 50, 52)          # W1a's failing seeds
PASSING = (43, 44, 45, 48, 49, 51)
LIFT = 0.95                          # Key / Time: pooled success under K / T >= 0.95
SEARCH_SHARE = 0.60                  # Search: share of unchanged failures with no key pick-up
                                     # and the box never in view, if neither K nor T lifts

IMPLEMENTATION_CHOICES = {
    "gate_strictness": ("episode for episode (validity, success, steps) against W1a's recorded "
                        "sampled handoff episodes, which implies the per-seed counts the plan asks for"),
    "K": ("after the chain, if the episode is valid and a key lies on the grid, that cell is set "
          "empty and the observation is regenerated through the env's own observation wrappers; "
          "before the change the regenerated observation must equal the one the chain returned"),
    "T": ("a second SkillNode built from the same spec with timeout 240, sharing the loaded "
          "policy object; its spec_hash differs and it is never saved. Within the first 120 steps "
          "a T episode must equal the unchanged one (checked: every unchanged success recurs with "
          "the same steps, and every T success within 120 steps was an unchanged success)"),
    "box_in_view": "box_in_view as H-7a.5: the agent's 7x7 view (gen_obs) contains a box; step 0 = at the start",
    "right_room": "agent x > door x; step 0 if pickup_box starts there",
    "holding_key_steps": "env steps after which the agent carries a key",
    "seed_42": "traced and labelled exploratory; not in any pooled number or reading",
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def current_obs(env):
    """The pipeline's observation of the env's current state (no step taken)."""
    layers, e = [], env
    while isinstance(e, gym.Wrapper):
        if isinstance(e, gym.ObservationWrapper):
            layers.append(e)
        e = e.env
    obs = env.unwrapped.gen_obs()
    for w in reversed(layers):
        obs = w.observation(obs)
    return obs


def box_in_view(u) -> bool:
    return bool((u.gen_obs()["image"][:, :, 0] == BOX).any())


def timeout_copy(node, spec, timeout: int):
    """Same weights, same goal, longer timeout. Diagnostic only; never saved."""
    t = build_node({**spec, "timeout": timeout})
    t.policy = node.policy
    t.status = node.status
    return t


# --------------------------------------------------------------------------
# one traced episode
# --------------------------------------------------------------------------

def trace_episode(node, env, seed, *, prepare, intervention=None) -> dict:
    obs, _ = env.reset(seed=seed)
    obs = prepare(env, obs)
    u = env.unwrapped
    sv = build_state_view(env, 0)
    door = sv.door()
    valid = bool(door is not None and door.is_open and sv.carrying_type is None)
    key_pos, _ = _find(u, "key")
    key_removed = False
    if intervention == "K" and valid and key_pos is not None:
        if not np.array_equal(current_obs(env), obs):
            raise RuntimeError(f"seed {seed}: regenerated observation differs from the chain's")
        u.grid.set(*key_pos, None)
        obs = current_obs(env)
        sv = build_state_view(env, 0)
        key_removed = True
    torch.manual_seed(torch_seed(SAMPLING_SEED, seed))

    door_pos, _ = _find(u, "door")
    door_x = door_pos[0]
    start_room = _room(int(u.agent_pos[0]), door_x)
    right_step = 0 if int(u.agent_pos[0]) > door_x else None
    box_step = 0 if box_in_view(u) else None
    ex = Executor(node, execution_mode=STOCHASTIC)
    key_pickups = drops = holding_key = 0
    steps, version, env_done = 0, 0, False
    while steps < node.timeout:
        action = ex.act(obs, sv)
        if action is None:
            break
        before = _held(u)
        obs, _, terminated, truncated, _ = env.step(action)
        steps += 1
        after = _held(u)
        key_pickups += int(after == "key" and before != "key")
        drops += int(before is not None and after is None)
        holding_key += int(after == "key")
        if right_step is None and int(u.agent_pos[0]) > door_x:
            right_step = steps
        if box_step is None and box_in_view(u):
            box_step = steps
        version += 1
        sv = build_state_view(env, version)
        ex.observe_reward(0.0)
        if terminated or truncated:
            env_done = True
            break
    success = bool(node.goal(sv))
    return {"seed": seed, "valid": valid, "success": success, "steps": steps,
            "timeout": (not success) and steps >= node.timeout,
            "env_ended": (not success) and env_done,
            "start_room": start_room, "key_on_grid_at_start": key_pos is not None,
            "key_removed": key_removed,
            "end_room": _room(int(u.agent_pos[0]), door_x), "end_holding": _held(u),
            "key_pickups": key_pickups, "drops": drops, "steps_holding_key": holding_key,
            "right_room_step": right_step, "box_in_view_step": box_step}


def run_condition(node, seeds, prepare, intervention=None) -> list[dict]:
    env = execution_env_factory(UNLOCKPICKUP, NO_HOOK)()
    try:
        return [trace_episode(node, env, s, prepare=prepare, intervention=intervention)
                for s in seeds]
    finally:
        env.close()


# --------------------------------------------------------------------------
# summaries
# --------------------------------------------------------------------------

def _mean(xs):
    return round(sum(xs) / len(xs), 2) if xs else None


def _share(k, n):
    return round(k / n, 4) if n else None


def describe(group: list[dict]) -> dict:
    n = len(group)
    ends: dict[str, int] = {}
    for r in group:
        key = f"{r['end_room']}/{r['end_holding']}"
        ends[key] = ends.get(key, 0) + 1
    no_key_no_box = sum(r["key_pickups"] == 0 and r["box_in_view_step"] is None for r in group)
    return {
        "n": n,
        "key_picked_up": sum(r["key_pickups"] >= 1 for r in group),
        "key_picked_up_share": _share(sum(r["key_pickups"] >= 1 for r in group), n),
        "mean_key_pickups": _mean([r["key_pickups"] for r in group]),
        "mean_steps_holding_key": _mean([r["steps_holding_key"] for r in group]),
        "mean_drops": _mean([r["drops"] for r in group]),
        "never_entered_right": sum(r["right_room_step"] is None for r in group),
        "never_entered_right_share": _share(sum(r["right_room_step"] is None for r in group), n),
        "mean_right_room_step": _mean([r["right_room_step"] for r in group
                                       if r["right_room_step"] is not None]),
        "box_never_in_view": sum(r["box_in_view_step"] is None for r in group),
        "box_never_in_view_share": _share(sum(r["box_in_view_step"] is None for r in group), n),
        "mean_box_in_view_step": _mean([r["box_in_view_step"] for r in group
                                        if r["box_in_view_step"] is not None]),
        "no_key_pickup_and_box_never_in_view": no_key_no_box,
        "no_key_pickup_and_box_never_in_view_share": _share(no_key_no_box, n),
        "start_room": {k: sum(r["start_room"] == k for r in group) for k in ("left", "doorway", "right")},
        "timeouts": sum(r["timeout"] for r in group),
        "env_ended": sum(r["env_ended"] for r in group),
        "end_room_holding": dict(sorted(ends.items())),
        "mean_steps": _mean([r["steps"] for r in group]),
    }


def rate(records: list[dict]) -> dict:
    valid = [r for r in records if r["valid"]]
    n, k = len(valid), sum(r["success"] for r in valid)
    return {"episodes": len(records), "valid": n, "successes": k,
            "success_rate": round(k / n, 4) if n else None,
            "wilson_lower": round(wilson_lower(k, n), 4) if n else None,
            "wilson_upper": round(wilson_upper(k, n), 4) if n else None}


def summarise(records: list[dict]) -> dict:
    valid = [r for r in records if r["valid"]]
    return {**rate(records),
            "failures": describe([r for r in valid if not r["success"]]),
            "successes_described": describe([r for r in valid if r["success"]])}


def compact(records: list[dict]) -> list:
    return [[r["seed"], int(r["valid"]), int(r["success"]), r["steps"], r["start_room"],
             r["end_room"], r["end_holding"], r["key_pickups"], r["drops"], r["steps_holding_key"],
             r["right_room_step"], r["box_in_view_step"]] for r in records]


COMPACT_FIELDS = ["seed", "valid", "success", "steps", "start_room", "end_room", "end_holding",
                  "key_pickups", "drops", "steps_holding_key", "right_room_step", "box_in_view_step"]


# --------------------------------------------------------------------------
# gate and checks
# --------------------------------------------------------------------------

def w1a_sampled_handoff(seed: int) -> dict:
    """W1a's recorded sampled handoff episodes for one model: seed -> (valid, success, steps)."""
    path = W1A_OUT_EXT if seed in EXT_SEEDS else W1A_OUT
    row = json.loads(path.read_text(encoding="utf-8"))["models"][str(seed)]["rows"]["handoff"]
    return {e[0]: (bool(e[2]), bool(e[3]), int(e[4])) for e in row["episodes"]
            if e[1] == SAMPLING_SEED}


def gate(seed: int, records: list[dict]) -> dict:
    ref = w1a_sampled_handoff(seed)
    mine = {r["seed"]: (r["valid"], r["success"], r["steps"]) for r in records}
    diff = [s for s in mine if mine[s] != ref.get(s)]
    sub = {s: ref[s] for s in mine}
    return {"episodes_compared": len(mine), "episodes_differing": len(diff),
            "first_differences": [{"seed": s, "w1a": ref.get(s), "now": mine[s]} for s in diff[:5]],
            "successes_w1a": sum(v[0] and v[1] for v in sub.values()),
            "successes_now": sum(r["valid"] and r["success"] for r in records),
            "passed": not diff}


def check_t(unchanged: list[dict], t_records: list[dict]) -> None:
    """Within the first 120 steps a T episode must be the unchanged episode."""
    for u0, t in zip(unchanged, t_records):
        if u0["seed"] != t["seed"] or u0["valid"] != t["valid"]:
            raise RuntimeError(f"T: episode mismatch at seed {u0['seed']}")
        if u0["success"] and not (t["success"] and t["steps"] == u0["steps"]):
            raise RuntimeError(f"T: seed {u0['seed']} succeeded unchanged but not identically under T")
        if t["success"] and t["steps"] <= 120 and not u0["success"]:
            raise RuntimeError(f"T: seed {u0['seed']} succeeded within 120 under T only")


# --------------------------------------------------------------------------
# readings
# --------------------------------------------------------------------------

def pooled(models: dict, seeds, part: str) -> dict:
    recs = [r for s in seeds for r in models[str(s)]["_records"][part]]
    return summarise(recs)


def readings(models: dict, parts) -> dict:
    out = {}
    for name, seeds in (("failing", FAILING), ("passing", PASSING)):
        g = {"seeds": list(seeds), "unchanged": pooled(models, seeds, "unchanged")}
        for p in ("K", "T"):
            if p in parts:
                g[p] = pooled(models, seeds, p)
        if "K" in parts and "T" in parts:
            key = g["K"]["success_rate"] >= LIFT
            tim = g["T"]["success_rate"] >= LIFT
            f = g["unchanged"]["failures"]
            search = (not key and not tim and f["n"] > 0
                      and f["no_key_pickup_and_box_never_in_view"] >= SEARCH_SHARE * f["n"])
            held = [k for k, ok in (("Key", key), ("Time", tim), ("Search", search)) if ok]
            g["reading"] = {"key": key, "time": tim, "search": search,
                            "reading": " + ".join(held) if held else "no reading"}
        out[name] = g
    out["rules"] = {"key": f"pooled success under K >= {LIFT}",
                    "time": f"pooled success under T >= {LIFT}",
                    "search": (f"neither, and >= {SEARCH_SHARE:.0%} of unchanged failures with no key "
                               "pick-up and the box never in view"),
                    "else": "no reading",
                    "judged_on": "the four failing seeds pooled; passing seeds shown alike"}
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default=",".join(PARTS))
    ap.add_argument("--out", default=None, help=f"default {OUT.relative_to(REPO)} (full run only)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--torch-threads", type=int, default=2)
    args = ap.parse_args()
    parts = tuple(p for p in args.parts.split(",") if p)
    if "gate" not in parts or any(p not in PARTS for p in parts):
        raise SystemExit(f"--parts must include gate; choose from {PARTS}")
    out_path = Path(args.out) if args.out else (OUT if set(parts) == set(PARTS) else None)
    if not args.smoke:
        if out_path is None:
            raise SystemExit("a partial run needs --out (never under runs/v2/)")
        if out_path.exists():
            raise SystemExit(f"{out_path} exists; it is never overwritten")
    set_torch_threads(args.torch_threads)
    t0 = time.perf_counter()

    registry = NodeRegistry(RUNS)
    _, t3 = load_unlock_graph(registry, verbose=False)
    dh = load_h7_primitive("drop_held", registry, verbose=False)
    prep = chain(t3, dh)
    seeds = HANDOFF_SEEDS[:SMOKE_N] if args.smoke else HANDOFF_SEEDS
    spec = h7_primitive_specs()["pickup_box"]

    w1a = json.loads(W1A_OUT_EXT.read_text(encoding="utf-8"))["verdict"]["seed_passes"]
    if tuple(int(s) for s, v in w1a.items() if v is False) != FAILING:
        raise RuntimeError(f"W1a's failing seeds are not {FAILING}: {w1a}")

    models, all_gates = {}, True
    for seed in (42, *TEST_SEEDS, *EXT_SEEDS):
        node, _, info = load_seed42(registry) if seed == 42 else load_run_store_model(seed)
        recs = {"unchanged": run_condition(node, seeds, prep)}
        g = gate(seed, recs["unchanged"])
        all_gates &= g["passed"]
        line = f"  seed {seed:>2}: gate {g['successes_now']}/{g['successes_w1a']} " \
               f"({g['episodes_differing']} differ) {'PASSED' if g['passed'] else 'FAILED'}"
        if "K" in parts:
            recs["K"] = run_condition(node, seeds, prep, intervention="K")
        if "T" in parts:
            recs["T"] = run_condition(timeout_copy(node, spec, T_TIMEOUT), seeds, prep)
            check_t(recs["unchanged"], recs["T"])
        m = {**{k: v for k, v in info.items() if k != "recorded_heldout"},
             "label": "exploratory; in no pooled number" if seed == 42
             else ("failing" if seed in FAILING else "passing"),
             "gate": g, "_records": recs}
        if "trace" in parts:
            m["unchanged"] = summarise(recs["unchanged"])
        for p in ("K", "T"):
            if p in parts:
                m[p] = summarise(recs[p])
                line += f" | {p} {m[p]['successes']}/{m[p]['valid']} ({m[p]['success_rate']})"
        if "trace" in parts:
            u = m["unchanged"]
            f = u["failures"]
            line += (f" | unchanged {u['successes']}/{u['valid']}; failures {f['n']}: key "
                     f"{f['key_picked_up']}, never right {f['never_entered_right']}, box never "
                     f"seen {f['box_never_in_view']}, timeouts {f['timeouts']}")
        print(line, flush=True)
        models[str(seed)] = m

    out = {
        "plan": "docs/plans/2026-11-15_W1a_handoff_diagnostic_plan.md",
        "parts": list(parts), "env": UNLOCKPICKUP,
        "handoff_seeds": f"{seeds[0]}-{seeds[-1]}", "sampling_seed": SAMPLING_SEED,
        "torch_seed_per_episode": "1_000_003 * sampling_seed + env_seed, set after the chain (and after K's change)",
        "chain": [t3.id, dh.id], "t_timeout": T_TIMEOUT if "T" in parts else None,
        "failing_seeds": list(FAILING), "passing_seeds": list(PASSING),
        "implementation_choices": IMPLEMENTATION_CHOICES,
        "gate_passed_all": all_gates,
        "note": ("evaluation only; K and T are counterfactual diagnostics, not results for pickup_box, "
                 "and never a gate"),
    }
    if not all_gates:
        print("gate FAILED: stop")
    elif "trace" in parts:
        out["readings"] = readings(models, parts)
        for name in ("failing", "passing"):
            g = out["readings"][name]
            f = g["unchanged"]["failures"]
            msg = (f"  {name} pooled: unchanged {g['unchanged']['successes']}/{g['unchanged']['valid']}"
                   f"; failures {f['n']}, no key and box never seen {f['no_key_pickup_and_box_never_in_view']}")
            for p in ("K", "T"):
                if p in g:
                    msg += f"; {p} {g[p]['successes']}/{g[p]['valid']} = {g[p]['success_rate']}"
            if "reading" in g:
                msg += f"  -> {g['reading']['reading']}"
            print(msg)
    out["models"] = {s: {**{k: v for k, v in m.items() if k != "_records"},
                         "episodes": {"fields": COMPACT_FIELDS,
                                      **{p: compact(r) for p, r in m["_records"].items()}}}
                     for s, m in models.items()}
    out["wall_seconds"] = round(time.perf_counter() - t0, 1)
    if args.smoke:
        print(f"smoke run, {out['wall_seconds']}s; nothing written")
        return 0 if all_gates else 1
    out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {out_path} ({out['wall_seconds']}s)")
    return 0 if all_gates else 1


if __name__ == "__main__":
    raise SystemExit(main())
