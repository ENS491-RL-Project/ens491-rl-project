"""
H-7.1 / H-7.3 / H-7.5: the pre-flight for the reuse experiment.

The H-7 plan was written without access to this repo's conda env: its "verified
facts" (V1-V7) were measured on a clean minigrid 3.1.0 install. This script
re-measures every one of them on the code that will actually run, audits the new
reset distributions for solvability, and takes the F29 transfer rows the plan
gates on. Where a measurement disagrees with the plan, the measurement wins and
is recorded as a deviation (plan header).

Stages, each merged into `runs/v2/h7_preflight.json`:

  base        V1-V7; solvability audits of the attempt-1 pickup_box hook and
              move_blocker's hook; F29 rows for {1,1} unlock_door on UnlockPickup
              (gating, A1) and on BlockedUnlockPickup (expected fail). H-7.1;
              not re-run after amendment 1, so its record stays as measured
  audit       [H-7a.1] solvability audit of the two amendment-1 reset
              distributions, {0,9} drop_held's and the new {0,4} pickup_box's,
              500 resets each by exhaustive search. Counts toward gate_passed
  pickup_box  [H-7a.2] three F29 rows on UnlockPickup, 50 held-out seeds:
              drop_held after unlock_door (gating, A2a); pickup_box after
              unlock_door then drop_held (gating, A2b); pickup_box after
              unlock_door with the key still in hand (not gating: SR, drop share,
              what it holds at the end)
  blocked     H-7b: refused until H-7b is planned (amendment 1, section 2)
  full_hand   every child any H-7 composite can select, started holding nothing,
              the key, and (BlockedUnlockPickup) the ball. Not gating

Usage:
    python scripts/eval_h7_preflight.py --stage base
    python scripts/eval_h7_preflight.py --stage audit
    python scripts/eval_h7_preflight.py --stage pickup_box
    python scripts/eval_h7_preflight.py --stage full_hand
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from minigrid.core.constants import OBJECT_TO_IDX  # noqa: E402

from src.envs.option_env import ResetHookEnv  # noqa: E402
from src.envs.subgoal_env import (  # noqa: E402
    NO_HOOK,
    ResetHook,
    _find,
    compose,
    give_key,
    open_door,
    randomize_agent_pose,
    reachable_cells,
    relocate_blocker,
)
from src.skills.executor import Executor  # noqa: E402
from src.skills.registry import NodeRegistry  # noqa: E402
from src.skills.state import build_state_view  # noqa: E402
from src.training.device import describe  # noqa: E402
from src.training.evaluation import (  # noqa: E402
    evaluate_node,
    heldout_seeds,
    stopping_seeds,
    wilson_lower,
)
from src.training.node_trainer import execution_env_factory  # noqa: E402
from h7_nodes import (  # noqa: E402
    BLOCKED,
    UNLOCKPICKUP,
    build_unlock_pickup,
    h7_primitive_specs,
    load_h7_primitive,
    load_unlock_graph,
    pickup_box_attempt1_spec,
)
from train_primitives import UNLOCK  # noqa: E402

RUNS = REPO / "runs" / "v2"
OUT = RUNS / "h7_preflight.json"
N_EVAL = 50
N_FACTS = 400
GATE = 0.90
LEFT, RIGHT, FORWARD, PICKUP, DROP, TOGGLE = 0, 1, 2, 3, 4, 5
ACTION_NAMES = {0: "left", 1: "right", 2: "forward", 3: "pickup", 4: "drop",
                5: "toggle", 6: "done"}
DIRS = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}


def _adj(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _neigh(c):
    x, y = c
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _in_view(u, type_):
    return bool((u.gen_obs()["image"][:, :, 0] == OBJECT_TO_IDX[type_]).any())


def hold(type_: str) -> ResetHook:
    """Move the first object of `type_` off the grid into the agent's hand (probe only)."""

    def fn(env):
        u = env.unwrapped
        if u.carrying is not None:
            return
        pos, obj = _find(u, type_)
        if obj is None:
            raise RuntimeError(f"hold: no {type_} in this layout")
        u.grid.set(pos[0], pos[1], None)
        u.carrying = obj
        u.carrying.cur_pos = np.array([-1, -1])

    return ResetHook(f"hold:{type_}", fn)


def roll(node, env, obs):
    """Run one frozen node to its own termination; return the obs it leaves."""
    version = 0
    sv = build_state_view(env, version)
    ex = Executor(node)
    for _ in range(node.timeout):
        action = ex.act(obs, sv)
        if action is None:
            break
        obs, _, term, trunc, _ = env.step(action)
        version += 1
        sv = build_state_view(env, version)
        ex.observe_reward(0.0)
        if term or trunc:
            break
    return obs


def chain(*nodes):
    def prepare(env, obs):
        for n in nodes:
            obs = roll(n, env, obs)
        return obs
    return prepare


def f29_row(node, env_id, hook, prepare, label, gate):
    res = evaluate_node(node, execution_env_factory(env_id, hook),
                        seeds=heldout_seeds(N_EVAL), prepare=prepare)
    print(f"  {node.id:<24}{env_id.replace('MiniGrid-', ''):<26}{label:<52}"
          f"{res.success_rate:>6.2f}{res.wilson_lower:>7.3f}{res.mean_length:>7.1f}"
          f"   {'GATE' if gate else '-'}")
    return {"node": node.id, "env": env_id, "hook": hook.spec, "start_state": label,
            "gate": gate, **res.as_dict()}


# --------------------------------------------------------------------------
# V1-V7
# --------------------------------------------------------------------------

def facts() -> dict:
    out = {}
    seeds = range(N_FACTS)

    # V1 grid and budget
    v1 = {}
    for env_id in (UNLOCK, UNLOCKPICKUP, BLOCKED):
        env = gym.make(env_id)
        env.reset(seed=0)
        u = env.unwrapped
        v1[env_id] = {"width": u.width, "height": u.height, "max_steps": u.max_steps}
        env.close()
    out["V1"] = v1

    # V2 UnlockPickup = Unlock + a box in the right room; success = pickup of that box
    from minigrid.envs.unlockpickup import UnlockPickupEnv
    step_src = inspect.getsource(UnlockPickupEnv.step)
    gen_src = inspect.getsource(UnlockPickupEnv._gen_grid)
    env = gym.make(UNLOCKPICKUP)
    box_right = 0
    for s in seeds:
        env.reset(seed=s)
        u = env.unwrapped
        bp, _ = _find(u, "box")
        dp, _ = _find(u, "door")
        box_right += int(bp is not None and bp[0] > dp[0])
    env.close()
    out["V2"] = {
        "step_checks_carrying_is_obj": "self.carrying" in step_src and "self.obj" in step_src,
        "gen_grid_adds_box": "box" in gen_src.lower(),
        "box_right_of_door": f"{box_right}/{N_FACTS}",
    }

    # V3 box in view at reset (UnlockPickup)
    env = gym.make(UNLOCKPICKUP)
    n = 0
    for s in seeds:
        env.reset(seed=s)
        n += _in_view(env.unwrapped, "box")
    env.close()
    out["V3"] = {"box_in_view_at_reset": f"{n}/{N_FACTS}"}

    # V4 at the handoff the key is held; pickup with a full hand is refused;
    # drop-then-pickup succeeds and terminates the env with reward.
    env = gym.make(UNLOCKPICKUP)
    refused = tried = dp_ok = dp_tried = 0
    for s in seeds:
        env.reset(seed=s)
        give_key()(env)
        open_door()(env)
        u = env.unwrapped
        bp, _ = _find(u, "box")
        stands = [c for c in _neigh(bp) if u.grid.get(*c) is None]
        if not stands:
            continue
        c = stands[0]
        u.agent_pos = np.array(c)
        u.agent_dir = [d for d, v in DIRS.items() if (c[0] + v[0], c[1] + v[1]) == bp][0]
        tried += 1
        env.step(PICKUP)
        refused += int(u.carrying is not None and u.carrying.type == "key"
                       and _find(u, "box")[0] == bp)
        # turn to a free cell, drop, turn back, pick up
        for turn, back in ((LEFT, RIGHT), (RIGHT, LEFT)):
            env.step(turn)
            fx, fy = (int(v) for v in u.front_pos)
            if u.grid.get(fx, fy) is None:
                dp_tried += 1
                env.step(DROP)
                env.step(back)
                _, r, term, _, _ = env.step(PICKUP)
                dp_ok += int(term and r > 0 and u.carrying is not None
                             and u.carrying.type == "box")
                break
            env.step(back)
    env.close()
    out["V4"] = {"pickup_refused_with_key_in_hand": f"{refused}/{tried}",
                 "drop_then_pickup_succeeds_and_terminates": f"{dp_ok}/{dp_tried}"}

    # V5 at the handoff (door open, on the door-front cell, facing the door) the box is in view
    env = gym.make(UNLOCKPICKUP)
    n = 0
    for s in seeds:
        env.reset(seed=s)
        give_key()(env)
        open_door()(env)
        u = env.unwrapped
        dp, _ = _find(u, "door")
        u.agent_pos = np.array((dp[0] - 1, dp[1]))
        u.agent_dir = 0
        n += _in_view(u, "box")
    env.close()
    out["V5"] = {"box_in_view_at_handoff": f"{n}/{N_FACTS}"}

    # V6 BlockedUnlockPickup
    env = gym.make(BLOCKED)
    on_front = front_reach = key_reach = ball_view = 0
    for s in seeds:
        env.reset(seed=s)
        u = env.unwrapped
        dp, _ = _find(u, "door")
        bp, _ = _find(u, "ball")
        kp, _ = _find(u, "key")
        front = (dp[0] - 1, dp[1])
        on_front += int(bp == front)
        start = (int(u.agent_pos[0]), int(u.agent_pos[1]))
        reach = set(reachable_cells(u, start)) | {start}
        front_reach += int(front in reach)
        key_reach += int(any(c in reach for c in _neigh(kp)))
        ball_view += _in_view(u, "ball")
    env.close()
    out["V6"] = {"ball_on_door_front_cell": f"{on_front}/{N_FACTS}",
                 "door_front_reachable_without_moving_ball": f"{front_reach}/{N_FACTS}",
                 "key_reachable_without_moving_ball": f"{key_reach}/{N_FACTS}",
                 "ball_in_view_at_reset": f"{ball_view}/{N_FACTS}"}

    # V7 no frozen node was trained on a box or ball env
    trained_on = {}
    for meta in sorted(RUNS.glob("*/meta.json")):
        m = json.loads(meta.read_text(encoding="utf-8"))
        trained_on[meta.parent.name] = m.get("env_ids") or m.get("spec", {}).get("env_ids")
    box_ball = [k for k, v in trained_on.items()
                if v and any(e in (UNLOCKPICKUP, BLOCKED) for e in v)]
    out["V7"] = {"frozen_nodes_env_ids": trained_on,
                 "nodes_trained_on_a_box_or_ball_env": box_ball}

    for k, v in out.items():
        if k != "V7":
            print(f"  {k}: {v}")
    print(f"  V7: nodes trained on a box/ball env: {box_ball or 'none'}")
    return out


# --------------------------------------------------------------------------
# solvability audits
# --------------------------------------------------------------------------

AUDIT_SEEDS = (list(range(N_FACTS)) + stopping_seeds(N_EVAL) + heldout_seeds(N_EVAL))
MAX_SEARCH_STATES = 400_000


def solvable(u, goal: str) -> bool | None:
    """
    Exhaustive breadth-first search over MiniGrid's own movement rules, from the
    current state, for a state that satisfies `goal` ("box": carrying the box;
    "move_blocker": no ball next to the door and an empty hand; "empty_hand":
    carrying nothing, `{0,9} drop_held`'s goal).

    [H-7.1 deviation] The plan's audit was a flood fill: "a cell adjacent to the
    box is reachable". That calls a layout unsolvable when an object a policy
    can simply pick up stands in the way -- the key lying on the open door's
    only approach cell, or an agent boxed in by the ball it is meant to move.
    Those episodes are solvable, and H-6.6's lesson is about episodes NO policy
    can solve. The flood-fill numbers are kept beside this one.

    Rules mirrored from `MiniGridEnv.step` (minigrid 3.1.0): forward onto an
    empty cell or an open door; pickup a key/ball/box with an empty hand; drop
    onto an empty cell; toggle a door (locked -> open with the matching key,
    otherwise open <-> closed). Toggling a box destroys it and is never useful
    for these goals, so it is not expanded. Returns None if the search is cut
    off at MAX_SEARCH_STATES.
    """
    from collections import deque

    walls = set()
    objs = []
    door = None
    for x in range(u.width):
        for y in range(u.height):
            c = u.grid.get(x, y)
            if c is None:
                continue
            if c.type == "wall":
                walls.add((x, y))
            elif c.type == "door":
                door = ((x, y), c.color, "open" if c.is_open else "locked" if c.is_locked else "closed")
            elif c.type in ("key", "ball", "box"):
                objs.append((x, y, c.type, c.color))
    carry = (u.carrying.type, u.carrying.color) if u.carrying is not None else None
    dpos, dcol, dstate = door
    start = (int(u.agent_pos[0]), int(u.agent_pos[1]), int(u.agent_dir), carry,
             tuple(sorted(objs)), dstate)

    def is_goal(st):
        _, _, _, cy, ob, _ = st
        if goal == "box":
            return cy is not None and cy[0] == "box"
        if goal == "empty_hand":
            return cy is None
        return cy is None and not any(t == "ball" and _adj((x, y), dpos) for x, y, t, _ in ob)

    seen = {start}
    q = deque([start])
    while q:
        st = q.popleft()
        if is_goal(st):
            return True
        if len(seen) > MAX_SEARCH_STATES:
            return None
        ax, ay, d, cy, ob, ds = st
        fx, fy = ax + DIRS[d][0], ay + DIRS[d][1]
        at = {(x, y): (t, c) for x, y, t, c in ob}
        nxt = [(ax, ay, (d - 1) % 4, cy, ob, ds), (ax, ay, (d + 1) % 4, cy, ob, ds)]
        front_is_door = (fx, fy) == dpos
        front_free = (fx, fy) not in walls and (fx, fy) not in at and not front_is_door
        if front_free or (front_is_door and ds == "open"):
            nxt.append((fx, fy, d, cy, ob, ds))
        if cy is None and (fx, fy) in at:
            t, c = at[(fx, fy)]
            nxt.append((ax, ay, d, (t, c), tuple(o for o in ob if (o[0], o[1]) != (fx, fy)), ds))
        if cy is not None and front_free:
            nxt.append((ax, ay, d, None, tuple(sorted(ob + ((fx, fy, cy[0], cy[1]),))), ds))
        if front_is_door:
            if ds == "locked":
                if cy == ("key", dcol):
                    nxt.append((ax, ay, d, cy, ob, "open"))
            else:
                nxt.append((ax, ay, d, cy, ob, "closed" if ds == "open" else "open"))
        for n in nxt:
            if n not in seen:
                seen.add(n)
                q.append(n)
    return False


def _audit(env_id, hook, goal, proxy):
    env = ResetHookEnv(gym.make(env_id), hook)
    bad, cut, proxy_bad, held = [], [], [], 0
    for s in AUDIT_SEEDS:
        env.reset(seed=s)
        u = env.unwrapped
        held += u.carrying is not None
        r = solvable(u, goal)
        if r is False:
            bad.append(s)
        elif r is None:
            cut.append(s)
        if not proxy(u):
            proxy_bad.append(s)
    env.close()
    return {"env": env_id, "hook": hook.spec, "goal": goal, "n": len(AUDIT_SEEDS),
            "key_in_hand": held,
            "unsolvable": len(bad), "unsolvable_seeds": bad[:20],
            "search_cut_off": len(cut),
            "flood_fill_proxy_fails": len(proxy_bad),
            "flood_fill_proxy_fail_seeds": proxy_bad[:20]}


def _proxy_pickup_box(u):
    start = (int(u.agent_pos[0]), int(u.agent_pos[1]))
    reach = set(reachable_cells(u, start)) | {start}
    bp, _ = _find(u, "box")
    return bp is not None and any(c in reach for c in _neigh(bp))


def _proxy_move_blocker(u):
    start = (int(u.agent_pos[0]), int(u.agent_pos[1]))
    reach = set(reachable_cells(u, start)) | {start}
    bp, _ = _find(u, "ball")
    dp, _ = _find(u, "door")
    holding = u.carrying is not None
    drop = [c for c in reach if u.grid.get(*c) is None and c != start and not _adj(c, dp)]
    return (bp is not None and any(c in reach for c in _neigh(bp))
            and len(drop) >= (2 if holding else 1))


def _proxy_drop_held(u):
    start = (int(u.agent_pos[0]), int(u.agent_pos[1]))
    return any(u.grid.get(*c) is None and c != start for c in reachable_cells(u, start))


def audit_pickup_box(env_id) -> dict:
    """The withdrawn attempt-1 hook (H-7.1's audit; `--stage base`)."""
    return _audit(env_id, pickup_box_attempt1_spec()["hook"], "box", _proxy_pickup_box)


def audit_h7a() -> list[dict]:
    """[H-7a.1] The two amendment-1 reset distributions, on UnlockPickup."""
    sp = h7_primitive_specs()
    return [
        {"node": sp["drop_held"]["node_id"],
         **_audit(UNLOCKPICKUP, sp["drop_held"]["hook"], "empty_hand", _proxy_drop_held)},
        {"node": sp["pickup_box"]["node_id"],
         **_audit(UNLOCKPICKUP, sp["pickup_box"]["hook"], "box", _proxy_pickup_box)},
    ]


def audit_move_blocker() -> dict:
    return _audit(BLOCKED, h7_primitive_specs()["move_blocker"]["hook"], "move_blocker",
                  _proxy_move_blocker)


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

def stage_base(registry) -> dict:
    print("V1-V7, re-measured over seeds 0-399:")
    out = {"facts": facts()}
    print("\nsolvability audits (seeds 0-399 + both fixed sets):")
    audits = [audit_pickup_box(UNLOCKPICKUP), audit_pickup_box(BLOCKED), audit_move_blocker()]
    for a in audits:
        print(f"  {a['hook'][:60]:<62}{a['env'].replace('MiniGrid-', ''):<26}"
              f"unsolvable {a['unsolvable']}/{a['n']} (search cut off {a['search_cut_off']}; "
              f"flood-fill proxy fails {a['flood_fill_proxy_fails']})  key in hand {a['key_in_hand']}")
    out["solvability"] = audits
    print("\nF29 rows, 50 held-out seeds:")
    _, t3 = load_unlock_graph(registry)
    out["rows"] = [
        f29_row(t3, UNLOCKPICKUP, NO_HOOK, None, "episode start (A1)", True),
        f29_row(t3, BLOCKED, NO_HOOK, None, "[expected fail] episode start, ball on door-front", False),
    ]
    return out


def stage_audit() -> dict:
    print(f"solvability audits of the amendment-1 hooks ({len(AUDIT_SEEDS)} resets each):")
    audits = audit_h7a()
    for a in audits:
        print(f"  {a['node']:<20}{a['hook']:<56}unsolvable {a['unsolvable']}/{a['n']} "
              f"(search cut off {a['search_cut_off']}; flood-fill proxy fails "
              f"{a['flood_fill_proxy_fails']})  key in hand {a['key_in_hand']}")
    return {"solvability": audits}


def stage_pickup_box(registry) -> dict:
    """[H-7a.2] Amendment 1's three rows; replaces the attempt-1 rows of plan H-7.3."""
    _, t3 = load_unlock_graph(registry)
    dh = load_h7_primitive("drop_held", registry)
    pb = load_h7_primitive("pickup_box", registry)
    print("\nF29 rows on UnlockPickup, 50 held-out seeds:")
    rows = [
        f29_row(dh, UNLOCKPICKUP, NO_HOOK, chain(t3), "drop_held after unlock_door (A2a)", True),
        f29_row(pb, UNLOCKPICKUP, NO_HOOK, chain(t3, dh),
                "pickup_box after unlock_door, drop_held (A2b)", True),
    ]
    # Not gating: pickup_box handed the key, as a goal-only parent may do. What
    # it does with a full hand (V13: undefined is not the same as failing).
    full = run_full_hand(pb, UNLOCKPICKUP, NO_HOOK, prepare=chain(t3))
    label = "[not gating] pickup_box after unlock_door, key still in hand"
    print(f"  {pb.id:<24}{'UnlockPickup-v0':<26}{label:<52}{full['success_rate']:>6.2f}"
          f"{full['wilson_lower']:>7.3f}{full['mean_length']:>7.1f}   -   drop share "
          f"{full['drop_share']:.3f}, holding at end {full['holding_at_end']}")
    rows.append({"node": pb.id, "env": UNLOCKPICKUP, "hook": NO_HOOK.spec,
                 "start_state": label, "gate": False, **full})
    return {"rows": rows, "spec": "H-7 amendment 1"}


def stage_blocked(registry, args) -> dict:
    # [H-7 amendment 1, section 2] H-7b. Its rows depend on move_blocker and on a
    # T7 whose children have changed; it is re-planned after H-7a's result.
    raise SystemExit("--stage blocked: deferred to H-7b (H-7 amendment 1, section 2)")
    prims, t3 = load_unlock_graph(registry)
    mb_spec = None
    if args.move_blocker_fallback:
        from h7_nodes import move_blocker_fallback_spec
        mb_spec = move_blocker_fallback_spec()
    mb = load_h7_primitive("move_blocker", registry, spec=mb_spec)
    pb = load_h7_primitive("pickup_box", registry)
    up = build_unlock_pickup(t3, pb)
    from src.training.node_trainer import load_frozen_composite
    load_frozen_composite(up, registry, UNLOCKPICKUP, NO_HOOK)
    print(f"  loaded {up}  spec_hash={up.spec_hash()}")

    print("\nF29 rows on BlockedUnlockPickup, 50 held-out seeds:")
    gate_after_failed = not args.move_blocker_fallback
    rows = [
        f29_row(mb, BLOCKED, NO_HOOK, None, "move_blocker from episode start", True),
        f29_row(mb, BLOCKED, NO_HOOK, chain(t3), "move_blocker after a failed unlock_door",
                gate_after_failed),
        # A5, the measurement
        f29_row(prims["get_key"], BLOCKED, NO_HOOK, chain(mb), "A5: get_key after move_blocker", False),
        f29_row(prims["open_door"], BLOCKED, NO_HOOK, chain(mb, prims["get_key"]),
                "A5: open_door after move_blocker, get_key", False),
        f29_row(t3, BLOCKED, NO_HOOK, chain(mb), "A5: unlock_door after move_blocker", False),
        f29_row(up, BLOCKED, NO_HOOK, chain(mb), "A5: unlock_pickup after move_blocker", False),
        f29_row(pb, BLOCKED, NO_HOOK, chain(mb, t3), "pickup_box after move_blocker, unlock_door", False),
    ]
    a5 = {r["start_state"]: r["success_rate"] for r in rows if r["start_state"].startswith("A5")}
    ud = a5["A5: unlock_door after move_blocker"]
    upr = a5["A5: unlock_pickup after move_blocker"]
    branch = "transferred" if (ud >= GATE and upr >= GATE) else "rebuilt"
    print(f"\n  A5: unlock_door {ud:.2f}, unlock_pickup {upr:.2f} -> reuse: {branch}")
    return {"rows": rows, "a5": {"unlock_door": ud, "unlock_pickup": upr,
                                 "threshold": GATE, "reuse": branch},
            "move_blocker_hook": mb.train_hook}


def run_full_hand(node, env_id, hook, prepare=None):
    env = execution_env_factory(env_id, hook)()
    acts, end_hold = Counter(), Counter()
    ok = lengths = at_start = 0
    try:
        for seed in heldout_seeds(N_EVAL):
            obs, _ = env.reset(seed=seed)
            if prepare is not None:
                obs = prepare(env, obs)
            version = 0
            sv = build_state_view(env, version)
            at_start += int(bool(node.goal(sv)))
            ex = Executor(node)
            steps = 0
            for _ in range(node.timeout):
                a = ex.act(obs, sv)
                if a is None:
                    break
                acts[int(a)] += 1
                obs, _, term, trunc, _ = env.step(a)
                steps += 1
                version += 1
                sv = build_state_view(env, version)
                ex.observe_reward(0.0)
                if term or trunc:
                    break
            ok += int(bool(node.goal(sv)))
            lengths += steps
            end_hold[sv.carrying_type or "nothing"] += 1
    finally:
        env.close()
    total = sum(acts.values())
    return {
        "n": N_EVAL, "successes": ok, "success_rate": round(ok / N_EVAL, 4),
        "wilson_lower": round(wilson_lower(ok, N_EVAL), 4),
        "goal_true_at_start": at_start,
        "mean_length": round(lengths / N_EVAL, 2),
        "drop_share": round(acts[DROP] / total, 4) if total else 0.0,
        "action_share": {ACTION_NAMES[k]: round(acts[k] / total, 4) for k in sorted(acts)} if total else {},
        "holding_at_end": dict(end_hold),
    }


def stage_full_hand(registry) -> dict:
    prims, t3 = load_unlock_graph(registry)
    children = {"get_key": prims["get_key"], "open_door": prims["open_door"], "unlock_door": t3}
    skipped = []
    for name in ("drop_held", "pickup_box", "move_blocker"):
        try:
            children[name] = load_h7_primitive(name, registry)
        except Exception as exc:                       # not trained yet
            skipped.append(f"{name}: {type(exc).__name__}")
    if "pickup_box" in children and "drop_held" in children:
        try:
            from src.training.node_trainer import load_frozen_composite
            up = build_unlock_pickup(t3, children["drop_held"], children["pickup_box"])
            load_frozen_composite(up, registry, UNLOCKPICKUP, NO_HOOK)
            children["unlock_pickup"] = up
        except Exception as exc:
            skipped.append(f"unlock_pickup: {type(exc).__name__}")

    # The state each child is normally handed, before the hand is changed:
    # pickup_box after the door is open; everything else at episode start, with
    # the blocker already moved on BlockedUnlockPickup (the state move_blocker
    # leaves); move_blocker itself with the ball still on the door-front cell.
    def base(name, env_id):
        if name == "pickup_box":
            return [relocate_blocker(), open_door()]
        if name == "drop_held":
            return [relocate_blocker(), open_door(), randomize_agent_pose(anchor=None)]
        if name == "move_blocker":
            return []
        return [relocate_blocker()] if env_id == BLOCKED else []

    hands = {"nothing": [], "key": [give_key()], "ball": [hold("ball")]}
    rows = []
    print(f"\n{'child':<24}{'env':<22}{'holding':<9}{'SR':>6}{'drop':>7}{'at0':>6}   holding at end")
    for name, node in children.items():
        envs = [BLOCKED] if name == "move_blocker" else [UNLOCKPICKUP, BLOCKED]
        for env_id in envs:
            for hand, extra in hands.items():
                if hand == "ball" and env_id != BLOCKED:
                    continue
                hook = compose(*base(name, env_id), *extra)
                r = run_full_hand(node, env_id, hook)
                rows.append({"child": node.id, "env": env_id, "holding": hand,
                             "hook": hook.spec, **r})
                print(f"{node.id:<24}{env_id.replace('MiniGrid-', ''):<22}{hand:<9}"
                      f"{r['success_rate']:>6.2f}{r['drop_share']:>7.3f}"
                      f"{r['goal_true_at_start']:>6}   {r['holding_at_end']}")
    return {"rows": rows, "skipped": skipped}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["base", "audit", "pickup_box", "blocked", "full_hand"])
    ap.add_argument("--move-blocker-fallback", action="store_true",
                    help="load move_blocker trained with the empty-handed fallback hook")
    args = ap.parse_args()

    print(describe())
    registry = NodeRegistry(RUNS)
    doc = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    doc.update({"n_eval": N_EVAL, "seed_base": heldout_seeds(1)[0], "gate_threshold": GATE})

    if args.stage == "base":
        doc["base"] = stage_base(registry)
    elif args.stage == "audit":
        doc["h7a_audit"] = stage_audit()
    elif args.stage == "pickup_box":
        doc["pickup_box"] = stage_pickup_box(registry)
    elif args.stage == "blocked":
        doc["blocked"] = stage_blocked(registry, args)
    else:
        doc["full_hand"] = stage_full_hand(registry)

    gating = [r for st in ("base", "pickup_box", "blocked") if st in doc
              for r in doc[st].get("rows", []) if r["gate"]]
    # base: the attempt-1 and move_blocker hooks (H-7.1); h7a_audit: amendment 1.
    unsolvable = sum(a["unsolvable"] for st in ("base", "h7a_audit")
                     for a in doc.get(st, {}).get("solvability", []))
    doc["gate_passed"] = bool(gating) and all(r["success_rate"] >= GATE for r in gating) \
        and unsolvable == 0
    doc["gating_rows"] = [{"node": r["node"], "env": r["env"], "start_state": r["start_state"],
                           "success_rate": r["success_rate"]} for r in gating]
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}  gate_passed={doc['gate_passed']}")
    failed = [r for r in gating if r["success_rate"] < GATE]
    for r in failed:
        print(f"  GATE ROW BELOW {GATE}: {r['node']} on {r['env']} '{r['start_state']}' "
              f"SR {r['success_rate']:.2f}")
    if unsolvable:
        print(f"  UNSOLVABLE LAYOUTS: {unsolvable}")
    return 0 if doc["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
