"""
Execution-mode audit (`docs/plans/2026-10-02_execution_mode_audit.md`): what
deterministic and sampled execution do across the library. Evaluation only.

Nothing is trained, continued or frozen, and no checkpoint is written. v2, the
Executor, the stabilisation rule and every default are untouched. A stochastic
success rate below is not a gate and freezes nothing.

    E.0  which levels sample under STOCHASTIC: every policy of a real two-level
         graph wrapped with a recorder, one episode per mode
    E.1  the 9 FROZEN nodes on each of their env_ids, under their own hooks
    E.2  every gating F29 row under runs/v2/, re-run; plus, diagnostic, the
         scripted sequence unlock_door -> drop_held -> pickup_box (TRAINING
         checkpoint) on UnlockPickup from the env's own reset, three variants
    E.3  the twelve flat PPO runs: both modes where the model was saved, the
         training record where it was not
    E.4  runs/v2/execution_mode_audit.json, with the two flags fixed in advance

Seeds: 40000-40049 (unused by any script or result file before this audit).
Every row: 50 deterministic episodes and 3 x 50 stochastic ones with sampling
seeds 0, 1, 2. A stochastic episode sets torch.manual_seed(1_000_003 *
sampling_seed + env_seed) once the node under test is about to start (H-7a.5's
definition). Nodes that only build the start state (a `prepare` chain) run
deterministically in both modes, as frozen nodes do by default.

A state repeat is H-7a.5's: the full grid encoding, the agent's pose and what it
carries recur within one node's own episode, checked in deterministic episodes
only. For a primitive that proves failure (memoryless policy, deterministic
actions) and the script raises if a repeat is followed by a success. For a
composite it does not: the Executor's stack and step counters are state the
grid does not show, so repeats followed by a success are counted, not raised.

    python scripts/audit_execution_mode.py            ->  runs/v2/execution_mode_audit.json
    python scripts/audit_execution_mode.py --smoke    ->  5 seeds, nothing written under runs/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import torch  # noqa: E402

from src.envs.subgoal_env import NO_HOOK, two_keys  # noqa: E402
from src.skills.executor import DETERMINISTIC, STOCHASTIC, Executor  # noqa: E402
from src.skills.registry import NodeRegistry  # noqa: E402
from src.skills.state import build_state_view  # noqa: E402
from src.training.evaluation import (  # noqa: E402
    evaluate_node,
    fixed_seeds,
    heldout_seeds,
    run_episode,
    wilson_lower,
    wilson_upper,
)
from src.training.node_trainer import (  # noqa: E402
    attach_sb3_primitive_policy,
    execution_env_factory,
    load_frozen_composite,
    load_frozen_primitive,
    set_torch_threads,
    subgoal_env_factory,
)
from src.types import NodeStatus  # noqa: E402
from diagnose_pickup_box import _full_state  # noqa: E402   H-7a.5's state, reused
from eval_h7_preflight import chain  # noqa: E402
from h7_nodes import (  # noqa: E402
    UNLOCKPICKUP,
    h7_primitive_specs,
    load_h7_primitive,
)
from train_choice_task import COLORS, choice_specs, open_door_tk_spec  # noqa: E402
from train_flat_baseline import BASELINE_RUNS, TASKS, build_flat_ppo  # noqa: E402
from train_primitives import DOORKEY, UNLOCK, build_node, specs  # noqa: E402
from train_solve_doorkey import build_t5  # noqa: E402
from train_unlock_door import build_t3  # noqa: E402

RUNS = REPO / "runs" / "v2"
SEED_BASE = 40_000
N = 50
SAMPLING_SEEDS = (0, 1, 2)
MODE_SENSITIVE = 0.10          # fixed by the audit file before the run
FLAT_SEEDS = (42, 43, 44)
FLAT_TASKS = ("doorkey", "unlock", "unlockpickup", "blocked")


def torch_seed(sampling_seed: int, env_seed: int) -> int:
    return 1_000_003 * int(sampling_seed) + int(env_seed)


class Excluded(Exception):
    """The start state this row is defined on was not reached (choice rows only)."""


# --------------------------------------------------------------------------
# one episode, with H-7a.5's bookkeeping
# --------------------------------------------------------------------------

def episode(node, env, seed, *, mode, sampling_seed=None, prepare=None) -> dict:
    """As `evaluation.run_episode` runs it: same Executor, same step limit."""
    obs, _ = env.reset(seed=seed)
    if prepare is not None:
        obs = prepare(env, obs)             # raises Excluded where the row says so
    if sampling_seed is not None:
        torch.manual_seed(torch_seed(sampling_seed, seed))
    u = env.unwrapped
    version = 0
    sv = build_state_view(env, version)
    ex = Executor(node, execution_mode=mode)
    seen = {_full_state(u): 0}
    repeat_step = period = None
    steps = 0
    while steps < node.timeout:
        action = ex.act(obs, sv)
        if action is None:
            break
        obs, _, terminated, truncated, _ = env.step(action)
        steps += 1
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
    if repeat_step is not None and success and node.is_primitive:
        raise RuntimeError(f"{node.id} seed {seed}: a deterministic state repeat, "
                           "yet the episode succeeded")
    return {"seed": seed, "sampling_seed": sampling_seed, "success": success, "steps": steps,
            "repeat_step": repeat_step, "repeat_period": period}


def run_row(node, env_factory, seeds, *, prepare=None) -> dict:
    """50 deterministic + 3 x 50 stochastic episodes of one row."""
    env = env_factory()
    det, sto, excluded = [], [], []
    try:
        for s in seeds:
            try:
                det.append(episode(node, env, s, mode=DETERMINISTIC, prepare=prepare))
            except Excluded:
                excluded.append(s)
        for k in SAMPLING_SEEDS:
            for s in seeds:
                if s in excluded:
                    continue
                sto.append(episode(node, env, s, mode=STOCHASTIC, sampling_seed=k,
                                   prepare=prepare))
    finally:
        env.close()
    return summarise(node, det, sto, excluded)


def _rate(records) -> dict:
    n = len(records)
    k = sum(r["success"] for r in records)
    ok_steps = [r["steps"] for r in records if r["success"]]
    return {"n": n, "successes": k, "success_rate": round(k / n, 4) if n else None,
            "wilson_lower": round(wilson_lower(k, n), 4),
            "wilson_upper": round(wilson_upper(k, n), 4),
            "mean_steps_successes": round(sum(ok_steps) / len(ok_steps), 2) if ok_steps else None}


def mode_sensitive(det: dict, sto: dict) -> bool:
    """|pooled stochastic - deterministic| >= 0.10, in exact integer arithmetic."""
    kd, nd, ks, ns = det["successes"], det["n"], sto["successes"], sto["n"]
    if not nd or not ns:
        return False
    return abs(ks * nd - kd * ns) * 10 >= nd * ns


def summarise(node, det, sto, excluded) -> dict:
    d, s = _rate(det), _rate(sto)
    fails = [r for r in det if not r["success"]]
    periods: dict[str, int] = {}
    for r in fails:
        if r["repeat_period"] is not None:
            periods[str(r["repeat_period"])] = periods.get(str(r["repeat_period"]), 0) + 1
    d.update({
        "failures": len(fails),
        "failures_with_state_repeat": sum(r["repeat_step"] is not None for r in fails),
        "failure_repeat_periods": dict(sorted(periods.items(), key=lambda kv: int(kv[0]))),
        "successes_with_state_repeat": sum(r["repeat_step"] is not None
                                           for r in det if r["success"]),
    })
    s["by_sampling_seed"] = {str(k): sum(r["success"] for r in sto if r["sampling_seed"] == k)
                             for k in SAMPLING_SEEDS}
    out = {"deterministic": d, "stochastic": s,
           "delta_stochastic_minus_deterministic": (
               round(s["success_rate"] - d["success_rate"], 4)
               if d["n"] and s["n"] else None),
           "mode_sensitive": mode_sensitive(d, s),
           "episodes": [[r["seed"], r["sampling_seed"], int(r["success"]), r["steps"],
                         r["repeat_step"], r["repeat_period"]] for r in det + sto]}
    if excluded:
        out["excluded_seeds"] = excluded
    return out


def print_row(label: str, row: dict) -> None:
    d, s = row["deterministic"], row["stochastic"]
    if not d["n"]:
        print(f"  {label:<70} no episodes")
        return
    print(f"  {label:<70} det {d['successes']:>2}/{d['n']:<3} sto {s['successes']:>3}/{s['n']:<3}"
          f" {s['by_sampling_seed']}  rep {d['failures_with_state_repeat']}/{d['failures']}"
          f"{'  MODE-SENSITIVE' if row['mode_sensitive'] else ''}", flush=True)


# --------------------------------------------------------------------------
# cross-checks against the standard harness
# --------------------------------------------------------------------------

def crosscheck(node, env_factory, seeds, row, *, prepare=None) -> dict:
    """
    The deterministic episodes must match `evaluate_node` episode for episode,
    and the stochastic ones of sampling seed 0 must match `run_episode` under the
    same torch seed. Raises on any difference.
    """
    det_seeds = [s for s in seeds if s not in row.get("excluded_seeds", [])]
    ref = evaluate_node(node, env_factory, seeds=det_seeds, prepare=prepare)
    mine = [bool(e[2]) for e in row["episodes"] if e[1] is None]
    if ref.per_episode != mine or ref.lengths != [e[3] for e in row["episodes"] if e[1] is None]:
        raise RuntimeError(f"{node.id}: deterministic episodes differ from evaluate_node")
    env = env_factory()
    try:
        ref0 = []
        for s in det_seeds:
            torch.manual_seed(torch_seed(0, s))   # prepare and reset consume no torch RNG
            ok, steps, _ = run_episode(node, env, s, execution_mode=STOCHASTIC, prepare=prepare)
            ref0.append((ok, steps))
    finally:
        env.close()
    mine0 = [(bool(e[2]), e[3]) for e in row["episodes"] if e[1] == 0]
    if ref0 != mine0:
        raise RuntimeError(f"{node.id}: stochastic episodes (sampling seed 0) differ from run_episode")
    return {"deterministic_vs_evaluate_node": "identical",
            "stochastic_seed0_vs_run_episode": "identical", "episodes": len(det_seeds)}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def primitive_specs() -> dict:
    """The primitive specs eval_regression.py loads, in its order."""
    all_specs = dict(specs())
    all_specs.update(choice_specs())
    all_specs["open_door_tk"] = open_door_tk_spec()
    h7 = h7_primitive_specs()
    all_specs["drop_held"] = h7["drop_held"]
    return all_specs


def load_library(registry):
    prims = {}
    for name, spec in primitive_specs().items():
        node = build_node(spec)
        load_frozen_primitive(node, registry, spec["env_ids"][0], spec["hook"])
        prims[name] = (node, spec)
    t3 = build_t3([prims["get_key"][0], prims["open_door"][0]])
    load_frozen_composite(t3, registry, UNLOCK, NO_HOOK)
    t5 = build_t5(t3, prims["go_to_goal"][0])
    load_frozen_composite(t5, registry, DOORKEY, NO_HOOK)
    for n in [p for p, _ in prims.values()] + [t3, t5]:
        if n.status is not NodeStatus.FROZEN:
            raise RuntimeError(f"{n.id} did not load as FROZEN")
    return prims, t3, t5


def load_pickup_box_training(registry):
    """The spec-2 seed-42 checkpoint, TRAINING and never frozen (as H-7a.5 loads it)."""
    spec = h7_primitive_specs()["pickup_box"]
    node = build_node(spec)
    attach_sb3_primitive_policy(node, UNLOCKPICKUP, spec["hook"])
    registry.load_into(node, require_frozen=False)
    if node.status is not NodeStatus.TRAINING:
        raise RuntimeError(f"{node.id}: expected TRAINING, found {node.status.value}")
    return node


# --------------------------------------------------------------------------
# E.0
# --------------------------------------------------------------------------

class _Recorder:
    def __init__(self, inner, log, node_id):
        self.inner, self.log, self.node_id = inner, log, node_id

    def act(self, obs, deterministic=True):
        self.log.setdefault(self.node_id, []).append(bool(deterministic))
        return self.inner.act(obs, deterministic=deterministic)

    def select(self, obs, mask, deterministic=True):
        self.log.setdefault(self.node_id, []).append(bool(deterministic))
        return self.inner.select(obs, mask, deterministic=deterministic)


def e0(t5, seeds) -> dict:
    """
    Wrap every policy in `{2,1} solve_doorkey`'s graph and record the flag each
    receives. Episodes run seed by seed until every node has been called.
    """
    nodes = [t5, *t5.descendants()]
    saved = {id(n): n.policy for n in nodes}
    out = {}
    try:
        for mode in (DETERMINISTIC, STOCHASTIC):
            log: dict[str, list[bool]] = {}
            for n in nodes:
                n.policy = _Recorder(saved[id(n)], log, n.id)
            env = execution_env_factory(DOORKEY, NO_HOOK)()
            used = []
            try:
                for seed in seeds:
                    if mode == STOCHASTIC:
                        torch.manual_seed(torch_seed(0, seed))
                    run_episode(t5, env, seed, execution_mode=mode)
                    used.append(seed)
                    if len(log) == len(nodes):
                        break
            finally:
                env.close()
            out[mode] = {
                "episodes": len(used), "seeds": used,
                "nodes_called": len(log), "nodes_in_graph": len(nodes),
                "calls": {nid: {"n": len(v), "deterministic_flags": sorted(set(v))}
                          for nid, v in sorted(log.items())},
            }
    finally:
        for n in nodes:
            n.policy = saved[id(n)]
    return out


# --------------------------------------------------------------------------
# E.2's scripted sequence
# --------------------------------------------------------------------------

def sequence_episode(nodes_modes, env, seed, sampling_seed, goal) -> dict:
    """
    unlock_door -> drop_held -> pickup_box, each run to its own end, as `chain`
    rolls them, from the env's own reset. Stops when the env terminates or
    truncates (UnlockPickup: 288 steps).
    """
    obs, _ = env.reset(seed=seed)
    if sampling_seed is not None:
        torch.manual_seed(torch_seed(sampling_seed, seed))
    u = env.unwrapped
    segs, total, env_done = [], 0, False
    sv = build_state_view(env, 0)
    for node, mode in nodes_modes:
        version = 0
        sv = build_state_view(env, version)
        ex = Executor(node, execution_mode=mode)
        seen = {_full_state(u): 0}
        rep = per = None
        steps = 0
        while steps < node.timeout:
            a = ex.act(obs, sv)
            if a is None:
                break
            obs, _, term, trunc, _ = env.step(a)
            steps += 1
            if mode == DETERMINISTIC and rep is None:
                s = _full_state(u)
                if s in seen:
                    rep, per = steps, steps - seen[s]
                else:
                    seen[s] = steps
            version += 1
            sv = build_state_view(env, version)
            ex.observe_reward(0.0)
            if term or trunc:
                env_done = True
                break
        total += steps
        segs.append({"node": node.id, "steps": steps, "goal": bool(node.goal(sv)),
                     "repeat_step": rep, "repeat_period": per})
        if env_done:
            break
    success = bool(goal(sv))
    failed_in = None if success else next(
        (g["node"] for g in segs if not g["goal"]), "env ended before pickup_box")
    return {"seed": seed, "sampling_seed": sampling_seed, "success": success, "steps": total,
            "failed_in": failed_in, "segments": segs}


def run_sequence(variant, nodes_modes_det, nodes_modes_sto, seeds, goal) -> dict:
    env = execution_env_factory(UNLOCKPICKUP, NO_HOOK)()
    try:
        det = [sequence_episode(nodes_modes_det, env, s, None, goal) for s in seeds]
        sto = [sequence_episode(nodes_modes_sto, env, s, k, goal)
               for k in SAMPLING_SEEDS for s in seeds]
    finally:
        env.close()
    d, s = _rate(det), _rate(sto)
    fails = [r for r in det if not r["success"]]
    where: dict[str, int] = {}
    for r in fails:
        where[r["failed_in"]] = where.get(r["failed_in"], 0) + 1
    d["failures"] = len(fails)
    d["failures_by_segment"] = where
    d["failures_with_state_repeat_in_failing_segment"] = sum(
        any(g["node"] == r["failed_in"] and g["repeat_step"] is not None for g in r["segments"])
        for r in fails)
    s["by_sampling_seed"] = {str(k): sum(r["success"] for r in sto if r["sampling_seed"] == k)
                             for k in SAMPLING_SEEDS}
    s["failures_by_segment"] = {}
    for r in sto:
        if not r["success"]:
            s["failures_by_segment"][r["failed_in"]] = s["failures_by_segment"].get(r["failed_in"], 0) + 1
    return {"variant": variant, "deterministic": d, "stochastic": s,
            "delta_stochastic_minus_deterministic": round(s["success_rate"] - d["success_rate"], 4),
            "mode_sensitive": mode_sensitive(d, s),
            "episodes": [[r["seed"], r["sampling_seed"], int(r["success"]), r["steps"],
                          r["failed_in"]] for r in det + sto]}


# --------------------------------------------------------------------------
# E.3
# --------------------------------------------------------------------------

def flat_saved_models() -> dict:
    """
    (task, seed) -> (run_dir or git blob source). The baseline registry keeps one
    directory per node id, so later seeds overwrote earlier ones; git history
    keeps the overwritten DoorKey model of seed 44 (commit ecc9188).
    """
    found = {}
    for task in FLAT_TASKS:
        node_dir = BASELINE_RUNS / f"0-B_flat_{TASKS[task]['name']}_ppo"
        rel = node_dir.relative_to(REPO).as_posix()
        commits = subprocess.run(
            ["git", "log", "--all", "--format=%h", "--", f"{rel}/policy_state_dict.pt"],
            cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
        on_disk = None
        if (node_dir / "meta.json").exists():
            on_disk = json.loads((node_dir / "meta.json").read_text(encoding="utf-8"))["seed"]
            found[(task, on_disk)] = {"source": f"{rel}/ (working tree)", "dir": node_dir}
        for c in commits:
            meta = json.loads(subprocess.run(
                ["git", "show", f"{c}:{rel}/meta.json"], cwd=REPO, capture_output=True,
                text=True, check=True).stdout)
            key = (task, int(meta["seed"]))
            if key not in found:
                found[key] = {"source": f"git {c}:{rel}/", "commit": c, "rel": rel}
    return found


def materialise(entry, tmp: Path) -> Path:
    if "dir" in entry:
        return entry["dir"]
    d = tmp / entry["commit"]
    d.mkdir(parents=True, exist_ok=True)
    for name in ("meta.json", "policy_state_dict.pt"):
        blob = subprocess.run(["git", "show", f"{entry['commit']}:{entry['rel']}/{name}"],
                              cwd=REPO, capture_output=True, check=True).stdout
        (d / name).write_bytes(blob)
    return d


def flat_record(task: str, seed: int) -> dict:
    """What the training run recorded, for a model that was not saved."""
    name = "flat_ppo" if task == "doorkey" else f"flat_{task}_ppo"
    path = RUNS / f"{name}_s{seed}.json"
    rep = json.loads(path.read_text(encoding="utf-8"))["report"]
    hist = rep["criterion"]["history"]
    best = max(hist, key=lambda h: h["success_rate"])
    log = {"doorkey": f"h64_flat_ppo_s{seed}.log"}.get(task, f"h72_flat_{task}_s{seed}.log")
    return {
        "source": path.relative_to(REPO).as_posix(), "log": f"runs/v2/{log}",
        "stable": rep["stable"], "steps_trained": rep["steps_trained"],
        "evaluations": len(hist),
        "best_deterministic_eval_success_rate": best["success_rate"],
        "best_eval_at_evaluation": hist.index(best) + 1,
        "heldout_deterministic": rep["heldout"]["success_rate"],
        "rollout_success_or_reward_in_log": None,
        "note": ("the training log records no rollout success rate or mean episode reward "
                 "(PPO ran with verbose=0); the only numbers it holds are the deterministic "
                 "evaluations on the stopping seeds and the deterministic held-out score"),
    }


def e3(registry_unused, seeds, smoke) -> dict:
    saved = flat_saved_models()
    rows = {}
    with tempfile.TemporaryDirectory() as tmp:
        for task in FLAT_TASKS:
            for seed in FLAT_SEEDS:
                key = f"{task}_s{seed}"
                env_id = TASKS[task]["env"]
                entry = saved.get((task, seed))
                if entry is None:
                    rows[key] = {"task": task, "seed": seed, "env": env_id,
                                 "model_saved": False, "training_record": flat_record(task, seed)}
                    print(f"  {key:<20} model not saved", flush=True)
                    continue
                node = build_flat_ppo(task)
                attach_sb3_primitive_policy(node, env_id, NO_HOOK)
                d = materialise(entry, Path(tmp))
                NodeRegistry(BASELINE_RUNS).load_into(node, run_dir=d, require_frozen=True)
                meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
                if int(meta["seed"]) != seed:
                    raise RuntimeError(f"{key}: loaded seed {meta['seed']}")
                factory = subgoal_env_factory(node, env_id, NO_HOOK, monitor=False)
                # Load check, as eval_regression.py does: the recorded held-out
                # score must reproduce exactly (deterministic only).
                hl = evaluate_node(node, factory, seeds=heldout_seeds(N))
                load_ok = (round(hl.success_rate, 4) == meta["heldout"]["success_rate"]
                           and round(hl.mean_length, 2) == meta["heldout"]["mean_length"])
                if not load_ok and not smoke:
                    raise RuntimeError(f"{key}: held-out score does not reproduce "
                                       f"({hl.success_rate}, {hl.mean_length} vs {meta['heldout']})")
                row = run_row(node, factory, seeds)
                rows[key] = {"task": task, "seed": seed, "env": env_id, "model_saved": True,
                             "source": entry["source"], "spec_hash": node.spec_hash(),
                             "load_check_heldout": {"recorded": meta["heldout"]["success_rate"],
                                                    "now": round(hl.success_rate, 4),
                                                    "identical": load_ok},
                             "training_record": flat_record(task, seed), **row}
                print_row(f"{key} ({entry['source']})", row)
    return rows


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--torch-threads", type=int, default=2)
    args = ap.parse_args()
    n = 5 if args.smoke else N
    seeds = fixed_seeds(n, SEED_BASE)
    set_torch_threads(args.torch_threads)
    t0 = time.perf_counter()

    registry = NodeRegistry(RUNS)
    prims, t3, t5 = load_library(registry)
    pb = load_pickup_box_training(registry)
    out = {"seeds": [seeds[0], seeds[-1]], "sampling_seeds": list(SAMPLING_SEEDS),
           "torch_seed_per_episode": "1_000_003 * sampling_seed + env_seed",
           "mode_sensitive_rule": "|pooled stochastic - deterministic| >= 0.10",
           "state_repeat": ("H-7a.5: full grid encoding + agent pose + carried object recur "
                            "within one node's own episode; deterministic episodes only"),
           "note": ("evaluation only; a stochastic success rate is not a gate and freezes "
                    "nothing; v2 §5.3 evaluates deterministically")}

    # ---- E.0 -----------------------------------------------------------
    print("E.0  which levels sample (solve_doorkey graph, until every node was called)")
    out["E0"] = e0(t5, seeds)
    for mode, r in out["E0"].items():
        print(f"  {mode:<14}" + "  ".join(f"{k}: {v['deterministic_flags']}"
                                         for k, v in r["calls"].items()))

    # ---- E.1 -----------------------------------------------------------
    print("\nE.1  frozen nodes under their own hooks")
    e1 = []
    for name, (node, spec) in prims.items():
        for env_id in spec["env_ids"]:
            factory = subgoal_env_factory(node, env_id, spec["hook"], monitor=False)
            row = run_row(node, factory, seeds)
            row["crosscheck"] = crosscheck(node, factory, seeds, row)
            e1.append({"node": node.id, "env": env_id, "hook": spec["hook"].spec,
                       "harness": "SubgoalEnv (as eval_regression.py)", **row})
            print_row(f"{node.id} | {env_id.replace('MiniGrid-', '')}", row)
    for node, env_id in ((t3, UNLOCK), (t5, DOORKEY)):
        factory = execution_env_factory(env_id, NO_HOOK)
        row = run_row(node, factory, seeds)
        row["crosscheck"] = crosscheck(node, factory, seeds, row)
        e1.append({"node": node.id, "env": env_id, "hook": NO_HOOK.spec,
                   "harness": "plain env, Executor (as eval_regression.py)", **row})
        print_row(f"{node.id} | {env_id.replace('MiniGrid-', '')}", row)
    out["E1"] = e1

    # ---- E.2 -----------------------------------------------------------
    print("\nE.2  gating F29 rows")
    gk, od, gtg = prims["get_key"][0], prims["open_door"][0], prims["go_to_goal"][0]
    dh, odtk = prims["drop_held"][0], prims["open_door_tk"][0]
    yellow, blue = prims["get_key_yellow"][0], prims["get_key_blue"][0]

    def matching_key(env, obs):
        """train_choice_task.stage_f29's matching-key start; Excluded if the key child failed."""
        sv = build_state_view(env, 0)
        obs = chain(yellow if sv.door().color == "yellow" else blue)(env, obs)
        if build_state_view(env, 1).carrying_type != "key":
            raise Excluded
        return obs

    two = two_keys(COLORS)
    f29 = [
        ("parent_reachable_eval.json[0]", gk, UNLOCK, NO_HOOK, None, "episode start"),
        ("parent_reachable_eval.json[1]", gk, UNLOCK, NO_HOOK, chain(od), "after a failed open_door"),
        ("parent_reachable_eval.json[2]", od, UNLOCK, NO_HOOK, chain(gk), "after get_key succeeded"),
        ("parent_reachable_eval.json[3]", od, DOORKEY, NO_HOOK, chain(gk), "after get_key succeeded"),
        ("parent_reachable_eval.json[4]", gtg, DOORKEY, NO_HOOK, chain(gk, od),
         "after the door was unlocked"),
        ("h6_preflight.json rows[0]", t3, UNLOCK, NO_HOOK, None, "episode start (H-5 regression)"),
        ("h6_preflight.json rows[1]", t3, DOORKEY, NO_HOOK, None, "episode start (layout transfer)"),
        ("h6_preflight.json rows[2]", gtg, DOORKEY, NO_HOOK, chain(t3),
         "after unlock_door ran (the handoff)"),
        ("h7_preflight.json base.rows[0]", t3, UNLOCKPICKUP, NO_HOOK, None, "episode start (A1)"),
        ("h7_preflight.json pickup_box.rows[0]", dh, UNLOCKPICKUP, NO_HOOK, chain(t3),
         "drop_held after unlock_door (A2a)"),
        ("choice_f29_open_door.json rows[0]", od, DOORKEY, two, matching_key,
         "[GATE] holding the matching key"),
        ("choice_f29_open_door_tk.json rows[0]", odtk, DOORKEY, two, matching_key,
         "[GATE] holding the matching key"),
    ]
    e2 = []
    for source, node, env_id, hook, prep, label in f29:
        factory = execution_env_factory(env_id, hook)
        row = run_row(node, factory, seeds, prepare=prep)
        row["crosscheck"] = crosscheck(node, factory, seeds, row, prepare=prep)
        e2.append({"source": f"runs/v2/{source}", "node": node.id, "env": env_id,
                   "hook": hook.spec, "start_state": label, "gate": True, **row})
        print_row(f"{node.id} | {env_id.replace('MiniGrid-', '')} | {label}", row)
    out["E2_f29"] = e2

    print("\nE.2  [diagnostic] scripted unlock_door -> drop_held -> pickup_box (TRAINING), UnlockPickup")
    goal = pb.goal
    variants = [
        ("all three deterministic", [(t3, DETERMINISTIC), (dh, DETERMINISTIC), (pb, DETERMINISTIC)],
         [(t3, DETERMINISTIC), (dh, DETERMINISTIC), (pb, DETERMINISTIC)]),
        ("only pickup_box sampled", [(t3, DETERMINISTIC), (dh, DETERMINISTIC), (pb, DETERMINISTIC)],
         [(t3, DETERMINISTIC), (dh, DETERMINISTIC), (pb, STOCHASTIC)]),
        ("all three sampled", [(t3, DETERMINISTIC), (dh, DETERMINISTIC), (pb, DETERMINISTIC)],
         [(t3, STOCHASTIC), (dh, STOCHASTIC), (pb, STOCHASTIC)]),
    ]
    seq = []
    for name, det_nm, sto_nm in variants:
        r = run_sequence(name, det_nm, sto_nm, seeds, goal)
        seq.append(r)
        d, s = r["deterministic"], r["stochastic"]
        print(f"  {name:<28} det {d['successes']}/{d['n']}  sampled {s['successes']}/{s['n']} "
              f"{s['by_sampling_seed']}  det failures by segment {d['failures_by_segment']}"
              f"{'  MODE-SENSITIVE' if r['mode_sensitive'] else ''}", flush=True)
    # The deterministic half is the same in every variant; it must also match
    # evaluate_node(pickup_box, prepare=chain(unlock_door, drop_held)).
    ref = evaluate_node(pb, execution_env_factory(UNLOCKPICKUP, NO_HOOK), seeds=seeds,
                        prepare=chain(t3, dh))
    mine = [bool(e[2]) for e in seq[0]["episodes"] if e[1] is None]
    seq_check = {"deterministic_vs_evaluate_node_chain": ref.per_episode == mine,
                 "evaluate_node_successes": ref.successes}
    if not seq_check["deterministic_vs_evaluate_node_chain"]:
        print(f"  NOTE: sequence vs evaluate_node(chain) differ: {ref.successes} vs {sum(mine)}")
    out["E2_scripted_sequence"] = {
        "label": "diagnostic; pickup_box is the spec-2 seed-42 TRAINING checkpoint, never frozen",
        "env": UNLOCKPICKUP, "pickup_box_spec_hash": pb.spec_hash(),
        "pickup_box_status": pb.status.value, "crosscheck": seq_check, "variants": seq}

    # ---- E.3 -----------------------------------------------------------
    print("\nE.3  flat PPO")
    out["E3_flat_ppo"] = e3(registry, seeds, args.smoke)

    # ---- E.4 -----------------------------------------------------------
    frozen_rows = [("E1", r) for r in e1] + [("E2", r) for r in e2]
    fails = sum(r["deterministic"]["failures"] for _, r in frozen_rows)
    reps = sum(r["deterministic"]["failures_with_state_repeat"] for _, r in frozen_rows)
    # E.2's h6 rows[0] is E.1's unlock_door row again (same env, hook, seeds, harness).
    dup = next(r for r in e2 if r["source"].endswith("h6_preflight.json rows[0]"))
    fails_u = fails - dup["deterministic"]["failures"]
    reps_u = reps - dup["deterministic"]["failures_with_state_repeat"]
    prim_rows = [r for _, r in frozen_rows if r["node"] not in (t3.id, t5.id)]
    out["E4"] = {
        "mode_sensitive_rows": (
            [f"E1 {r['node']} | {r['env']}" for r in e1 if r["mode_sensitive"]]
            + [f"E2 {r['node']} | {r['env']} | {r['start_state']}" for r in e2 if r["mode_sensitive"]]
            + [f"E2 scripted sequence: {r['variant']}" for r in seq if r["mode_sensitive"]]
            + [f"E3 flat {k}" for k, r in out["E3_flat_ppo"].items()
               if r.get("model_saved") and r["mode_sensitive"]]),
        "state_repeat_share_frozen": {
            "over": "deterministic failures of frozen nodes, E.1 and E.2 F29 rows",
            "failures": fails, "with_state_repeat": reps,
            "share": round(reps / fails, 4) if fails else None,
            "without_duplicate_row": {"failures": fails_u, "with_state_repeat": reps_u,
                                      "share": round(reps_u / fails_u, 4) if fails_u else None,
                                      "duplicate": "E.2 h6_preflight rows[0] = E.1 unlock_door | Unlock"},
            "primitives_only": {
                "failures": sum(r["deterministic"]["failures"] for r in prim_rows),
                "with_state_repeat": sum(r["deterministic"]["failures_with_state_repeat"]
                                         for r in prim_rows)},
        },
    }
    up = {k: v for k, v in out["E3_flat_ppo"].items() if v["task"] == "unlockpickup"}
    flat_avail = [k for k, v in up.items() if v.get("model_saved")]
    by = {r["variant"]: r for r in seq}
    out["E4"]["unlockpickup_comparison"] = {
        "deterministic": {
            "scripted_sequence_all_deterministic": by["all three deterministic"]["deterministic"]["successes"],
            "n": n,
            "flat_ppo": ({k: up[k]["deterministic"]["successes"] for k in flat_avail}
                         if flat_avail else "not available"),
        },
        "sampled": {
            "scripted_sequence_all_three_sampled": by["all three sampled"]["stochastic"]["successes"],
            "scripted_sequence_only_pickup_box_sampled": by["only pickup_box sampled"]["stochastic"]["successes"],
            "n": 3 * n,
            "flat_ppo": ({k: up[k]["stochastic"]["successes"] for k in flat_avail}
                         if flat_avail else "not available"),
        },
        "flat_models_not_available": [k for k, v in out["E3_flat_ppo"].items()
                                      if not v.get("model_saved")],
    }
    out["wall_seconds"] = round(time.perf_counter() - t0, 1)
    print(f"\nE.4  mode-sensitive: {out['E4']['mode_sensitive_rows']}")
    print(f"     state-repeat share: {out['E4']['state_repeat_share_frozen']}")
    print(f"     UnlockPickup: {json.dumps(out['E4']['unlockpickup_comparison'])}")

    if args.smoke:
        print(f"\nsmoke run, {out['wall_seconds']}s; nothing written")
        return 0
    path = RUNS / "execution_mode_audit.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {path.relative_to(REPO)} ({out['wall_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
