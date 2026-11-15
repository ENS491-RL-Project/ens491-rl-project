"""
W1a (`docs/plans/2026-11-15_W1a_pickup_box_sampled_plan.md`): does `{0,4}
pickup_box`, spec 2 unchanged, pass when it acts by sampling? Evaluation only.

Nothing is trained, continued, frozen or registered, and nothing under
`runs/v2/0-4_pickup_box/` or `runs/v2/baselines/` is written. The models are the
final checkpoints of the W1a training runs (seeds 43-47, run store), plus the
seed-42 registry checkpoint, labelled exploratory and counted toward nothing.

Episodes are run as `audit_execution_mode.py` runs them: one Executor per
episode, the node's own timeout, and for a sampled episode
torch.manual_seed(1_000_003 * sampling_seed + env_seed) once the node under test
is about to start (H-7a.5's definition). The chain that builds the handoff state
runs deterministically, as frozen nodes act today.

Rows, per model (seeds fixed by the plan before any result):
    own hook   500 env seeds 50000-50499 on pickup_box's own reset hook
               (SubgoalEnv, as training's held-out score); argmax, then sampled
               with sampling seed 0
    handoff    500 env seeds 51000-51499 on UnlockPickup from the env's reset:
               unlock_door -> drop_held (both FROZEN, argmax), then pickup_box
               argmax and sampled (sampling seed 0). An episode is valid if the
               chain left the handoff state (door open, hand empty); invalid
               episodes are counted and excluded from the rate.

Validation gate (run first, always): the seed-42 checkpoint on H-7a.5's handoff
condition must reproduce `runs/v2/pickup_box_spec2_trace.json` (condition H:
held-out seeds 20000-20049, deterministic and sampling seeds 0, 1, 2) episode
for episode. If it does not, nothing is scored.

Load checks: each model's weights must hash to the policy_sha256 its run
recorded, and its deterministic score on the held-out seeds 20000-20049 under
its own hook must equal the held-out score its training run recorded.

    python scripts/eval_w1a.py --validate   gate only; nothing written
    python scripts/eval_w1a.py --smoke      gate on 10 seeds, 10 seeds per row; nothing written
    python scripts/eval_w1a.py              ->  runs/v2/w1a_pickup_box_sampled.json (never overwritten)
    python scripts/eval_w1a.py --extension  ->  runs/v2/w1a_pickup_box_sampled_ext.json: the declared
                                                extension, seeds 48-52 scored the same way, ten-seed
                                                verdict (>= 7 of 10) with 43-47 read from the file above
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import torch  # noqa: E402

from src.envs.subgoal_env import NO_HOOK  # noqa: E402
from src.skills.executor import DETERMINISTIC, STOCHASTIC, Executor  # noqa: E402
from src.skills.registry import NodeRegistry, SpecMismatch  # noqa: E402
from src.skills.state import build_state_view  # noqa: E402
from src.training.evaluation import (  # noqa: E402
    evaluate_node,
    heldout_seeds,
    run_episode,
    wilson_lower,
    wilson_upper,
)
from src.training.node_trainer import (  # noqa: E402
    attach_sb3_primitive_policy,
    execution_env_factory,
    set_torch_threads,
    subgoal_env_factory,
)
from src.training.run_store import policy_hash  # noqa: E402
from src.types import NodeStatus  # noqa: E402
from diagnose_pickup_box import _full_state  # noqa: E402   H-7a.5's state, reused
from eval_h7_preflight import chain  # noqa: E402
from h7_nodes import RUNS, UNLOCKPICKUP, h7_primitive_specs, load_h7_primitive, load_unlock_graph  # noqa: E402
from train_primitives import build_node  # noqa: E402

OUT = RUNS / "w1a_pickup_box_sampled.json"
OUT_EXT = RUNS / "w1a_pickup_box_sampled_ext.json"
TRACE = RUNS / "pickup_box_spec2_trace.json"
SPEC_HASH = "a529aa4e74808ba0"

OWN_SEEDS = list(range(50_000, 50_500))
HANDOFF_SEEDS = list(range(51_000, 51_500))
SAMPLING_SEED = 0
TRACE_SAMPLING_SEEDS = (0, 1, 2)
SMOKE_N = 10

# Fixed by the plan before any result.
PASS_RATE = 0.90            # sampled success rate over valid episodes, >=
MIN_VALID = 475             # of 500
TEST_SEEDS = (43, 44, 45, 46, 47)
EXT_SEEDS = (48, 49, 50, 51, 52)       # the declared extension, used only if exactly 3 of 5 pass
EXT_MIN_PASS = 7                       # of the ten seeds 43-52

# Step 1's runs (run store). policy_sha256 and held-out are what each run recorded.
MODELS = {
    43: {"run_id": "20261115T101446-5873ceac6862464b85649487814a89c9",
         "policy_sha256": "93a8f59055514688cc8da6afc71749dece330460512255f3864caf92b1ad47d5",
         "heldout": (33, 46.34)},
    44: {"run_id": "20261115T101446-69710ce424b34302bfa06089bc095e23",
         "policy_sha256": "91ba6bfe74aead3631a89e7ab5a1f91ab6948c123fac91f73b97449036e01160",
         "heldout": (32, 49.44)},
    45: {"run_id": "20261115T101446-6302c245a9234d03899de05502766e70",
         "policy_sha256": "c5214ed682dfd583d47c3b7d6db82fd65f08ebf61a5b47cfae5a54f7fbb5e330",
         "heldout": (33, 46.4)},
    46: {"run_id": "20261115T101446-87237f38da3541c5abaa2ec28cfffcc9",
         "policy_sha256": "1f1da91fb87e3f41d005946f4e5b2023dce7e77bb8910725788276cdae7d50b5",
         "heldout": (39, 33.16)},
    47: {"run_id": "20261115T101446-ad3cc19a52c04c1c889c028952820498",
         "policy_sha256": "54a52944e67ec5d586daa24cf9544d5e9ac0c3b8186eb31f386f7139ef05cad0",
         "heldout": (38, 35.3)},
    # The declared extension (exactly 3 of 43-47 passed), trained the same way.
    48: {"run_id": "20261115T110149-d06cd25e74a64b5ba0c55aa3ca382678",
         "policy_sha256": "5beb10e129e6561b69b129c8191822f28977906d45efd14703b31b44e6bab356",
         "heldout": (42, 28.02)},
    49: {"run_id": "20261115T110149-6836754e7b2b406881534d659b579e19",
         "policy_sha256": "bd0b5c6308d50cf0e032b5cec106045dc0fb853606f72925695e86845b0da6c3",
         "heldout": (35, 41.6)},
    50: {"run_id": "20261115T110149-a8aaf057e0aa44d599b9a807c264a3ce",
         "policy_sha256": "c2a32ec827cb4f5150bc513674eaa2ed2440945cf7dd379cdca57cfef19edc42",
         "heldout": (35, 42.16)},
    51: {"run_id": "20261115T110149-64d86d6b213a48d69577c0c7015d27c5",
         "policy_sha256": "8e55e05d4117e04414f9033bb3ac80f4d55280e8e1b0b8daa2a754dd3658cf29",
         "heldout": (31, 50.3)},
    52: {"run_id": "20261115T110149-93a6755ede694e5b8efa8db58f247697",
         "policy_sha256": "b796adb6077740f51d7c0fa5323928beaf1459f2a69e4840f4a0639c7ad7bf51",
         "heldout": (35, 41.36)},
}


# Choices the plan does not spell out, made when this script was written (before
# any W1a score was seen) and recorded in the result file.
IMPLEMENTATION_CHOICES = {
    "loading": ("run-store directories hold no registry meta.json, so NodeRegistry.load_into "
                "cannot read them; the script builds the node from the spec, requires run.json's "
                "node_spec to equal it, loads policy_state_dict.pt, and requires the loaded "
                "weights' policy_sha256 to equal the one the run recorded"),
    "load_check": ("each model's argmax score on the held-out seeds 20000-20049 under its own "
                   "hook must equal the held-out score its training run recorded, or the script stops"),
    "own_hook_harness": "SubgoalEnv, the harness training's held-out score uses",
    "handoff_harness": ("plain env (execution_env_factory) from UnlockPickup's own reset, "
                        "chain(unlock_door, drop_held) as H-7a.5's condition H"),
    "crosscheck": ("in every row the argmax episodes must equal evaluate_node and the sampled "
                   "ones run_episode under the same torch seed, episode for episode, or the script stops"),
    "validation_gate_strictness": ("episode for episode (success and steps), not only the "
                                   "per-sampling-seed counts the plan asks for"),
    "validity_check": ("read once after the chain, from the StateView: a door exists and is open, "
                       "carrying_type is None; validity is the same in the argmax and sampled pass "
                       "(the chain is argmax and consumes no torch RNG), checked"),
    "timeouts": ("failures that used pickup_box's full timeout (120 steps); failures cut short "
                 "by the env's own step limit are counted separately as env_ended"),
    "state_repeats": "counted over argmax failures among valid episodes",
    "verdict": "computed by the script from the plan's fixed rule; seed 42 is excluded",
    "smoke": "--smoke runs the gate on 10 seeds and 10 seeds per row and writes nothing",
}
# Added for the declared extension, after the 43-47 result (which was exactly 3 of 5).
# Seeds, rows, thresholds and harnesses are unchanged; only where the result goes.
EXTENSION_CHOICES = {
    "extension_output": ("--extension scores seeds 48-52 exactly as 43-47 were scored and writes "
                         "runs/v2/w1a_pickup_box_sampled_ext.json; the 43-47 file is read, never "
                         "rewritten, and the ten-seed verdict (>= 7 of 10) is computed from both"),
    "env_ended_label": ("on the own-hook row (SubgoalEnv) the wrapper truncates at pickup_box's own "
                        "timeout, so env_ended there equals timeouts and does not mean the env's "
                        "288-step limit; on the handoff row it does"),
}


def torch_seed(sampling_seed: int, env_seed: int) -> int:
    return 1_000_003 * int(sampling_seed) + int(env_seed)


def runs_root() -> Path:
    return Path(os.environ.get("ENS491_RUNS_ROOT", REPO / "runs" / "training"))


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def _fresh_pickup_box():
    spec = h7_primitive_specs()["pickup_box"]
    node = build_node(spec)
    attach_sb3_primitive_policy(node, UNLOCKPICKUP, spec["hook"])
    if node.spec_hash() != SPEC_HASH:
        raise SpecMismatch(f"pickup_box spec_hash is {node.spec_hash()}, the plan names {SPEC_HASH}")
    return node, spec


def load_run_store_model(seed: int):
    """A run-store final checkpoint, TRAINING, never frozen. Spec and weights are checked."""
    m = MODELS[seed]
    d = runs_root() / m["run_id"]
    manifest = json.loads((d / "run.json").read_text(encoding="utf-8"))
    node, spec = _fresh_pickup_box()
    if manifest["node_spec"] != json.loads(json.dumps(node.spec())):
        raise SpecMismatch(f"seed {seed}: {d} was trained under a different spec")
    if manifest["policy_sha256"] != m["policy_sha256"]:
        raise RuntimeError(f"seed {seed}: run.json policy_sha256 differs from the one step 1 recorded")
    for name in ("model.zip", "policy_state_dict.pt", "complete.json"):
        if not (d / name).exists():
            raise FileNotFoundError(f"seed {seed}: no {name} in {d}")
    state = torch.load(d / "policy_state_dict.pt", map_location="cpu", weights_only=True)
    node.policy.load_state_dict(state)
    node.status = NodeStatus(manifest["node_status"])
    if node.status is not NodeStatus.TRAINING:
        raise RuntimeError(f"seed {seed}: expected TRAINING, found {node.status.value}")
    sha = policy_hash(node.policy)
    if sha != m["policy_sha256"]:
        raise RuntimeError(f"seed {seed}: loaded weights hash to {sha}, not {m['policy_sha256']}")
    return node, spec, {"source": f"run store {m['run_id']}", "run_id": m["run_id"],
                        "policy_sha256": sha, "recorded_heldout": m["heldout"]}


def load_seed42(registry):
    """The spec-2 seed-42 registry checkpoint, TRAINING (as H-7a.5 loads it). Exploratory."""
    node, spec = _fresh_pickup_box()
    registry.load_into(node, require_frozen=False)
    if node.status is not NodeStatus.TRAINING:
        raise RuntimeError(f"{node.id}: expected TRAINING, found {node.status.value}")
    meta = registry.read_meta(node.id)
    if int(meta["seed"]) != 42:
        raise RuntimeError(f"registry pickup_box is seed {meta['seed']}, not 42")
    held = meta["heldout"]
    return node, spec, {"source": "runs/v2/0-4_pickup_box/ (registry, TRAINING)", "run_id": None,
                        "policy_sha256": policy_hash(node.policy),
                        "recorded_heldout": (held["successes"], held["mean_length"])}


# --------------------------------------------------------------------------
# one episode
# --------------------------------------------------------------------------

def episode(node, env, seed, *, mode, sampling_seed=None, prepare=None) -> dict:
    """
    As `audit_execution_mode.episode` runs it, plus the handoff check: after
    `prepare`, is the door open and the hand empty?
    """
    obs, _ = env.reset(seed=seed)
    if prepare is not None:
        obs = prepare(env, obs)
    u = env.unwrapped
    sv = build_state_view(env, 0)
    door = sv.door()
    valid = bool(door is not None and door.is_open and sv.carrying_type is None)
    env_ended_before = int(u.step_count) >= int(u.max_steps)
    if sampling_seed is not None:
        torch.manual_seed(torch_seed(sampling_seed, seed))
    version = 0
    ex = Executor(node, execution_mode=mode)
    seen = {_full_state(u): 0}
    repeat_step = period = None
    steps = 0
    env_done = False
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
            env_done = True
            break
    success = bool(node.goal(sv))
    if repeat_step is not None and success:
        raise RuntimeError(f"{node.id} seed {seed}: a deterministic state repeat, yet the episode succeeded")
    return {"seed": seed, "sampling_seed": sampling_seed, "valid": valid, "success": success,
            "steps": steps, "timeout": (not success) and steps >= node.timeout,
            "env_ended": (not success) and env_done, "env_ended_before_node": env_ended_before,
            "repeat_step": repeat_step, "repeat_period": period}


def run_mode(node, env_factory, seeds, *, mode, sampling_seed=None, prepare=None) -> list[dict]:
    env = env_factory()
    try:
        return [episode(node, env, s, mode=mode, sampling_seed=sampling_seed, prepare=prepare)
                for s in seeds]
    finally:
        env.close()


# --------------------------------------------------------------------------
# validation gate
# --------------------------------------------------------------------------

def validate(seed42, t3, dh, n: int) -> dict:
    """seed 42 on H-7a.5's condition H must reproduce the trace, episode for episode."""
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    if trace["spec_hash"] != SPEC_HASH:
        raise RuntimeError(f"trace spec_hash {trace['spec_hash']} != {SPEC_HASH}")
    seeds = heldout_seeds(50)[:n]
    ref = {(r["seed"], r["sampling_seed"]): (bool(r["success"]), int(r["steps"]))
           for r in trace["episodes"] if r["condition"] == "H" and r["seed"] in seeds}
    factory = execution_env_factory(UNLOCKPICKUP, NO_HOOK)
    prep = chain(t3, dh)
    mine = run_mode(seed42, factory, seeds, mode=DETERMINISTIC, prepare=prep)
    for k in TRACE_SAMPLING_SEEDS:
        mine += run_mode(seed42, factory, seeds, mode=STOCHASTIC, sampling_seed=k, prepare=prep)
    got = {(r["seed"], r["sampling_seed"]): (r["success"], r["steps"]) for r in mine}
    diff = sorted((k for k in ref if ref[k] != got.get(k)),
                  key=lambda k: (k[1] is not None, k[1] or 0, k[0]))
    counts = {}
    for label, k in [("deterministic", None)] + [(f"sampled_{k}", k) for k in TRACE_SAMPLING_SEEDS]:
        counts[label] = {"trace": sum(v[0] for kk, v in ref.items() if kk[1] == k),
                         "now": sum(v[0] for kk, v in got.items() if kk[1] == k)}
    return {"against": TRACE.relative_to(REPO).as_posix() + " condition H",
            "seeds": f"{seeds[0]}-{seeds[-1]}", "sampling_seeds": list(TRACE_SAMPLING_SEEDS),
            "episodes_compared": len(ref), "episodes_differing": len(diff),
            "first_differences": [{"seed": k[0], "sampling_seed": k[1], "trace": ref[k],
                                   "now": got.get(k)} for k in diff[:5]],
            "successes": counts, "passed": not diff and len(ref) == len(got) == 4 * n}


# --------------------------------------------------------------------------
# rows
# --------------------------------------------------------------------------

def rate(records) -> dict:
    valid = [r for r in records if r["valid"]]
    n, k = len(valid), sum(r["success"] for r in valid)
    ok_steps = [r["steps"] for r in valid if r["success"]]
    return {"episodes": len(records), "valid": n, "invalid": len(records) - n, "successes": k,
            "success_rate": round(k / n, 4) if n else None,
            "wilson_lower": round(wilson_lower(k, n), 4) if n else None,
            "wilson_upper": round(wilson_upper(k, n), 4) if n else None,
            "mean_steps_successes": round(sum(ok_steps) / len(ok_steps), 2) if ok_steps else None,
            "timeouts": sum(r["timeout"] for r in valid),
            "env_ended": sum(r["env_ended"] for r in valid),
            "env_ended_before_node": sum(r["env_ended_before_node"] for r in records)}


def score_row(node, factory, seeds, *, prepare=None) -> dict:
    det = run_mode(node, factory, seeds, mode=DETERMINISTIC, prepare=prepare)
    sto = run_mode(node, factory, seeds, mode=STOCHASTIC, sampling_seed=SAMPLING_SEED, prepare=prepare)
    # Cross-check against the standard harness: identical episodes or raise.
    ref = evaluate_node(node, factory, seeds=seeds, prepare=prepare)
    if ref.per_episode != [r["success"] for r in det] or ref.lengths != [r["steps"] for r in det]:
        raise RuntimeError(f"{node.id}: argmax episodes differ from evaluate_node")
    env = factory()
    try:
        ref0 = []
        for s in seeds:
            torch.manual_seed(torch_seed(SAMPLING_SEED, s))   # prepare and reset consume no torch RNG
            ok, steps, _ = run_episode(node, env, s, execution_mode=STOCHASTIC, prepare=prepare)
            ref0.append((ok, steps))
    finally:
        env.close()
    if ref0 != [(r["success"], r["steps"]) for r in sto]:
        raise RuntimeError(f"{node.id}: sampled episodes differ from run_episode")
    d, s = rate(det), rate(sto)
    fails = [r for r in det if r["valid"] and not r["success"]]
    d["failures_with_state_repeat"] = sum(r["repeat_step"] is not None for r in fails)
    d["failures"] = len(fails)
    if [r["valid"] for r in det] != [r["valid"] for r in sto]:
        raise RuntimeError(f"{node.id}: validity differs between argmax and sampled episodes")
    row_pass = (None if s["valid"] < MIN_VALID * len(seeds) / 500
                else bool(s["successes"] >= PASS_RATE * s["valid"]))
    return {"seeds": f"{seeds[0]}-{seeds[-1]}", "sampling_seed": SAMPLING_SEED,
            "argmax": d, "sampled": s, "passes": row_pass,
            "crosscheck": {"argmax_vs_evaluate_node": "identical",
                           "sampled_vs_run_episode": "identical"},
            "invalid_seeds": [r["seed"] for r in det if not r["valid"]],
            "episodes": [[r["seed"], r["sampling_seed"], int(r["valid"]), int(r["success"]),
                          r["steps"], r["repeat_step"], r["repeat_period"]] for r in det + sto]}


def score_model(node, spec, info, t3, dh, own_seeds, handoff_seeds) -> dict:
    # Load check: the training run's held-out score, deterministic, own hook, SubgoalEnv.
    own = subgoal_env_factory(node, UNLOCKPICKUP, spec["hook"], monitor=False)
    held = evaluate_node(node, own, seeds=heldout_seeds(50))
    k, ml = info["recorded_heldout"]
    load_ok = held.successes == k and round(held.mean_length, 2) == ml
    if not load_ok:
        raise RuntimeError(f"held-out does not reproduce: {held.successes}, {held.mean_length} "
                           f"vs recorded {k}, {ml}")
    rows = {
        "own_hook": {"env": UNLOCKPICKUP, "hook": spec["hook"].spec,
                     "harness": "SubgoalEnv (as training's held-out)",
                     **score_row(node, own, own_seeds)},
        "handoff": {"env": UNLOCKPICKUP, "hook": NO_HOOK.spec,
                    "harness": "plain env from reset; unlock_door -> drop_held (FROZEN, argmax), then pickup_box",
                    **score_row(node, execution_env_factory(UNLOCKPICKUP, NO_HOOK), handoff_seeds,
                                prepare=chain(t3, dh))},
    }
    passes = [rows["own_hook"]["passes"], rows["handoff"]["passes"]]
    return {**{k2: v for k2, v in info.items() if k2 != "recorded_heldout"},
            "spec_hash": node.spec_hash(), "status": node.status.value,
            "load_check_heldout": {"recorded": [k, ml],
                                   "now": [held.successes, round(held.mean_length, 2)],
                                   "identical": load_ok},
            "rows": rows,
            "passes": (None if None in passes else all(passes))}


def verdict(models: dict) -> dict:
    per = {s: models[str(s)]["passes"] for s in TEST_SEEDS}
    k = sum(v is True for v in per.values())
    undecided_rows = [s for s, v in per.items() if v is None]
    if undecided_rows:
        outcome = "chain problem in seeds " + ", ".join(map(str, undecided_rows)) + "; not judged"
    elif k >= 4:
        outcome = "W1a passes"
    elif k <= 2:
        outcome = "W1a fails"
    else:
        outcome = "undecided (exactly 3): the declared extension applies (seeds 48-52, >= 7 of 10)"
    return {"rule": ("row passes: sampled SR over valid >= 0.90 and >= 475/500 valid; seed passes: "
                     "both rows; over 43-47: >= 4 pass -> pass, <= 2 -> fail, 3 -> extension"),
            "test_seeds": list(TEST_SEEDS), "seed_passes": {str(s): v for s, v in per.items()},
            "seeds_passing": k, "outcome": outcome,
            "seed_42": "exploratory; counts toward nothing"}


def verdict_extension(first: dict, ext_models: dict) -> dict:
    """The declared extension: >= 7 of the ten seeds 43-52 pass."""
    if not first["verdict"]["outcome"].startswith("undecided (exactly 3)"):
        raise RuntimeError(f"the extension applies only after exactly 3 of 5; "
                           f"{OUT.name} says {first['verdict']['outcome']!r}")
    per = {**{int(s): v for s, v in first["verdict"]["seed_passes"].items()},
           **{s: ext_models[str(s)]["passes"] for s in EXT_SEEDS}}
    k = sum(v is True for v in per.values())
    undecided = [s for s, v in per.items() if v is None]
    if undecided:
        outcome = ("undeterminable (chain): fewer than 475 valid handoff episodes in seeds "
                   + ", ".join(map(str, undecided)))
    else:
        outcome = "W1a passes" if k >= EXT_MIN_PASS else "W1a fails"
    return {"rule": "extension declared in the plan: >= 7 of the ten seeds 43-52 pass",
            "seeds": sorted(per), "seed_passes": {str(s): per[s] for s in sorted(per)},
            "seeds_passing": k, "outcome": outcome,
            "first_five_from": OUT.relative_to(REPO).as_posix(),
            "seed_42": "exploratory; counts toward nothing"}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="validation gate only; nothing written")
    ap.add_argument("--smoke", action="store_true", help="10 seeds everywhere; nothing written")
    ap.add_argument("--extension", action="store_true",
                    help="score seeds 48-52 (the declared extension) into a separate file")
    ap.add_argument("--torch-threads", type=int, default=2)
    args = ap.parse_args()
    set_torch_threads(args.torch_threads)
    t0 = time.perf_counter()
    out_path = OUT_EXT if args.extension else OUT
    if not args.validate and not args.smoke and out_path.exists():
        raise SystemExit(f"{out_path.relative_to(REPO)} exists; it is never overwritten")
    first = None
    if args.extension:
        if not OUT.exists():
            raise SystemExit(f"--extension needs {OUT.relative_to(REPO)} (seeds 43-47)")
        first = json.loads(OUT.read_text(encoding="utf-8"))
        missing = [s for s in EXT_SEEDS if s not in MODELS]
        if missing:
            raise SystemExit(f"no recorded run for extension seeds {missing}")

    registry = NodeRegistry(RUNS)
    _, t3 = load_unlock_graph(registry, verbose=False)
    dh = load_h7_primitive("drop_held", registry, verbose=False)
    seed42, spec42, info42 = load_seed42(registry)

    gate_n = SMOKE_N if args.smoke else 50
    print(f"validation gate: seed 42 on the trace's condition H, {gate_n} seeds x (argmax + 3 sampled)")
    gate = validate(seed42, t3, dh, gate_n)
    print(f"  {gate['episodes_compared']} episodes compared, {gate['episodes_differing']} differ; "
          f"successes {gate['successes']}  -> {'PASSED' if gate['passed'] else 'FAILED'}", flush=True)
    if not gate["passed"]:
        print(f"  first differences: {gate['first_differences']}")
        print("validation gate failed: nothing scored")
        return 1
    if args.validate:
        print(f"gate only; nothing written ({time.perf_counter() - t0:.1f}s)")
        return 0

    own_seeds = OWN_SEEDS[:SMOKE_N] if args.smoke else OWN_SEEDS
    handoff_seeds = HANDOFF_SEEDS[:SMOKE_N] if args.smoke else HANDOFF_SEEDS
    models = {}
    for seed in (EXT_SEEDS if args.extension else (42, *TEST_SEEDS)):
        node, spec, info = (seed42, spec42, info42) if seed == 42 else load_run_store_model(seed)
        r = score_model(node, spec, info, t3, dh, own_seeds, handoff_seeds)
        if seed == 42:
            r["label"] = "exploratory; counts toward nothing"
        models[str(seed)] = r
        line = []
        for name, row in r["rows"].items():
            a, s = row["argmax"], row["sampled"]
            line.append(f"{name}: valid {s['valid']}/{s['episodes']} argmax {a['successes']} "
                        f"sampled {s['successes']} ({s['success_rate']}, LB {s['wilson_lower']}) "
                        f"pass={row['passes']}")
        print(f"  seed {seed}: " + " | ".join(line) + f"  -> seed pass={r['passes']}", flush=True)

    out = {
        "plan": "docs/plans/2026-11-15_W1a_pickup_box_sampled_plan.md (revision 2)",
        "node": seed42.id, "spec_hash": SPEC_HASH, "env": UNLOCKPICKUP,
        "own_hook_seeds": f"{own_seeds[0]}-{own_seeds[-1]}",
        "handoff_seeds": f"{handoff_seeds[0]}-{handoff_seeds[-1]}",
        "sampling_seed": SAMPLING_SEED,
        "torch_seed_per_episode": "1_000_003 * sampling_seed + env_seed, set after the chain",
        "chain": {"nodes": [t3.id, dh.id], "mode": "argmax (FROZEN default)",
                  "spec_hashes": [t3.spec_hash(), dh.spec_hash()],
                  "policy_sha256": [policy_hash(t3.policy), policy_hash(dh.policy)]},
        "valid_episode": "after the chain the door is open and the hand is empty",
        "state_repeat": ("H-7a.5: full grid encoding + agent pose + carried object recur within "
                         "pickup_box's own episode; argmax episodes only"),
        "implementation_choices": ({**IMPLEMENTATION_CHOICES, **EXTENSION_CHOICES}
                                   if args.extension else IMPLEMENTATION_CHOICES),
        "validation_gate": gate,
        "models": models,
        "verdict": verdict_extension(first, models) if args.extension else verdict(models),
        "note": ("evaluation only; nothing frozen or registered. Sampled and argmax numbers are "
                 "different measurements and are not compared as one. Evaluation episodes are "
                 "not counted as training cost."),
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }
    print(f"verdict: {out['verdict']['outcome']} ({out['verdict']['seed_passes']})")
    if args.smoke:
        print(f"smoke run, {out['wall_seconds']}s; nothing written")
        return 0
    out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {out_path.relative_to(REPO)} ({out['wall_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
