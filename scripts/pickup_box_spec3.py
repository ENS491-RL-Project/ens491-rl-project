"""
`{0,4} pickup_box` specification 3 (decision 0011, `Proposed`;
`docs/plans/2026-11-15_W1a_spec3_plan.md`): spec 2 with half of the training
episodes started at the real handoff.

This experiment runs while 0011 is `Proposed`, with Emir's go-ahead, under
`scripts/` only. Spec 2 (`h7_nodes.py`) and its hash are untouched; nothing is
frozen or registered from here.

    with probability 0.5   spec 2's hook (open_door, key_off_door, random pose)
    with probability 0.5   UnlockPickup from the env's reset, then the FROZEN
                           unlock_door and drop_held run to their ends (argmax);
                           pickup_box starts where they leave. If that is not the
                           handoff state (door open, hand empty), UnsuitableLayout.

Frozen nodes run inside a reset: `SubgoalEnv` hands the hook the raw env, so the
hook wraps it in the observation pipeline (`apply_obs_pipeline`, no reset) and
rolls the chain with `eval_h7_preflight.chain`, the same code the W1a evaluation
uses. DummyVecEnv only (one process; the frozen modules are shared).

Env steps used by the chain inside resets are counted per env instance; the
first `n_envs` instances are the training envs (built first, by
`make_training_vec_env`), the rest are evaluation envs.

    python scripts/pickup_box_spec3.py --gate                 step 1 gate, 200 env seeds (51000-51199)
    python scripts/pickup_box_spec3.py --gate --smoke         gate on 10 seeds
    python scripts/pickup_box_spec3.py --smoke                training loop smoke (8,192 steps), nothing kept
    python scripts/pickup_box_spec3.py --seed 43 --no-freeze --out-name pickup_box_s3_s43
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

from src.envs.obs import apply_obs_pipeline  # noqa: E402
from src.envs.subgoal_env import NO_HOOK, ResetHook, UnsuitableLayout  # noqa: E402
from src.skills.registry import NodeRegistry  # noqa: E402
from src.skills.state import build_state_view  # noqa: E402
from src.training.device import describe  # noqa: E402
from src.training.evaluation import heldout_seeds  # noqa: E402
from src.training.node_trainer import (  # noqa: E402
    WEAK_HELDOUT_LB,
    execution_env_factory,
    pooled_eval_fn,
    set_torch_threads,
    subgoal_env_factory,
    train_primitive,
)
from diagnose_pickup_box import _full_state  # noqa: E402
from diagnose_w1a_handoff import current_obs  # noqa: E402
from eval_h7_preflight import chain  # noqa: E402
from h7_nodes import H7A_BUDGETS, RUNS, UNLOCKPICKUP, h7_primitive_specs, load_h7_primitive, load_unlock_graph  # noqa: E402
from train_primitives import build_node  # noqa: E402

P_HANDOFF = 0.5
SPEC2_HASH = "a529aa4e74808ba0"
GATE_SEEDS = list(range(51_000, 51_200))      # W1a's handoff seeds; include known invalid ones
SMOKE_GATE_N = 10


# --------------------------------------------------------------------------
# chain statistics
# --------------------------------------------------------------------------

class ChainStats:
    """Per env instance: hook calls, branch taken, chain env steps, rejections."""

    def __init__(self) -> None:
        self.order: list[int] = []
        self.per: dict[int, dict] = {}
        self.log: list[tuple[int, str, bool]] = []     # (instance, branch, accepted), in call order

    def _rec(self, u) -> dict:
        k = id(u)
        if k not in self.per:
            self.order.append(k)
            self.per[k] = {"calls": 0, "handoff": 0, "spec2": 0, "chain_steps": 0,
                           "rejected_handoff": 0}
        return self.per[k]

    def branch(self, u, name: str) -> None:
        r = self._rec(u)
        r["calls"] += 1
        r[name] += 1

    def chain_done(self, u, steps: int, accepted: bool) -> None:
        r = self._rec(u)
        r["chain_steps"] += steps
        r["rejected_handoff"] += int(not accepted)
        self.log.append((id(u), "handoff", accepted))

    def snapshot(self, n_train: int) -> dict:
        def total(keys):
            out = {f: 0 for f in ("calls", "handoff", "spec2", "chain_steps", "rejected_handoff")}
            for k in keys:
                for f in out:
                    out[f] += self.per[k][f]
            return out
        return {"training_envs": total(self.order[:n_train]),
                "evaluation_envs": total(self.order[n_train:]),
                "n_env_instances": len(self.order)}


# --------------------------------------------------------------------------
# hooks
# --------------------------------------------------------------------------

def handoff_hook(t3, dh, stats: ChainStats) -> ResetHook:
    """UnlockPickup from reset, then unlock_door -> drop_held (FROZEN, argmax)."""
    roll = chain(t3, dh)

    def fn(env):
        u = env.unwrapped
        start = int(u.step_count)
        pipe = apply_obs_pipeline(env)          # wraps the raw env; no reset
        roll(pipe, current_obs(pipe))
        sv = build_state_view(env, 0)
        door = sv.door()
        ok = bool(door is not None and door.is_open and sv.carrying_type is None
                  and int(u.step_count) < int(u.max_steps))
        stats.chain_done(u, int(u.step_count) - start, ok)
        if not ok:
            raise UnsuitableLayout("handoff_chain: the chain did not leave the handoff state")

    return ResetHook(f"handoff_chain:{t3.id}>{dh.id}:argmax", fn)


def mixture_hook(p: float, handoff: ResetHook, other: ResetHook, stats: ChainStats) -> ResetHook:
    """With probability p (one draw from the env's own RNG) `handoff`, else `other`."""

    def fn(env):
        u = env.unwrapped
        if u._rand_float(0, 1) < p:
            stats.branch(u, "handoff")
            handoff(env)
        else:
            stats.branch(u, "spec2")
            other(env)

    return ResetHook(f"mix:p={p}:[{handoff.spec}]|[{other.spec}]", fn)


def spec3(registry):
    """(spec dict, stats, (unlock_door, drop_held), handoff hook) for pickup_box spec 3."""
    base = h7_primitive_specs()["pickup_box"]
    if build_node(base).spec_hash() != SPEC2_HASH:
        raise RuntimeError("spec 2's hash changed; spec 3 is defined against a529aa4e74808ba0")
    _, t3 = load_unlock_graph(registry, verbose=False)
    dh = load_h7_primitive("drop_held", registry, verbose=False)
    stats = ChainStats()
    hand = handoff_hook(t3, dh, stats)
    return {**base, "hook": mixture_hook(P_HANDOFF, hand, base["hook"], stats)}, stats, (t3, dh), hand


# --------------------------------------------------------------------------
# gate
# --------------------------------------------------------------------------

def eval_state(env, prep, seed):
    """What eval_w1a's handoff row starts pickup_box from: (valid, state, obs)."""
    obs, _ = env.reset(seed=seed)
    obs = prep(env, obs)
    sv = build_state_view(env, 0)
    door = sv.door()
    valid = bool(door is not None and door.is_open and sv.carrying_type is None)
    return valid, _full_state(env.unwrapped), np.asarray(obs)


def gate(registry, seeds) -> dict:
    spec, stats, (t3, dh), hand = spec3(registry)
    node = build_node(spec)
    prep = chain(t3, dh)
    ev = execution_env_factory(UNLOCKPICKUP, NO_HOOK)()
    raw = gym.make(UNLOCKPICKUP)
    sub_h = subgoal_env_factory(node, UNLOCKPICKUP, hand, monitor=False)()
    sub_m = subgoal_env_factory(node, UNLOCKPICKUP, spec["hook"], monitor=False)()
    res = {"direct": {"equal": 0, "both_rejected": 0, "mismatch": []},
           "subgoal_env_handoff_only": {"equal": 0, "redrawn_valid": 0, "mismatch": []},
           "subgoal_env_mixture": {"handoff_first_equal": 0, "spec2_first": 0,
                                   "redrawn": 0, "mismatch": []}}
    try:
        for s in seeds:
            valid, e_state, e_obs = eval_state(ev, prep, s)

            # (a) the hook on a raw env, as SubgoalEnv calls it
            raw.reset(seed=s)
            try:
                hand(raw)
                accepted = True
            except UnsuitableLayout:
                accepted = False
            if accepted != valid or (valid and _full_state(raw.unwrapped) != e_state):
                res["direct"]["mismatch"].append(s)
            elif valid:
                res["direct"]["equal"] += 1
            else:
                res["direct"]["both_rejected"] += 1

            # (b) through SubgoalEnv + obs pipeline, handoff branch only
            obs, _ = sub_h.reset(seed=s)
            u = sub_h.unwrapped
            ok_now = (u.carrying is None and build_state_view(sub_h, 0).door().is_open)
            if valid:
                if _full_state(u) == e_state and np.array_equal(np.asarray(obs), e_obs):
                    res["subgoal_env_handoff_only"]["equal"] += 1
                else:
                    res["subgoal_env_handoff_only"]["mismatch"].append(s)
            elif ok_now:
                res["subgoal_env_handoff_only"]["redrawn_valid"] += 1
            else:
                res["subgoal_env_handoff_only"]["mismatch"].append(s)

            # (c) through SubgoalEnv with the mixture: the first draw decides the branch
            n_log = len(stats.log)
            k_calls = sum(r["calls"] for r in stats.per.values())
            obs, _ = sub_m.reset(seed=s)
            calls = sum(r["calls"] for r in stats.per.values()) - k_calls
            new = stats.log[n_log:]
            first_handoff = bool(new) and new[0][1] == "handoff"
            # first call was handoff iff the first chain record belongs to that first call:
            # if calls == 1 the episode used exactly one draw.
            if calls > 1:
                res["subgoal_env_mixture"]["redrawn"] += 1
            elif first_handoff and new[0][2]:
                if _full_state(sub_m.unwrapped) == e_state and np.array_equal(np.asarray(obs), e_obs):
                    res["subgoal_env_mixture"]["handoff_first_equal"] += 1
                else:
                    res["subgoal_env_mixture"]["mismatch"].append(s)
            else:
                res["subgoal_env_mixture"]["spec2_first"] += 1
    finally:
        for e in (ev, raw, sub_h, sub_m):
            e.close()
    passed = not any(r["mismatch"] for r in res.values())
    return {"seeds": f"{seeds[0]}-{seeds[-1]}", "n": len(seeds),
            "spec2_hash": SPEC2_HASH, "spec3_hash": node.spec_hash(), "spec3_hook": spec["hook"].spec,
            **res, "chain_stats": stats.snapshot(0)["evaluation_envs"], "passed": passed}


# --------------------------------------------------------------------------
# training (step 2; never freezes or registers)
# --------------------------------------------------------------------------

def train(args) -> int:
    budget = H7A_BUDGETS["pickup_box"]["budget"]
    eval_every = H7A_BUDGETS["pickup_box"]["eval_every"]
    n_envs, n_eval = args.n_envs, 50
    if args.smoke:
        budget, eval_every, n_envs, n_eval = 8_000, 4_096, 4, 10
    name = args.out_name or f"pickup_box_s3_s{args.seed}"
    path = RUNS / f"{name}{'_smoke' if args.smoke else ''}.json"
    if path.exists():
        raise SystemExit(f"{path.relative_to(REPO)} exists; it is never overwritten")
    print(describe())
    print(f"torch threads: {set_torch_threads(args.torch_threads)}")
    registry = NodeRegistry(RUNS)
    spec, stats, _, _ = spec3(registry)
    node = build_node(spec)
    print(f"\n{'=' * 72}\n{node.id} SPEC 3  envs={spec['env_ids']}  timeout={node.timeout}"
          f"\n  hook={spec['hook'].spec}\n  spec_hash={node.spec_hash()} (spec 2: {SPEC2_HASH})"
          f"\n  seed={args.seed} budget={budget:,} eval_every={eval_every:,}  freeze=False\n{'=' * 72}")
    report, _model = train_primitive(
        node, spec["env_ids"], spec["hook"],
        total_timesteps=budget, eval_every=eval_every, n_eval=n_eval,
        n_envs=n_envs, seed=args.seed, device=args.device, verbose=1,
    )
    chain_train = stats.snapshot(n_envs)
    print(f"  steps={report['steps_trained']:,} wall={report['wall_seconds']}s "
          f"({report['steps_per_second']} steps/s) stable={report['stable']}")
    report["spec_hash"] = node.spec_hash()
    report["spec"] = "pickup_box specification 3 (decision 0011, Proposed)"
    held = pooled_eval_fn(node, spec["env_ids"], spec["hook"], n_episodes=n_eval,
                          seed_base=heldout_seeds(1)[0])()
    report["heldout"] = held.as_dict()
    report["weak_heldout"] = bool(held.wilson_lower < WEAK_HELDOUT_LB)
    report["chain_env_steps_in_resets"] = {
        "during_training": chain_train,
        "note": ("env steps the frozen unlock_door -> drop_held chain used inside resets; not in "
                 "steps_trained and not added to the node's cost. Training envs are the first "
                 "n_envs instances; the rest are the stopping evaluations' envs"),
    }
    report["note"] = ("stopping evaluations (0004, argmax) and the held-out score run on the mixed "
                      "hook, so they are not comparable with spec 2's; never frozen or registered")
    print(f"  not frozen (spec 3, decision 0011 Proposed); held-out (mixed hook): {held}")
    print(f"  chain env steps in resets: {json.dumps(chain_train)}")
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] wrote {path.relative_to(REPO)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true", help="step 1 gate only; writes nothing")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-freeze", action="store_true", help="required for training")
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--torch-threads", type=int, default=2)
    args = ap.parse_args()
    if args.gate:
        set_torch_threads(args.torch_threads)
        t0 = time.perf_counter()
        seeds = GATE_SEEDS[:SMOKE_GATE_N] if args.smoke else GATE_SEEDS
        g = gate(NodeRegistry(RUNS), seeds)
        print(json.dumps(g, indent=1))
        print(f"gate {'PASSED' if g['passed'] else 'FAILED'} ({time.perf_counter() - t0:.1f}s); nothing written")
        return 0 if g["passed"] else 1
    if not args.smoke and (args.seed is None or not args.no_freeze):
        raise SystemExit("training needs --seed and --no-freeze (spec 3 is never frozen or registered here)")
    if args.seed is None:
        args.seed = 43
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
