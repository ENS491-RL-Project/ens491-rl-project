"""
W1a's test, unchanged, on `pickup_box` specification 3 (decision 0011, `Proposed`;
`docs/plans/2026-11-15_W1a_spec3_plan.md`, step 3). Evaluation only.

Everything is `eval_w1a.py`'s: episodes, seeding, rows, cross-checks, the row and
seed rule. Only the models and the env seed ranges differ.

    gate         before spec 3 is scored, the ten spec-2 models (W1a seeds 43-52)
                 must reproduce W1a's recorded episodes on W1a's ranges, row by row,
                 argmax and sampled, episode for episode (so W1a's counts)
    test         spec-3 models, seeds 43-47, on fresh ranges: own hook 52000-52499,
                 handoff 53000-53499. "Own hook" is spec 2's hook, as in W1a
    descriptive  the same spec-3 models on W1a's ranges (50000-50499, 51000-51499),
                 beside spec 2's recorded numbers for the same training seed; never
                 part of the verdict

    python scripts/eval_w1a_spec3.py --out <path>      full run
    python scripts/eval_w1a_spec3.py --smoke           10 seeds per row everywhere; nothing written
    python scripts/eval_w1a_spec3.py --extension <first run json> --out <path>
                                                       the declared extension: the gate again, then
                                                       seeds 48-52 scored the same way; ten-seed
                                                       verdict (>= 7 of 10)
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

import torch  # noqa: E402

from src.envs.subgoal_env import NO_HOOK  # noqa: E402
from src.skills.executor import DETERMINISTIC, STOCHASTIC  # noqa: E402
from src.skills.registry import NodeRegistry, SpecMismatch  # noqa: E402
from src.training.evaluation import heldout_seeds  # noqa: E402
from src.training.node_trainer import (  # noqa: E402
    attach_sb3_primitive_policy,
    execution_env_factory,
    pooled_eval_fn,
    set_torch_threads,
    subgoal_env_factory,
)
from src.training.run_store import policy_hash  # noqa: E402
from src.types import NodeStatus  # noqa: E402
from eval_h7_preflight import chain  # noqa: E402
from eval_w1a import (  # noqa: E402
    EXT_SEEDS,
    HANDOFF_SEEDS as W1A_HANDOFF,
    MIN_VALID,
    OUT as W1A_OUT,
    OUT_EXT as W1A_OUT_EXT,
    OWN_SEEDS as W1A_OWN,
    PASS_RATE,
    SAMPLING_SEED,
    TEST_SEEDS,
    load_run_store_model,
    runs_root,
    run_mode,
    score_row,
)
from h7_nodes import RUNS, UNLOCKPICKUP, h7_primitive_specs  # noqa: E402
from pickup_box_spec3 import spec3  # noqa: E402
from train_primitives import build_node  # noqa: E402

OWN_SEEDS = list(range(52_000, 52_500))
HANDOFF_SEEDS = list(range(53_000, 53_500))
SPEC3_HASH = "6c6428043d97ac75"
SMOKE_N = 10

# Step 2's runs (run store). policy_sha256 and held-out (mixed hook) are what each run recorded.
MODELS_S3 = {
    43: {"run_id": "20261115T125405-eaea36cdb8f3401dbb3b2da6b22bf6e6",
         "policy_sha256": "7d6c6432c17eee2cb41fb9ae1586e04978235adc1c663c7092748ddf57707f00",
         "heldout": (43, 24.12)},
    44: {"run_id": "20261115T125405-265dd652486a4819ad31a27f28cd5aec",
         "policy_sha256": "eb4a6ffac20085b03336939927c3bdfdc394d57932bc363ffc0c4f9bb44a1822",
         "heldout": (42, 24.8)},
    45: {"run_id": "20261115T125406-03312e12d82048218c20b91fa6cad0fc",
         "policy_sha256": "b05ae151ac43c398f4fb2597d13e1a7845fd07a9584f44510a42c5891bd32f93",
         "heldout": (41, 27.18)},
    46: {"run_id": "20261115T125406-3acc5eb0473a486a813bc1f8c8d06f99",
         "policy_sha256": "d6e41337ba3c1f916525177a12049aa4526773b15add31686d895389841c7bb9",
         "heldout": (43, 22.76)},
    47: {"run_id": "20261115T125406-1d8954e479834f4c872e7573a66ee0da",
         "policy_sha256": "e288776d05ed95dc9febbaeec86307a825d5396b570096af1a92760841058dee",
         "heldout": (36, 39.4)},
    # The declared extension (exactly 3 of 43-47 passed), trained the same way.
    48: {"run_id": "20261115T135439-1f929506660c437091715ecb9598ec6d",
         "policy_sha256": "5539c11647145764be7b10fceec06e908970fe06547e275df0a339c386a875f9",
         "heldout": (43, 22.7)},
    49: {"run_id": "20261115T135439-eeb1352055354d48ac15e645f5d65a25",
         "policy_sha256": "b591e0e32b428a818d59c9230fc6eaa160f553bfb7b319e1bb01e88437a966cf",
         "heldout": (44, 21.22)},
    50: {"run_id": "20261115T135439-a1b83ba59ce346d29b9e7f10d04de3ec",
         "policy_sha256": "e70a45bb25b7d109ba977a99a2cbcb19a9324447de93ceef02e2af10403bae69",
         "heldout": (40, 29.48)},
    51: {"run_id": "20261115T135439-edb7a0edcdb243b3a5d64563edcace4b",
         "policy_sha256": "2dcf91e5bab78666ea5727791f343b702bcb8321ef0b7226304ae88d170952ba",
         "heldout": (44, 20.3)},
    52: {"run_id": "20261115T135439-6a444ccab2ac4fd79ddff7cc3f835cea",
         "policy_sha256": "bd7639819c5eeb64062613421ac8ff69adb6248f843a025b0a27b2ebbf5ccd5c",
         "heldout": (43, 22.96)},
}

IMPLEMENTATION_CHOICES = {
    "reuse": "episodes, rows, cross-checks and the row rule are eval_w1a.py's functions, imported",
    "gate": ("the ten spec-2 models' argmax and sampled episodes on W1a's ranges are re-run with "
             "eval_w1a.run_mode and must equal W1a's recorded episodes (validity, success, steps); "
             "W1a's harness cross-checks are not repeated in the gate"),
    "spec3_loading": ("node built from pickup_box_spec3.spec3 (spec_hash 6c6428043d97ac75); run.json's "
                      "node_spec must equal it; weights must hash to the run's policy_sha256; the argmax "
                      "held-out score under the mixed hook must equal the training run's"),
    "own_hook_row": "spec 2's hook (open_door, key_off_door, random pose) through SubgoalEnv, as in W1a",
    "descriptive_rows": "spec-3 models on W1a's ranges with the same score_row; never in the verdict",
}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_spec3_model(seed: int, spec):
    m = MODELS_S3[seed]
    d = runs_root() / m["run_id"]
    manifest = json.loads((d / "run.json").read_text(encoding="utf-8"))
    node = build_node(spec)
    if node.spec_hash() != SPEC3_HASH:
        raise SpecMismatch(f"spec 3 hash is {node.spec_hash()}, expected {SPEC3_HASH}")
    attach_sb3_primitive_policy(node, UNLOCKPICKUP, spec["hook"])
    if manifest["node_spec"] != json.loads(json.dumps(node.spec())):
        raise SpecMismatch(f"seed {seed}: {d} was trained under a different spec")
    if manifest["policy_sha256"] != m["policy_sha256"]:
        raise RuntimeError(f"seed {seed}: run.json policy_sha256 differs from step 2's record")
    for name in ("model.zip", "policy_state_dict.pt", "complete.json"):
        if not (d / name).exists():
            raise FileNotFoundError(f"seed {seed}: no {name} in {d}")
    node.policy.load_state_dict(torch.load(d / "policy_state_dict.pt", map_location="cpu",
                                           weights_only=True))
    node.status = NodeStatus(manifest["node_status"])
    if node.status is not NodeStatus.TRAINING:
        raise RuntimeError(f"seed {seed}: expected TRAINING, found {node.status.value}")
    sha = policy_hash(node.policy)
    if sha != m["policy_sha256"]:
        raise RuntimeError(f"seed {seed}: loaded weights hash to {sha}")
    held = pooled_eval_fn(node, spec["env_ids"], spec["hook"], n_episodes=50,
                          seed_base=heldout_seeds(1)[0])()
    k, ml = m["heldout"]
    if held.successes != k or round(held.mean_length, 2) != ml:
        raise RuntimeError(f"seed {seed}: held-out does not reproduce ({held.successes}, "
                           f"{held.mean_length} vs {k}, {ml})")
    return node, {"run_id": m["run_id"], "policy_sha256": sha, "spec_hash": node.spec_hash(),
                  "load_check_heldout_mixed_hook": {"recorded": [k, ml],
                                                    "now": [held.successes, round(held.mean_length, 2)],
                                                    "identical": True}}


# --------------------------------------------------------------------------
# gate: spec 2 reproduces W1a
# --------------------------------------------------------------------------

def w1a_rows(seed: int) -> dict:
    path = W1A_OUT_EXT if seed in EXT_SEEDS else W1A_OUT
    return json.loads(path.read_text(encoding="utf-8"))["models"][str(seed)]["rows"]


def gate_spec2(seed: int, t3, dh, n: int) -> dict:
    node, spec2, _ = load_run_store_model(seed)
    rows = w1a_rows(seed)
    out = {}
    cases = {"own_hook": (subgoal_env_factory(node, UNLOCKPICKUP, spec2["hook"], monitor=False),
                          W1A_OWN[:n], None),
             "handoff": (execution_env_factory(UNLOCKPICKUP, NO_HOOK), W1A_HANDOFF[:n], chain(t3, dh))}
    for name, (factory, seeds, prep) in cases.items():
        rec = {(e[0], e[1]): (bool(e[2]), bool(e[3]), int(e[4])) for e in rows[name]["episodes"]}
        got = run_mode(node, factory, seeds, mode=DETERMINISTIC, prepare=prep)
        got += run_mode(node, factory, seeds, mode=STOCHASTIC, sampling_seed=SAMPLING_SEED, prepare=prep)
        mine = {(r["seed"], r["sampling_seed"]): (r["valid"], r["success"], r["steps"]) for r in got}
        diff = [k for k in mine if mine[k] != rec.get(k)]
        out[name] = {"episodes_compared": len(mine), "episodes_differing": len(diff),
                     "argmax_successes": [sum(v[0] and v[1] for k, v in mine.items() if k[1] is None),
                                          rows[name]["argmax"]["successes"] if n == 500 else None],
                     "sampled_successes": [sum(v[0] and v[1] for k, v in mine.items() if k[1] is not None),
                                           rows[name]["sampled"]["successes"] if n == 500 else None],
                     "first_differences": [{"key": list(k), "w1a": rec.get(k), "now": mine[k]}
                                           for k in diff[:3]]}
    out["passed"] = all(out[r]["episodes_differing"] == 0 for r in cases)
    return out


# --------------------------------------------------------------------------
# verdict (W1a's rule)
# --------------------------------------------------------------------------

def verdict(models: dict) -> dict:
    per = {s: models[str(s)]["test"]["passes"] for s in TEST_SEEDS}
    k = sum(v is True for v in per.values())
    undecided = [s for s, v in per.items() if v is None]
    if undecided:
        outcome = ("not determined (chain): fewer than 475 valid handoff episodes in seeds "
                   + ", ".join(map(str, undecided)))
    elif k >= 4:
        outcome = "spec 3 passes"
    elif k <= 2:
        outcome = "spec 3 fails"
    else:
        outcome = "undecided (exactly 3): train seeds 48-52 the same way, require >= 7 of 10"
    return {"rule": ("row passes: sampled SR over valid >= 0.90 and >= 475/500 valid; seed passes: "
                     "both rows; over 43-47: >= 4 pass -> pass, <= 2 -> fail, 3 -> extension"),
            "test_seeds": list(TEST_SEEDS), "seed_passes": {str(s): v for s, v in per.items()},
            "seeds_passing": k, "outcome": outcome}


def verdict_extension(first: dict, ext_models: dict) -> dict:
    """The declared extension: >= 7 of the ten seeds 43-52 pass."""
    if not first["verdict"]["outcome"].startswith("undecided (exactly 3)"):
        raise RuntimeError(f"the extension applies only after exactly 3 of 5; the first run says "
                           f"{first['verdict']['outcome']!r}")
    per = {**{int(s): v for s, v in first["verdict"]["seed_passes"].items()},
           **{s: ext_models[str(s)]["test"]["passes"] for s in EXT_SEEDS}}
    k = sum(v is True for v in per.values())
    undecided = [s for s, v in per.items() if v is None]
    if undecided:
        outcome = ("not determined (chain): fewer than 475 valid handoff episodes in seeds "
                   + ", ".join(map(str, undecided)))
    else:
        outcome = "spec 3 passes" if k >= 7 else "spec 3 fails"
    return {"rule": "extension declared in the plan: >= 7 of the ten seeds 43-52 pass",
            "seeds": sorted(per), "seed_passes": {str(s): per[s] for s in sorted(per)},
            "seeds_passing": k, "outcome": outcome}


def row_line(name, row):
    a, s = row["argmax"], row["sampled"]
    return (f"{name}: valid {s['valid']}/{s['episodes']} argmax {a['successes']} sampled "
            f"{s['successes']} ({s['success_rate']}, LB {s['wilson_lower']}) pass={row['passes']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--extension", default=None, metavar="FIRST_RUN_JSON",
                    help="score seeds 48-52 and compute the ten-seed verdict with the first run's file")
    ap.add_argument("--torch-threads", type=int, default=2)
    args = ap.parse_args()
    if not args.smoke:
        if not args.out:
            raise SystemExit("--out is required for a full run")
        if Path(args.out).exists():
            raise SystemExit(f"{args.out} exists; it is never overwritten")
    first = None
    if args.extension:
        first = json.loads(Path(args.extension).read_text(encoding="utf-8"))
        missing = [s for s in EXT_SEEDS if s not in MODELS_S3]
        if missing:
            raise SystemExit(f"no recorded run for extension seeds {missing}")
    set_torch_threads(args.torch_threads)
    t0 = time.perf_counter()
    n = SMOKE_N if args.smoke else 500

    registry = NodeRegistry(RUNS)
    spec, _stats, (t3, dh), _ = spec3(registry)
    spec2_hook = h7_primitive_specs()["pickup_box"]["hook"]

    print(f"gate: spec-2 models reproduce W1a on W1a's ranges ({n} seeds per row)")
    gate = {}
    for seed in (*TEST_SEEDS, *EXT_SEEDS):
        g = gate_spec2(seed, t3, dh, n)
        gate[str(seed)] = g
        print(f"  spec 2 seed {seed}: own {g['own_hook']['episodes_differing']} differ, handoff "
              f"{g['handoff']['episodes_differing']} differ -> {'PASSED' if g['passed'] else 'FAILED'}",
              flush=True)
    if not all(g["passed"] for g in gate.values()):
        print("gate FAILED: spec 3 not scored")
        return 1

    own_new, hand_new = OWN_SEEDS[:n], HANDOFF_SEEDS[:n]
    own_old, hand_old = W1A_OWN[:n], W1A_HANDOFF[:n]
    hand_factory = execution_env_factory(UNLOCKPICKUP, NO_HOOK)
    models = {}
    for seed in (EXT_SEEDS if args.extension else TEST_SEEDS):
        node, info = load_spec3_model(seed, spec)
        own_factory = subgoal_env_factory(node, UNLOCKPICKUP, spec2_hook, monitor=False)
        test = {"own_hook": {"seeds_range": "52000-52499", **score_row(node, own_factory, own_new)},
                "handoff": {"seeds_range": "53000-53499",
                            **score_row(node, hand_factory, hand_new, prepare=chain(t3, dh))}}
        tp = [test["own_hook"]["passes"], test["handoff"]["passes"]]
        test["passes"] = None if None in tp else all(tp)
        desc = {"own_hook": score_row(node, own_factory, own_old),
                "handoff": score_row(node, hand_factory, hand_old, prepare=chain(t3, dh))}
        w1a = w1a_rows(seed)
        desc["spec2_same_training_seed_recorded"] = {
            r: {"argmax": w1a[r]["argmax"], "sampled": w1a[r]["sampled"], "passes": w1a[r]["passes"]}
            for r in ("own_hook", "handoff")}
        models[str(seed)] = {**info, "test": test, "descriptive_old_ranges": desc}
        print(f"  spec 3 seed {seed}: TEST " + " | ".join(row_line(k, test[k]) for k in ("own_hook", "handoff"))
              + f" -> seed pass={test['passes']}", flush=True)
        print(f"               old ranges (descriptive) "
              + " | ".join(row_line(k, desc[k]) for k in ("own_hook", "handoff")), flush=True)

    out = {
        "plan": "docs/plans/2026-11-15_W1a_spec3_plan.md (revision 2), step 3",
        "decision": "docs/kb/decisions/0011-pickup-box-spec-3.md (Proposed)",
        "spec3_hash": SPEC3_HASH, "spec3_hook": spec["hook"].spec, "env": UNLOCKPICKUP,
        "test_ranges": {"own_hook": f"{own_new[0]}-{own_new[-1]}", "handoff": f"{hand_new[0]}-{hand_new[-1]}"},
        "descriptive_ranges": {"own_hook": f"{own_old[0]}-{own_old[-1]}", "handoff": f"{hand_old[0]}-{hand_old[-1]}"},
        "sampling_seed": SAMPLING_SEED, "thresholds": {"pass_rate": PASS_RATE, "min_valid": MIN_VALID},
        "chain": [t3.id, dh.id],
        "implementation_choices": IMPLEMENTATION_CHOICES,
        "gate_spec2_reproduces_w1a": gate,
        "models": models,
        "extension": bool(args.extension),
        "verdict": verdict_extension(first, models) if args.extension else verdict(models),
        "note": ("evaluation only; nothing frozen or registered. Sampled and argmax numbers are "
                 "different measurements. The descriptive rows on W1a's ranges are not part of the "
                 "verdict. Evaluation episodes are not counted as training cost."),
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }
    print(f"verdict: {out['verdict']['outcome']} ({out['verdict']['seed_passes']})")
    if args.smoke:
        print(f"smoke run, {out['wall_seconds']}s; nothing written")
        return 0
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {args.out} ({out['wall_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
