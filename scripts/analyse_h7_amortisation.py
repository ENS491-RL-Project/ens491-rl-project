"""
H-7a.4 (plan H-7.8, restricted to three tasks by H-7 amendment 1): the
amortisation table.

Reads only `runs/v2/**` and writes `runs/v2/h7_amortisation.json`. Per task, in
curriculum order: the nodes the hierarchy created for it and what they cost
(marginal), the running sum (cumulative), flat PPO's steps to STABLE on the same
task, and its running sum. Rules (plan H-7.8, amendment 1 §6/§7):

  - a primitive's cost is its `steps_trained`, a composite's its `env_steps_own`;
  - every composite cost is measured under one setting (`n_steps` 64,
    `eval_every` 512): `unlock_door` enters with its H-7a.3 re-measurement and
    `solve_doorkey` with its H-6.8 re-measurement; the costs the frozen
    checkpoints actually incurred go in a footnote;
  - multi-seed costs are min-max over seeds, and a sum of ranges is
    (sum of minima, sum of maxima);
  - a run not stable at its budget enters as `> steps` and makes every sum it is
    part of a lower bound, marked as such;
  - BlockedUnlockPickup is present and marked "H-7b, not run".

Two lines of their own (H-7a close-out K.3), so neither hides in a footnote:

  - **as built**: the costs the frozen checkpoints actually incurred, through
    DoorKey (`unlock_door` and `solve_doorkey` at `n_steps` 256, flat PPO's
    update size, instead of their re-measurements at 64);
  - **including the withdrawn specification**: the hierarchy's cumulative cost
    through UnlockPickup plus `pickup_box` attempt 1, per seed like every other
    cell; beside it the totals over all runs (six `pickup_box` runs, three flat
    PPO UnlockPickup runs).

The script recomputes every sum from the per-node files a second time, the slow
way, and refuses to write if the two disagree (plan success criterion 4).

    python scripts/analyse_h7_amortisation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs" / "v2"
OUT = RUNS / "h7_amortisation.json"
SEEDS = (42, 43, 44)

CAVEATS = [
    "1. `get_key` and `open_door` were trained on Unlock and DoorKey layouts together, so part of "
    "DoorKey's cost sits in the Unlock row.",
    "2. Primitive costs before H-7 come from one seed.",
    "3. Evaluation episodes are counted on neither side.",
    "4. Runs repeated because of a bug or a spec change are not counted; they are listed. "
    "`{0,4} pickup_box` took two specifications: the first (H-7 A.1: key in hand with probability "
    "0.5, UnlockPickup and BlockedUnlockPickup) was not stable within 600,064 env steps on any of "
    "three seeds, 3 x 600,064 = 1,800,192 env steps, not counted "
    "(`runs/v2/pickup_box_attempt1_s{42,43,44}.json`).",
    "5. The hierarchy was given hand-written subgoals and reset hooks that start it in states flat "
    "PPO must reach by itself. The comparison is \"designer decomposition plus reuse\" against "
    "\"flat\", not \"hierarchy\" against \"flat\" in the abstract.",
]


def load(rel: str) -> dict:
    return json.loads((RUNS / rel).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# one cost = per-seed values, each either exact or censored
# --------------------------------------------------------------------------

def cost(values: dict, source: str) -> dict:
    """values: {seed: (steps, exact)}"""
    vs = [v for v, _ in values.values()]
    censored = not all(ok for _, ok in values.values())
    return {"source": source,
            "per_seed": {str(s): {"steps": v, "stable": ok} for s, (v, ok) in values.items()},
            "min": min(vs), "max": max(vs), "lower_bound": censored}


def single(steps: int, source: str, seed: int = 42) -> dict:
    return cost({seed: (int(steps), True)}, source)


def primitive_meta(dirname: str) -> dict:
    m = load(f"{dirname}/meta.json")
    return single(m["steps_trained"], f"runs/v2/{dirname}/meta.json", m.get("seed", 42))


def primitive_seeds(name: str) -> dict:
    vals = {}
    for s in SEEDS:
        r = load(f"{name}_s{s}.json")
        vals[s] = (int(r["steps_trained"]), bool(r["stable"]))
    return cost(vals, f"runs/v2/{name}_s{{42,43,44}}.json steps_trained")


def composite_seeds(pattern: str, key=("composite",)) -> dict:
    vals = {}
    for s in SEEDS:
        r = load(pattern.format(s=s))
        for k in key:
            r = r[k]
        vals[s] = (int(r["env_steps_own"]), bool(r["stable"]))
    return cost(vals, f"runs/v2/{pattern.format(s='{42,43,44}')} env_steps_own")


def flat_seeds(pattern: str) -> dict:
    vals, best = {}, {}
    for s in SEEDS:
        r = load(pattern.format(s=s))["report"]
        vals[s] = (int(r["steps_trained"]), bool(r["stable"]))
        best[str(s)] = max(h["success_rate"] for h in r["criterion"]["history"])
    c = cost(vals, f"runs/v2/{pattern.format(s='{42,43,44}')} steps_trained")
    c["best_stopping_sr"] = best
    c["budget"] = load(pattern.format(s=SEEDS[0]))["report"]["budget"]
    return c


def not_run(why: str) -> dict:
    """A node the task needs that was never trained: contributes 0 and makes the sum a lower bound."""
    return {"source": None, "status": f"not run ({why})", "per_seed": {},
            "min": 0, "max": 0, "lower_bound": True}


def present(pattern: str) -> bool:
    return all((RUNS / pattern.format(s=s)).exists() for s in SEEDS)


def add(*costs: dict) -> dict:
    return {"min": sum(c["min"] for c in costs), "max": sum(c["max"] for c in costs),
            "lower_bound": any(c["lower_bound"] for c in costs)}


def fmt(c: dict) -> str:
    lo, hi = c["min"], c["max"]
    s = f"{lo:,}" if lo == hi else f"{lo:,} - {hi:,}"
    return ("> " + s) if c["lower_bound"] else s


# --------------------------------------------------------------------------

def build() -> dict:
    nodes = {
        "Unlock": {
            "{0,1} get_key": primitive_meta("0-1_get_key"),
            "{0,2} open_door": primitive_meta("0-2_open_door"),
            "{1,1} unlock_door (re-measured, n_steps 64)":
                composite_seeds("unlock_door_h7_s{s}.json"),
        },
        "DoorKey-6x6": {
            "{0,3} go_to_goal": primitive_meta("0-3_go_to_goal"),
            "{2,1} solve_doorkey (H-6.8, n_steps 64)":
                composite_seeds("solve_doorkey_h68_s{s}.json"),
        },
        "UnlockPickup": {
            "{0,9} drop_held": primitive_seeds("drop_held"),
            "{0,4} pickup_box (specification 2)": primitive_seeds("pickup_box"),
            "{2,2} unlock_pickup": (composite_seeds("unlock_pickup_s{s}.json")
                                    if present("unlock_pickup_s{s}.json")
                                    else not_run("A2b stop: pickup_box never froze")),
        },
    }
    flat = {
        "Unlock": flat_seeds("flat_unlock_ppo_s{s}.json"),
        "DoorKey-6x6": flat_seeds("flat_ppo_s{s}.json"),
        "UnlockPickup": flat_seeds("flat_unlockpickup_ppo_s{s}.json"),
        "BlockedUnlockPickup": flat_seeds("flat_blocked_ppo_s{s}.json"),
    }

    rows, h_cum, f_cum = [], None, None
    for task in ("Unlock", "DoorKey-6x6", "UnlockPickup"):
        marg = add(*nodes[task].values())
        h_cum = marg if h_cum is None else add(h_cum, marg)
        f_cum = flat[task] if f_cum is None else add(f_cum, flat[task])
        rows.append({
            "task": task, "status": "run",
            "hierarchy_nodes": nodes[task],
            "hierarchy_marginal": marg, "hierarchy_cumulative": h_cum,
            "flat_ppo": flat[task], "flat_cumulative": f_cum,
            "display": {"hierarchy_marginal": fmt(marg), "hierarchy_cumulative": fmt(h_cum),
                        "flat_ppo": fmt(flat[task]), "flat_cumulative": fmt(f_cum)},
        })
    rows.append({
        "task": "BlockedUnlockPickup", "status": "H-7b, not run",
        "hierarchy_nodes": None, "hierarchy_marginal": None, "hierarchy_cumulative": None,
        "flat_ppo": flat["BlockedUnlockPickup"], "flat_cumulative": None,
        "display": {"hierarchy_marginal": "H-7b, not run", "hierarchy_cumulative": "H-7b, not run",
                    "flat_ppo": fmt(flat["BlockedUnlockPickup"]), "flat_cumulative": "-"},
    })

    old_ud = load("1-1_unlock_door/meta.json")
    old_sd = load("2-1_solve_doorkey/meta.json")
    footnotes = [
        f"The frozen `{{1,1}} unlock_door` checkpoint was trained at `n_steps` 256 and cost "
        f"{old_ud['env_steps_own']:,} env steps of its own; the table uses its H-7a.3 "
        f"re-measurement at `n_steps` 64 (not a re-freeze).",
        f"The frozen `{{2,1}} solve_doorkey` checkpoint came from the earlier H-6.3 run at "
        f"{old_sd['env_steps_own']:,} env steps of its own; the table uses the H-6.8 "
        f"re-measurement at `n_steps` 64.",
        "Flat PPO's update size (`n_steps` 256 x 8 envs) was never revisited.",
        "A flat run not stable at its budget enters as `> steps_trained`; every sum containing it is "
        "a lower bound.",
    ]
    # [K.3] as built: the frozen checkpoints' own costs, through DoorKey
    as_built_nodes = {
        "Unlock": {
            "{0,1} get_key": nodes["Unlock"]["{0,1} get_key"],
            "{0,2} open_door": nodes["Unlock"]["{0,2} open_door"],
            "{1,1} unlock_door (frozen checkpoint, n_steps 256)":
                single(old_ud["env_steps_own"], "runs/v2/1-1_unlock_door/meta.json env_steps_own"),
        },
        "DoorKey-6x6": {
            "{0,3} go_to_goal": nodes["DoorKey-6x6"]["{0,3} go_to_goal"],
            "{2,1} solve_doorkey (frozen checkpoint, n_steps 256)":
                single(old_sd["env_steps_own"], "runs/v2/2-1_solve_doorkey/meta.json env_steps_own"),
        },
    }
    as_built, cum = [], None
    for task in ("Unlock", "DoorKey-6x6"):
        marg = add(*as_built_nodes[task].values())
        cum = marg if cum is None else add(cum, marg)
        f = next(r["flat_cumulative"] for r in rows if r["task"] == task)
        as_built.append({"task": task, "hierarchy_nodes": as_built_nodes[task],
                         "hierarchy_marginal": marg, "hierarchy_cumulative": cum,
                         "flat_cumulative": f,
                         "display": {"hierarchy_marginal": fmt(marg),
                                     "hierarchy_cumulative": fmt(cum),
                                     "flat_cumulative": fmt(f)}})

    # [K.3] including the withdrawn specification, through UnlockPickup
    attempt1 = primitive_seeds("pickup_box_attempt1")
    up = next(r for r in rows if r["task"] == "UnlockPickup")
    incl = add(up["hierarchy_cumulative"], attempt1)
    up_seeds = [v["steps"] for v in up["flat_ppo"]["per_seed"].values()]
    pb_runs = ([v["steps"] for v in attempt1["per_seed"].values()]
               + [v["steps"] for v in
                  up["hierarchy_nodes"]["{0,4} pickup_box (specification 2)"]["per_seed"].values()])
    with_withdrawn = {
        "through": "UnlockPickup",
        "pickup_box_attempt1": attempt1,
        "hierarchy_cumulative": incl,
        "flat_cumulative": up["flat_cumulative"],
        "display": {"hierarchy_cumulative": fmt(incl),
                    "flat_cumulative": fmt(up["flat_cumulative"])},
        "totals_over_all_runs": {
            "pickup_box_six_runs": {"runs": len(pb_runs), "env_steps": sum(pb_runs),
                                    "stable": False},
            "flat_ppo_unlockpickup_three_runs": {"runs": len(up_seeds),
                                                 "env_steps": sum(up_seeds), "stable": False},
        },
    }
    return {"rows": rows, "as_built": as_built, "including_withdrawn_spec": with_withdrawn,
            "caveats": CAVEATS, "footnotes": footnotes,
            "rules": __doc__.split("Rules")[1].split("Two lines")[0].strip()}


def recheck(doc: dict) -> list[str]:
    """Recompute every sum from the per-seed numbers, independently of add()."""
    errs = []
    h_lo = h_hi = f_lo = f_hi = 0
    for r in doc["rows"]:
        if r["status"] != "run":
            continue
        lo = hi = 0
        for name, c in r["hierarchy_nodes"].items():
            steps = [v["steps"] for v in c["per_seed"].values()] or [0]
            if (min(steps), max(steps)) != (c["min"], c["max"]):
                errs.append(f"{r['task']} {name}: per-seed range mismatch")
            lo += min(steps)
            hi += max(steps)
        h_lo, h_hi = h_lo + lo, h_hi + hi
        fs = [v["steps"] for v in r["flat_ppo"]["per_seed"].values()]
        f_lo, f_hi = f_lo + min(fs), f_hi + max(fs)
        for key, want in (("hierarchy_marginal", (lo, hi)), ("hierarchy_cumulative", (h_lo, h_hi)),
                          ("flat_cumulative", (f_lo, f_hi))):
            got = (r[key]["min"], r[key]["max"])
            if got != want:
                errs.append(f"{r['task']} {key}: table {got} != recomputed {want}")

    # [K.3] as built: straight from the five meta files
    m = {d: load(f"{d}/meta.json") for d in ("0-1_get_key", "0-2_open_door", "1-1_unlock_door",
                                             "0-3_go_to_goal", "2-1_solve_doorkey")}
    want_unlock = (m["0-1_get_key"]["steps_trained"] + m["0-2_open_door"]["steps_trained"]
                   + m["1-1_unlock_door"]["env_steps_own"])
    want_dk = (want_unlock + m["0-3_go_to_goal"]["steps_trained"]
               + m["2-1_solve_doorkey"]["env_steps_own"])
    for r, want in zip(doc["as_built"], (want_unlock, want_dk)):
        got = r["hierarchy_cumulative"]
        if (got["min"], got["max"]) != (want, want):
            errs.append(f"as built {r['task']}: table {got} != recomputed {want}")

    # [K.3] including the withdrawn spec: per seed from the per-node files
    att = [load(f"pickup_box_attempt1_s{s}.json")["steps_trained"] for s in SEEDS]
    spec2 = [load(f"pickup_box_s{s}.json")["steps_trained"] for s in SEEDS]
    flat_up = [load(f"flat_unlockpickup_ppo_s{s}.json")["report"]["steps_trained"] for s in SEEDS]
    w = doc["including_withdrawn_spec"]
    want = (h_lo + min(att), h_hi + max(att))
    if (w["hierarchy_cumulative"]["min"], w["hierarchy_cumulative"]["max"]) != want:
        errs.append(f"incl. withdrawn: table {w['hierarchy_cumulative']} != recomputed {want}")
    t = w["totals_over_all_runs"]
    if t["pickup_box_six_runs"]["env_steps"] != sum(att) + sum(spec2):
        errs.append("pickup_box six-run total mismatch")
    if t["flat_ppo_unlockpickup_three_runs"]["env_steps"] != sum(flat_up):
        errs.append("flat UnlockPickup three-run total mismatch")
    return errs


def main() -> int:
    doc = build()
    errs = recheck(doc)
    doc["sums_verified"] = not errs
    if errs:
        for e in errs:
            print("MISMATCH", e)
        return 1
    print(f"{'task':<22}{'hierarchy marginal':>26}{'hierarchy cumulative':>26}"
          f"{'flat PPO':>28}{'flat cumulative':>28}")
    for r in doc["rows"]:
        d = r["display"]
        print(f"{r['task']:<22}{d['hierarchy_marginal']:>26}{d['hierarchy_cumulative']:>26}"
              f"{d['flat_ppo']:>28}{d['flat_cumulative']:>28}")
    print("\nas built (frozen checkpoints' own costs):")
    for r in doc["as_built"]:
        d = r["display"]
        print(f"{r['task']:<22}{d['hierarchy_marginal']:>26}{d['hierarchy_cumulative']:>26}"
              f"{'':>28}{d['flat_cumulative']:>28}")
    w = doc["including_withdrawn_spec"]
    print(f"\nincluding the withdrawn pickup_box specification, through UnlockPickup: "
          f"{w['display']['hierarchy_cumulative']}  vs flat {w['display']['flat_cumulative']}")
    t = w["totals_over_all_runs"]
    print(f"totals over all runs: pickup_box {t['pickup_box_six_runs']['runs']} runs "
          f"{t['pickup_box_six_runs']['env_steps']:,}; flat PPO UnlockPickup "
          f"{t['flat_ppo_unlockpickup_three_runs']['runs']} runs "
          f"{t['flat_ppo_unlockpickup_three_runs']['env_steps']:,}")
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"\nsums verified; wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
