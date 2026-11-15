"""
W7 (C-2) analysis: steps to STABLE per arm, and the paired test of the plan.

Plan: docs/plans/2026-11-15_W7_pn_lateral_plan.md, "Criterion". Reads
runs/v2/w7/runs/<arm>_s<seed>.json written by train_w7_lateral.py.

"X beats Y": one-sided exact Wilcoxon signed-rank on the paired differences
steps(X) - steps(Y) gives p < 0.05, and the median of steps(X) / steps(Y) over all
pairs is at most 0.80. Censored runs count at the budget (401,408).

    python scripts/analyse_w7_lateral.py --arms A          # step 2 summary and gate
    python scripts/analyse_w7_lateral.py --arms A B C      # all tests
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from train_w7_lateral import BUDGET, OUT, TRAINING_SEEDS  # noqa: E402

ALPHA = 0.05
MAX_RATIO = 0.80
GATE_MIN_STABLE = 7


def average_ranks(values: list[float]) -> list[float]:
    """Ranks 1..n of `values`, ties given the average of the ranks they span."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def wilcoxon_exact_lower(d: list[float]) -> dict:
    """
    One-sided exact signed-rank test that the candidate is faster (d < 0).

    Zero differences are dropped; |d| is ranked with average ranks for ties;
    W+ is the sum of ranks of positive d; p = P(W+ <= observed) by enumerating all
    2^m sign assignments of those ranks.
    """
    nz = [x for x in d if x != 0]
    m = len(nz)
    if m == 0:
        return {"m": 0, "w_plus": None, "p": 1.0}
    ranks = average_ranks([abs(x) for x in nz])
    w_plus = sum(r for r, x in zip(ranks, nz) if x > 0)
    eps = 1e-9
    count = 0
    for signs in itertools.product((0, 1), repeat=m):
        if sum(r for r, s in zip(ranks, signs) if s) <= w_plus + eps:
            count += 1
    return {"m": m, "w_plus": w_plus, "p": count / 2 ** m}


def beats(x: dict[int, int], y: dict[int, int]) -> dict:
    """Does arm x beat arm y? Both map training seed -> metric steps."""
    seeds = sorted(set(x) & set(y))
    d = [x[s] - y[s] for s in seeds]
    test = wilcoxon_exact_lower(d)
    ratio = statistics.median(x[s] / y[s] for s in seeds)
    return {
        "pairs": len(seeds),
        **test,
        "median_ratio": ratio,
        "beats": bool(test["p"] < ALPHA and ratio <= MAX_RATIO),
    }


def load_arm(arm: str) -> dict[int, dict]:
    runs = {}
    for seed in TRAINING_SEEDS:
        p = OUT / "runs" / f"{arm}_s{seed}.json"
        if p.exists():
            runs[seed] = json.loads(p.read_text(encoding="utf-8"))
    return runs


def arm_summary(runs: dict[int, dict]) -> dict:
    steps = [r["metric_steps"] for r in runs.values()]
    stable = [r for r in runs.values() if r["stable"]]
    logs = [math.log(s) for s in steps]
    walls = [r["report"]["wall_seconds"] for r in runs.values()]
    sps = [r["report"]["steps_per_second"] for r in runs.values()]
    return {
        "n_runs": len(runs),
        "n_stable": len(stable),
        "n_censored": len(runs) - len(stable),
        "metric_steps": {s: r["metric_steps"] for s, r in sorted(runs.items())},
        "median_steps": statistics.median(steps) if steps else None,
        "min_steps": min(steps) if steps else None,
        "max_steps": max(steps) if steps else None,
        "sd_log_steps": statistics.stdev(logs) if len(logs) > 1 else None,
        "median_wall_seconds": statistics.median(walls) if walls else None,
        "median_steps_per_second": statistics.median(sps) if sps else None,
        "threads": sorted({r["threads"] for r in runs.values()}),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["A", "B", "C"])
    args = ap.parse_args()

    data = {arm: load_arm(arm) for arm in args.arms}
    out = {"budget": BUDGET, "arms": {}}
    for arm, runs in data.items():
        s = arm_summary(runs)
        out["arms"][arm] = s
        print(f"\narm {arm}: {s['n_stable']}/{s['n_runs']} STABLE, censored {s['n_censored']}")
        for seed, v in s["metric_steps"].items():
            mark = "" if runs[seed]["stable"] else "  (censored)"
            print(f"  seed {seed}: {v:>9,}{mark}")
        if s["n_runs"]:
            print(f"  median {s['median_steps']:,.0f}  range {s['min_steps']:,}-{s['max_steps']:,}"
                  f"  sd(log) {s['sd_log_steps']:.3f}  median wall {s['median_wall_seconds']}s"
                  f"  median steps/s {s['median_steps_per_second']}  threads {s['threads']}")
    if "A" in data and len(data["A"]) == len(TRAINING_SEEDS):
        gate = out["arms"]["A"]["n_stable"] >= GATE_MIN_STABLE
        out["gate_step2"] = {"min_stable": GATE_MIN_STABLE, "passes": gate}
        print(f"\nstep-2 gate (>= {GATE_MIN_STABLE}/10 STABLE in arm A): "
              f"{'PASSES' if gate else 'FAILS'}")

    steps = {arm: {s: r["metric_steps"] for s, r in runs.items()} for arm, runs in data.items()}
    tests = {}
    for x, y in [("B", "A"), ("B", "C"), ("A", "B"), ("C", "A"), ("C", "B")]:
        if x in steps and y in steps and steps[x] and steps[y]:
            tests[f"{x}_vs_{y}"] = t = beats(steps[x], steps[y])
            print(f"{x} beats {y}? p={t['p']:.4f} (m={t['m']}, W+={t['w_plus']}) "
                  f"median ratio {t['median_ratio']:.3f} -> {t['beats']}")
    out["tests"] = tests
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
