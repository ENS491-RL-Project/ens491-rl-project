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

from train_w7_lateral import BUDGET, EVAL_EVERY, OUT, TRAINING_SEEDS  # noqa: E402

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


def outcome(tests: dict) -> dict:
    """The plan's four exclusive rows, then the C-as-candidate rewording."""
    b_a, b_c, a_b = (tests[k]["beats"] for k in ("B_vs_A", "B_vs_C", "A_vs_B"))
    if a_b:
        row = "Negative transfer"
    elif b_a and b_c:
        row = "Forward transfer shown"
    elif b_a:
        row = "Faster, but not transfer"
        if tests["C_vs_B"]["beats"]:
            row += " (the trained source was slower than an untrained one)"
    else:
        row = "No measurable benefit"
    return {"row": row, "B_beats_A": b_a, "B_beats_C": b_c, "A_beats_B": a_b}


def _quantize(steps):
    """Steps to STABLE as this design reports them: a multiple of EVAL_EVERY, censored at BUDGET."""
    import numpy as np
    q = np.ceil(np.asarray(steps) / EVAL_EVERY) * EVAL_EVERY
    return np.minimum(q, BUDGET)


def power(median: float, sd_log: float, *, reductions=(0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5),
          n_pairs: int = 10, n_sims: int = 4000, seed: int = 0) -> dict:
    """
    How often "candidate beats A" passes, if steps to STABLE are log-normal with
    arm A's measured median and spread (sd of log steps) and the candidate's median
    is (1 - r) x A's. Arms independent, as in the plan's own simulation; steps are
    quantised to EVAL_EVERY and censored at BUDGET; the exact test and the ratio
    rule are the ones above.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    out = {}
    for r in reductions:
        passes = 0
        for _ in range(n_sims):
            a = _quantize(np.exp(rng.normal(math.log(median), sd_log, n_pairs)))
            x = _quantize(np.exp(rng.normal(math.log(median * (1 - r)), sd_log, n_pairs)))
            res = beats(dict(enumerate(x.tolist())), dict(enumerate(a.tolist())))
            passes += res["beats"]
        out[f"{r:.2f}"] = round(passes / n_sims, 3)
    return out


ARM_STYLE = {   # dataviz reference palette, categorical slots 1-3 (validated, light surface)
    "A": dict(color="#2a78d6", ls="-", label="A: no lateral"),
    "B": dict(color="#eb6834", ls="--", label="B: frozen open_door"),
    "C": dict(color="#1baf7a", ls=":", label="C: untrained source"),
}


def figure(data: dict[str, dict[int, dict]], path: Path) -> None:
    """Success rate on the stopping seeds against env steps, and the share of runs STABLE."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    grid = np.arange(EVAL_EVERY, BUDGET + 1, EVAL_EVERY)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    for arm, runs in data.items():
        st = ARM_STYLE[arm]
        curves = []
        for r in runs.values():
            sr = [v for _, v in r["success_series"]]
            # A run stops at STABLE; its last evaluation is carried forward.
            curves.append(sr + [sr[-1]] * (len(grid) - len(sr)))
        c = np.array(curves)
        ax1.fill_between(grid, c.min(0), c.max(0), color=st["color"], alpha=0.12, lw=0)
        ax1.plot(grid, np.median(c, 0), color=st["color"], ls=st["ls"], lw=2, label=st["label"])
        stop = np.array([r["metric_steps"] if r["stable"] else np.inf for r in runs.values()])
        share = [(stop <= g).mean() for g in grid]
        ax2.step(grid, share, where="post", color=st["color"], ls=st["ls"], lw=2, label=st["label"])
    ax1.axhline(0.90, color="#8a8a85", lw=1, ls=(0, (2, 3)))
    ax1.text(4_000, 0.905, "0.90 bar", ha="left", va="bottom", fontsize=8, color="#5f5e5a")
    ax1.set(xlabel="env steps", ylabel="success rate (50 stopping seeds, argmax)",
            ylim=(0, 1.02), xlim=(0, BUDGET),
            title="Median and range over 10 seeds (last value carried after STABLE)")
    ax2.set(xlabel="env steps", ylabel="share of runs STABLE", ylim=(0, 1.02), xlim=(0, BUDGET),
            title="Runs STABLE by env step")
    for ax in (ax1, ax2):
        ax.grid(True, color="#e6e5e0", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
        ax.title.set_fontsize(9.5)
    ax2.legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.suptitle("W7: go_to_goal on DoorKey-6x6, steps to STABLE by lateral source", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_scores(arm: str) -> dict[int, dict] | None:
    p = OUT / "scores" / f"{arm}.json"
    if not p.exists():
        return None
    return {m["seed"]: m for m in json.loads(p.read_text(encoding="utf-8"))["models"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--write", action="store_true",
                    help="write runs/v2/w7/w7_lateral.json and the figure (needs all 30 runs and scores)")
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
    if not args.write:
        return 0

    out["outcome"] = outcome(tests)
    print(f"\noutcome row: {out['outcome']['row']}")
    a = out["arms"]["A"]
    out["power"] = {
        "model": "log-normal, arm A's median and sd of log steps, arms independent, "
                 "quantised to eval_every and censored at the budget, exact test + ratio rule",
        "median": a["median_steps"], "sd_log": round(a["sd_log_steps"], 4), "n_pairs": 10,
        "n_sims": 4000, "pass_rate_by_reduction": power(a["median_steps"], a["sd_log_steps"]),
    }
    print(f"power (pass rate by true reduction): {out['power']['pass_rate_by_reduction']}")
    for arm in data:
        scores = load_scores(arm)
        if scores is None or len(scores) != len(TRAINING_SEEDS):
            raise SystemExit(f"final scores for arm {arm} missing; run eval_w7_lateral.py")
        rates = [m["score"]["success_rate"] for m in scores.values()]
        out["arms"][arm]["final_score"] = {
            "env_seeds": "60000-60499",
            "per_seed": {s: m["score"]["success_rate"] for s, m in sorted(scores.items())},
            "median": statistics.median(rates), "min": min(rates), "max": max(rates),
        }
        print(f"final score arm {arm}: median {statistics.median(rates):.3f} "
              f"range {min(rates):.3f}-{max(rates):.3f}")
    total = sum(r["steps_trained"] for runs in data.values() for r in runs.values())
    out["cost"] = {"training_env_steps_30_runs": total,
                   "evaluation_episodes_not_counted": True}
    fig = OUT / "w7_lateral_curves.png"
    figure(data, fig)
    out["figure"] = fig.relative_to(REPO).as_posix()
    (OUT / "w7_lateral.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {(OUT / 'w7_lateral.json').relative_to(REPO)} and {out['figure']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
