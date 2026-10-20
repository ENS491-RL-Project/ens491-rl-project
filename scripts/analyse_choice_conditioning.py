"""
[H-7a close-out K.1] Does the strict-mask selector's first key depend on the door's colour?

Reads only existing run files (no training, no evaluation):
  * runs/v2/unlock_matching_s{42,43,44}.json   -> conditions.strict.first_decision (H-6.8)
  * runs/v2/choice_diagnosis_arm_*.json        -> first_key_heldout (H-6.9 arms, exploratory)

All seven runs were scored deterministically on the same 50 held-out seeds (23 yellow
doors, 27 blue doors). H-6.8's metric is the first *key* child selected (it skips an
`open_door_tk` pick, of which there were none under the strict mask); the arms' metric is
the first child selected, which under the strict mask and an empty hand is always a key.

Per run: yellow-first count on each door colour, the gap between the two rates, first-key
accuracy, the accuracy a colour-blind policy with the same overall key rates would score,
and a two-sided Fisher exact test on the 2x2 table (door colour x first key). No
correction for seven runs. Also the probe gap p(yellow | yellow door) - p(yellow | blue
door) at every evaluation of the arms.

Writes runs/v2/choice_conditioning.json.
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs" / "v2"
YELLOW, BLUE = "{0,6} get_key_yellow", "{0,7} get_key_blue"
DECISIONS_PER_UPDATE = 512


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]]: sum of tables no more likely."""
    r1, c1, n = a + b, a + c, a + b + c + d

    def p(x: int) -> float:
        return comb(c1, x) * comb(n - c1, r1 - x) / comb(n, r1)

    p_obs = p(a)
    lo, hi = max(0, r1 - (n - c1)), min(r1, c1)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= p_obs * (1 + 1e-9)))


def row(label: str, source: str, ent_coef: float, y_on_y: int, n_y: int,
        y_on_b: int, n_b: int) -> dict:
    n = n_y + n_b
    correct = y_on_y + (n_b - y_on_b)
    p_yellow = (y_on_y + y_on_b) / n
    blind = p_yellow * n_y / n + (1 - p_yellow) * n_b / n
    return {
        "run": label, "source": source, "ent_coef": ent_coef,
        "yellow_first_on_yellow_door": [y_on_y, n_y],
        "yellow_first_on_blue_door": [y_on_b, n_b],
        "gap": round(y_on_y / n_y - y_on_b / n_b, 4),
        "first_key_accuracy": round(correct / n, 4),
        "colour_blind_accuracy_same_rates": round(blind, 4),
        "fisher_two_sided_p": round(fisher_two_sided(y_on_y, n_y - y_on_y,
                                                     y_on_b, n_b - y_on_b), 4),
    }


def h68_row(seed: int) -> dict:
    f = f"unlock_matching_s{seed}.json"
    d = json.loads((RUNS / f).read_text())
    strict = d["conditions"]["strict"]
    fd = strict["first_decision"]
    assert fd["episodes_with_no_key_pick"] == 0
    y, b = fd["by_colour"]["yellow"], fd["by_colour"]["blue"]
    # correct on a yellow door = yellow first; correct on a blue door = blue first
    return row(f"H-6.8, seed {seed}", f, strict["composite"]["ppo"]["ent_coef"],
               y["correct"], y["n"], b["n"] - b["correct"], b["n"])


def arm_row(name: str, label: str) -> tuple[dict, list]:
    f = f"choice_diagnosis_arm_{name}.json"
    d = json.loads((RUNS / f).read_text())
    by = d["first_key_heldout"]["by_door_colour"]
    y, b = by["yellow"], by["blue"]
    assert sum(y["first"].values()) == y["n"] and sum(b["first"].values()) == b["n"]
    r = row(label, f, d["ent_coef"], y["first"].get(YELLOW, 0), y["n"],
            b["first"].get(YELLOW, 0), b["n"])
    course = [{
        "update": e["at_steps"] // DECISIONS_PER_UPDATE,
        "probe_gap": round(e["door_yellow"][YELLOW] - e["door_blue"][YELLOW], 4),
    } for e in d["composite"]["eval_extra"]]
    return r, course


def main() -> int:
    rows = [h68_row(s) for s in (42, 43, 44)]
    courses = {}
    for name, label in [("unchanged_s42", "arm (a), seed 42"),
                        ("ent05_s42", "arm (b), seed 42"),
                        ("ent05_s43", "arm (b), seed 43"),
                        ("ent05_s44", "arm (b), seed 44")]:
        r, c = arm_row(name, label)
        rows.append(r)
        courses[label] = c
    for r in rows:
        assert r["yellow_first_on_yellow_door"][1] == 23
        assert r["yellow_first_on_blue_door"][1] == 27

    def window(c, lo, hi):
        v = [e["probe_gap"] for e in c if lo <= e["update"] <= hi]
        return {"updates": [lo, hi], "min": min(v), "max": max(v),
                "mean": round(sum(v) / len(v), 4)}

    course_summary = {lab: {"1-40": window(c, 1, 40), "41-80": window(c, 41, 80),
                            "81-100": window(c, 81, 100)}
                      for lab, c in courses.items()}
    out = {"note": __doc__.strip().splitlines()[0], "heldout_seeds": 50,
           "fisher_correction": "none (seven runs)", "rows": rows,
           "probe_gap_course_summary": course_summary, "probe_gap_course": courses}
    (RUNS / "choice_conditioning.json").write_text(json.dumps(out, indent=2))

    print(f"{'run':20s} {'Y|Y':>6s} {'Y|B':>6s} {'gap':>6s} {'acc':>5s} {'blind':>5s} {'p':>7s}")
    for r in rows:
        print(f"{r['run']:20s} {'%d/%d' % tuple(r['yellow_first_on_yellow_door']):>6s} "
              f"{'%d/%d' % tuple(r['yellow_first_on_blue_door']):>6s} {r['gap']:+.2f} "
              f"{r['first_key_accuracy']:5.2f} {r['colour_blind_accuracy_same_rates']:5.2f} "
              f"{r['fisher_two_sided_p']:7.4f}")
    for lab, s in course_summary.items():
        print(lab, {k: (v["min"], v["max"], v["mean"]) for k, v in s.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
