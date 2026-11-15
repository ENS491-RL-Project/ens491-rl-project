"""Figures for ENS 492 Progress Report II (PNG + PDF).

Reads only files already in ``runs/v2/``; trains and evaluates nothing.
The PR-I version is ``scripts/generate_report_figures.py`` and is left as it is.

    python scripts/generate_report_figures_pr2.py [--out reports/progress2/figures]

Fig. 1  v2 architecture (drawn; no data).
Fig. 2  the skill graph as built: FROZEN nodes from ``runs/v2/*/meta.json``; planned
        BlockedUnlockPickup nodes (v2 §11, T8-T9b) dashed.
Fig. 3  ``pickup_box`` per training seed, SAMPLED success rates only, own hook and handoff
        rows, spec 2 (W1a) vs spec 3, with the 0.90 line. Argmax rates are not plotted
        (they go in a separate table in the report).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "v2"

# Reference palette (light mode, print). Text in ink tokens, never series colour.
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"  # categorical slot 1
ORANGE = "#eb6834"  # categorical slot 2
BLUE_FILL = "#cde2fb"  # blue ramp step 100
GREY_FILL = "#f0efec"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.titlecolor": INK,
        "savefig.dpi": 300,
    }
)


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out / name}.png/.pdf")


# --------------------------------------------------------------------------- Fig. 1
def _box(ax, x, y, w, h, text, *, fill="white", edge=INK_2, ls="-", lw=1.2, size=9,
         weight="normal", color=INK):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       fc=fill, ec=edge, ls=ls, lw=lw)
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            color=color, weight=weight, linespacing=1.35)


def _arrow(ax, a, b, *, ls="-", color=INK_2, lw=1.1, style="-|>", rad=0.0):
    ax.add_patch(
        FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=11, color=color, lw=lw,
                        ls=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2)
    )


def fig1_architecture(out: Path) -> None:
    # Drawn at roughly page width (1 unit = 1 inch) so the fonts survive placement.
    fig, ax = plt.subplots(figsize=(8, 5.3))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 5.3)
    ax.axis("off")
    t = dict(fontsize=9, weight="bold", color=INK)

    # (a) skill node
    ax.text(0.05, 5.12, "(a) A skill node", **t)
    _box(ax, 0.05, 2.45, 2.75, 2.45, "", fill=BLUE_FILL, edge=BLUE)
    lines = [
        ("SkillNode {n, m}", "bold"),
        ("initiation I(s): when it may start", "normal"),
        ("goal β(s): when it has succeeded", "normal"),
        ("timeout: steps before it fails", "normal"),
        ("policy: 147-dim observation →", "normal"),
        ("   an action (primitive, n = 0)", "normal"),
        ("   or a child index (composite)", "normal"),
        ("children: earlier FROZEN nodes", "normal"),
        ("status: DECLARED → TRAINING", "normal"),
        ("   → STABLE → FROZEN", "normal"),
    ]
    for i, (s, w) in enumerate(lines):
        ax.text(0.15, 4.72 - i * 0.225, s, fontsize=8, color=INK, weight=w, va="center")

    # (b) executor stack and env
    ax.text(3.0, 5.12, "(b) One executor, one active stack", **t)
    stack = ["root: {2,1} solve_doorkey", "{1,1} unlock_door", "{0,1} get_key (primitive)"]
    for i, s in enumerate(stack):
        _box(ax, 3.0, 4.3 - i * 0.6, 2.15, 0.45, s, size=8)
    _arrow(ax, (4.075, 4.3), (4.075, 4.15))
    _arrow(ax, (4.075, 3.7), (4.075, 3.55))
    _box(ax, 5.55, 3.25, 0.75, 1.0, "MiniGrid\nenv", fill=GREY_FILL, size=8)
    _arrow(ax, (5.15, 3.3), (5.55, 3.45))
    ax.text(5.2, 3.08, "action", fontsize=7.5, color=INK_2)
    _arrow(ax, (5.55, 4.1), (5.15, 4.5))
    ax.text(5.3, 4.5, "observation", fontsize=7.5, color=INK_2)
    ax.text(3.0, 2.95, "Every env step: goal and timeout are\nchecked top-down; a finished "
            "node pops;\nselection descends until a primitive\nis on top, and it acts.",
            fontsize=7.8, color=INK_2, va="top", linespacing=1.3)

    # (d) recognition
    ax.text(6.5, 5.12, "(d) Recognition", **t)
    _box(ax, 6.5, 4.0, 1.45, 0.9, "Autoencoder:\nnew task?\n(not built)", ls="--", size=8)
    _box(ax, 6.5, 2.45, 1.45, 1.3, "GRU: which\nknown task?\n→ root node\n(registry\nnot built)",
         size=8)

    # (c) training
    ax.text(0.05, 2.05, "(c) Training one new node; all others stay frozen", **t)
    _box(ax, 0.05, 0.1, 3.2, 1.75,
         "Primitive: PPO on SubgoalEnv\n"
         "Composite: MaskablePPO on OptionEnv;\n"
         "one step = one whole child run (SMDP)\n"
         "Reward from its own goal:\n"
         "1 − 0.9·steps/timeout on success\n"
         "Mask: a child is selectable\n"
         "unless its goal already holds", size=7.8)
    _box(ax, 3.5, 0.1, 2.6, 1.75,
         "Stabilisation (decision 0004):\n"
         "SR ≥ 0.90 and Wilson 95 %\n"
         "lower bound ≥ 0.80 in two\n"
         "consecutive 50-episode\n"
         "evaluations on fixed seeds;\n"
         "score quoted on held-out seeds", size=7.8)
    _arrow(ax, (3.25, 0.97), (3.5, 0.97))
    _box(ax, 6.35, 0.1, 1.6, 1.75, "FROZEN:\nno gradients,\nforward pass\nstill runs;\n"
         "reusable as\na child", fill=BLUE_FILL, edge=BLUE, size=7.8)
    _arrow(ax, (6.1, 0.97), (6.35, 0.97))

    _save(fig, out, "fig1_v2_architecture")


# --------------------------------------------------------------------------- Fig. 2
def _meta(dirname: str) -> dict:
    return json.loads((RUNS / dirname / "meta.json").read_text(encoding="utf-8"))


def fig2_skill_graph(out: Path) -> None:
    built = {
        "get_key": "0-1_get_key",
        "open_door": "0-2_open_door",
        "go_to_goal": "0-3_go_to_goal",
        "get_key_yellow": "0-6_get_key_yellow",
        "get_key_blue": "0-7_get_key_blue",
        "open_door_tk": "0-8_open_door_tk",
        "drop_held": "0-9_drop_held",
        "unlock_door": "1-1_unlock_door",
        "solve_doorkey": "2-1_solve_doorkey",
        "unlock_matching": "1-2_unlock_matching",
        "pickup_box": "0-4_pickup_box",
        "unlock_pickup": "2-2_unlock_pickup",
    }
    metas = {k: _meta(v) for k, v in built.items()}
    frozen = [k for k, m in metas.items() if m["status"] == "frozen"]
    assert len(frozen) == 11, frozen

    # (x, y) positions in inches at page width; y = level n
    dx = 0.93
    pos = {
        "get_key": (0 * dx, 0), "open_door": (1 * dx, 0), "go_to_goal": (2 * dx, 0),
        "drop_held": (3 * dx, 0), "pickup_box": (4 * dx, 0), "move_blocker": (5 * dx, 0),
        "get_key_yellow": (6.2 * dx, 0), "get_key_blue": (7.2 * dx, 0),
        "open_door_tk": (8.2 * dx, 0),
        "unlock_door": (0.5 * dx, 1), "unlock_matching": (7.2 * dx, 1),
        "solve_doorkey": (1.25 * dx, 2), "unlock_pickup": (3.0 * dx, 2),
        "blocked_pickup_a": (5.25 * dx, 2), "blocked_pickup_b": (3.9 * dx, 3),
    }
    labels = {
        "move_blocker": "move_blocker\n{0,5}\nH-7b",
        "blocked_pickup_a": "blocked_a\n{2,3}\nH-7b",
        "blocked_pickup_b": "blocked_b\n{3,1}\nH-7b",
    }
    edges = [
        ("unlock_door", "get_key"), ("unlock_door", "open_door"),
        ("solve_doorkey", "unlock_door"), ("solve_doorkey", "go_to_goal"),
        ("unlock_matching", "get_key_yellow"), ("unlock_matching", "get_key_blue"),
        ("unlock_matching", "open_door_tk"),
        ("unlock_pickup", "unlock_door"), ("unlock_pickup", "drop_held"),
        ("unlock_pickup", "pickup_box"),
        ("blocked_pickup_a", "move_blocker"), ("blocked_pickup_a", "unlock_door"),
        ("blocked_pickup_a", "pickup_box"),
        ("blocked_pickup_b", "move_blocker"), ("blocked_pickup_b", "unlock_pickup"),
    ]
    for k, m in metas.items():
        nm = m["id"].split(" ", 1)
        hs = m.get("heldout")
        if hs:
            labels[k] = f"{nm[1]}\n{nm[0]}\n{hs['successes']}/{hs['n_episodes']}"
        else:  # frozen under decision 0014: its argmax handoff row instead of a held-out score
            row = m["acceptance"]["b_argmax_handoff"]["successes"]
            labels[k] = f"{nm[1]}\n{nm[0]} (0014)\n{row}*"

    def style(k):
        if k in frozen:
            return dict(fill=BLUE_FILL, edge=BLUE, ls="-", lw=1.4)
        if k == "unlock_matching":
            return dict(fill="white", edge=MUTED, ls=":", lw=1.4)
        return dict(fill="white", edge=INK_2, ls="--", lw=1.2)

    w, h = 0.9, 0.62
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.set_xlim(-1.15, 8.15)
    ax.set_ylim(-0.4, 3.5)
    ax.axis("off")
    for lvl in range(4):
        ax.text(-1.12, lvl, f"level {lvl}", fontsize=7.5, color=MUTED, va="center")
    for a, b in edges:
        (xa, ya), (xb, yb) = pos[a], pos[b]
        planned = a not in frozen or b not in frozen
        ax.add_patch(FancyArrowPatch((xa, ya - h / 2), (xb, yb + h / 2), arrowstyle="-|>",
                                     mutation_scale=9, lw=1.0,
                                     color=MUTED if planned else INK_2,
                                     ls="--" if planned else "-", shrinkA=1, shrinkB=1))
    for k, (x, y) in pos.items():
        s = style(k)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc=s["fill"], ec=s["edge"], ls=s["ls"], lw=s["lw"]))
        ax.text(x, y, labels[k], ha="center", va="center", fontsize=6.0, color=INK,
                linespacing=1.15)
    handles = [
        FancyBboxPatch((0, 0), 1, 1, fc=BLUE_FILL, ec=BLUE, lw=1.4,
                       label=f"FROZEN, registered ({len(frozen)} nodes)"),
        FancyBboxPatch((0, 0), 1, 1, fc="white", ec=INK_2, ls="--", lw=1.2,
                       label="planned or not registered"),
        FancyBboxPatch((0, 0), 1, 1, fc="white", ec=MUTED, ls=":", lw=1.4,
                       label="trained, not STABLE"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=7.5)
    _save(fig, out, "fig2_skill_graph")


# --------------------------------------------------------------------------- Fig. 3
def _sampled_rows(files, row_key):
    rates = {}
    for f in files:
        d = json.loads((RUNS / f).read_text(encoding="utf-8"))
        for seed, m in d["models"].items():
            if seed == "42":  # exploratory, counts toward nothing
                continue
            rows = m[row_key]
            rates[int(seed)] = {
                "own": rows["own_hook"]["sampled"]["success_rate"],
                "handoff": rows["handoff"]["sampled"]["success_rate"],
                "passes": rows["passes"] if "passes" in rows else m["passes"],
            }
    return dict(sorted(rates.items()))


def fig3_pickup_box(out: Path) -> None:
    spec2 = _sampled_rows(["w1a_pickup_box_sampled.json", "w1a_pickup_box_sampled_ext.json"],
                          "rows")
    spec3 = _sampled_rows(["w1a_spec3_sampled.json", "w1a_spec3_sampled_ext.json"], "test")
    assert len(spec2) == len(spec3) == 10
    n2 = sum(v["passes"] for v in spec2.values())
    n3 = sum(v["passes"] for v in spec3.values())

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    panels = [
        (axes[0], spec2, f"Spec 2 (W1a): {n2}/10 seeds pass (rule ≥ 7)"),
        (axes[1], spec3, f"Spec 3: {n3}/10 seeds pass"),
    ]
    for ax, data, title in panels:
        seeds = list(data)
        xs = range(len(seeds))
        ax.axhline(0.90, color=INK_2, lw=1.0, ls="--", zorder=1)
        ax.text(len(seeds) - 0.5, 0.902, "0.90", fontsize=8, color=INK_2, ha="right",
                va="bottom")
        for i, s in enumerate(seeds):
            if not data[s]["passes"]:
                ax.axvspan(i - 0.45, i + 0.45, color=GREY_FILL, zorder=0, lw=0)
        ax.scatter([x - 0.12 for x in xs], [data[s]["own"] for s in seeds], s=42,
                   marker="o", color=BLUE, edgecolor="white", linewidth=1.0, zorder=3)
        ax.scatter([x + 0.12 for x in xs], [data[s]["handoff"] for s in seeds], s=42,
                   marker="s", color=ORANGE, edgecolor="white", linewidth=1.0, zorder=3)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"{s}\n{'✓' if data[s]['passes'] else '✗'}"
                            for s in seeds], fontsize=8)
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.set_xlabel("training seed")
        ax.set_ylim(0.75, 1.01)
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("sampled success rate\n(500 fresh episodes per row)")
    handles = [
        Line2D([], [], marker="o", ls="", color=BLUE, markersize=7,
               label="own-hook row"),
        Line2D([], [], marker="s", ls="", color=ORANGE, markersize=7,
               label="handoff row (after unlock_door → drop_held)"),
        Line2D([], [], ls="--", color=INK_2, label="row passes at ≥ 0.90"),
    ]
    # Shading, seed ranges and "sampled only" are stated in the report's caption.
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.1))
    _save(fig, out, "fig3_pickup_box_spec2_vs_spec3")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=ROOT / "reports" / "progress2" / "figures")
    args = p.parse_args()
    fig1_architecture(args.out)
    fig2_skill_graph(args.out)
    fig3_pickup_box(args.out)


if __name__ == "__main__":
    main()
