"""Figures for ENS 492 Progress Report II (PNG + PDF).

Reads only files already in ``runs/v2/``; trains and evaluates nothing.
The PR-I version is ``scripts/generate_report_figures.py`` and is left as it is.

    python scripts/generate_report_figures_pr2.py [--out reports/progress2/figures]

Every figure is drawn at the report's text width (6.3 in) in Times New Roman, so it is placed
in the document at 100 % and its font sizes are the printed sizes (8-10 pt).

Fig. 1  v2 architecture (drawn; no data).
Fig. 2  the skill graph as built: FROZEN nodes from ``runs/v2/*/meta.json``, levels as
        columns; planned BlockedUnlockPickup nodes (v2 §11, T8-T9b) dashed.
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "v2"
WIDTH = 6.3  # report text width, inches

INK = "#000000"
INK_2 = "#404040"
GRID = "#d9d9d9"
FROZEN_FILL = "#dbe8f6"
FROZEN_EDGE = "#1f4e8c"
PLAN_EDGE = "#595959"
BLUE = "#1f5fbf"
ORANGE = "#d9541e"
SHADE = "#ececec"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.8,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.titlecolor": INK,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    }
)


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02,
                    facecolor="white")
    plt.close(fig)
    print(f"wrote {out / name}.png/.pdf")


def _box(ax, x, y, w, h, text="", *, fill="white", edge=INK, ls="-", lw=0.9, size=9,
         weight="normal", ha="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", fc=fill, ec=edge,
                                ls=ls, lw=lw))
    if text:
        tx = x + w / 2 if ha == "center" else x + 0.07
        ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=size, weight=weight,
                color=INK, linespacing=1.25)


def _arrow(ax, a, b, *, ls="-", color=INK, lw=0.9, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, color=color, lw=lw,
                                 ls=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0,
                                 shrinkB=0))


def _panel(ax, x, y, text):
    ax.text(x, y, text, fontsize=10, weight="bold", ha="left", va="bottom", color=INK)


# --------------------------------------------------------------------------- Fig. 1
def fig1_architecture(out: Path) -> None:
    H = 4.1
    fig = plt.figure(figsize=(WIDTH, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, H)
    ax.axis("off")

    # (a) skill node
    _panel(ax, 0.0, 3.85, "(a) Skill node")
    _box(ax, 0.0, 2.25, 2.0, 1.5, fill=FROZEN_FILL, edge=FROZEN_EDGE)
    ax.text(1.0, 3.55, r"Skill node $\{n,m\}$", ha="center", va="center", fontsize=9.5,
            weight="bold")
    rows = [(r"$I(s)$", "initiation condition"), (r"$\beta(s)$", "goal condition"),
            (r"$T$", "timeout"), (r"$\pi$", "policy (147-dim input)"),
            (r"$C$", "children: earlier frozen nodes")]
    for i, (sym, txt) in enumerate(rows):
        yy = 3.27 - i * 0.22
        ax.text(0.12, yy, sym, fontsize=9, va="center")
        ax.text(0.5, yy, txt, fontsize=9, va="center")
    # life cycle, as text (boxes were too narrow for the state names)
    ax.text(1.0, 1.98, "DECLARED $\\rightarrow$ TRAINING $\\rightarrow$\nSTABLE $\\rightarrow$ FROZEN",
            ha="center", va="center", fontsize=8.5, linespacing=1.3)
    ax.text(1.0, 1.66, "life cycle of a node", ha="center", va="center", fontsize=8,
            color=INK_2, style="italic")

    # (b) execution
    _panel(ax, 2.45, 3.85, "(b) Execution: one active stack")
    stack = [("root (DoorKey task)", "level 2"), ("composite (unlock door)", "level 1"),
             ("primitive (get key)", "level 0")]
    bx, bw_, bh_, step = 2.95, 1.6, 0.34, 0.62
    for i, (t, lv) in enumerate(stack):
        y0 = 3.3 - i * step
        _box(ax, bx, y0, bw_, bh_, t, size=9)
        ax.text(bx - 0.06, y0 + bh_ / 2, lv, fontsize=8, ha="right", va="center", color=INK_2)
        if i:
            top = y0 + step
            _arrow(ax, (bx + bw_ / 2, top), (bx + bw_ / 2, y0 + bh_))
            ax.text(bx + bw_ / 2 + 0.07, (top + y0 + bh_) / 2, "selects a child", fontsize=8,
                    color=INK_2, va="center", style="italic")
    ex, ew = 5.3, 0.98
    _box(ax, ex, 2.06, ew, 1.58, "MiniGrid\nenvironment", fill="#f2f2f2", size=9)
    y_root, y_prim = 3.3 + bh_ / 2, 3.3 - 2 * step + bh_ / 2
    _arrow(ax, (ex, y_root), (bx + bw_, y_root))
    ax.text((bx + bw_ + ex) / 2, y_root + 0.05, "observation", fontsize=8, ha="center",
            va="bottom")
    _arrow(ax, (bx + bw_, y_prim), (ex, y_prim))
    ax.text((bx + bw_ + ex) / 2, y_prim + 0.05, "action", fontsize=8, ha="center", va="bottom")
    ax.text(2.45, 1.9, "Every environment step, goal and timeout are checked top-down;\n"
            "a finished node is popped and selection descends until a primitive acts.",
            fontsize=8, color=INK_2, va="top", linespacing=1.25)

    # (c) training pipeline
    _panel(ax, 0.0, 1.15, "(c) Adding one task to the library")
    steps = [("Train the new node", "primitive: PPO\ncomposite: masked PPO\n(one step = one child run)"),
             ("Stabilisation rule", "success $\\geq$ 0.90 and Wilson\nlower bound $\\geq$ 0.80,\ntwice (decision 0004)"),
             ("Freeze", "no gradients;\nforward pass\nstill runs"),
             ("Reuse", "child of any\nlater composite")]
    xs = [0.0, 1.72, 3.44, 4.88]
    ws = [1.52, 1.52, 1.24, 1.42]
    for (title, body), x0, w in zip(steps, xs, ws):
        fill = FROZEN_FILL if title in ("Freeze", "Reuse") else "white"
        edge = FROZEN_EDGE if title in ("Freeze", "Reuse") else INK
        _box(ax, x0, 0.05, w, 0.98, fill=fill, edge=edge)
        ax.text(x0 + w / 2, 0.86, title, ha="center", va="center", fontsize=9, weight="bold")
        ax.text(x0 + w / 2, 0.42, body, ha="center", va="center", fontsize=8, linespacing=1.2)
    for x0, w, nxt in zip(xs[:-1], ws[:-1], xs[1:]):
        _arrow(ax, (x0 + w, 0.54), (nxt, 0.54))

    _save(fig, out, "fig1_v2_architecture")


# --------------------------------------------------------------------------- Fig. 2
def _meta(dirname: str) -> dict:
    return json.loads((RUNS / dirname / "meta.json").read_text(encoding="utf-8"))


def fig2_skill_graph(out: Path) -> None:
    built = {
        "get_key": "0-1_get_key", "open_door": "0-2_open_door", "go_to_goal": "0-3_go_to_goal",
        "get_key_yellow": "0-6_get_key_yellow", "get_key_blue": "0-7_get_key_blue",
        "open_door_tk": "0-8_open_door_tk", "drop_held": "0-9_drop_held",
        "pickup_box": "0-4_pickup_box", "unlock_door": "1-1_unlock_door",
        "unlock_matching": "1-2_unlock_matching", "solve_doorkey": "2-1_solve_doorkey",
        "unlock_pickup": "2-2_unlock_pickup",
    }
    metas = {k: _meta(v) for k, v in built.items()}
    frozen = [k for k, m in metas.items() if m["status"] == "frozen"]
    assert len(frozen) == 11, frozen

    # levels as columns (left = level 0); y in row units, top = high
    col_x = [0.0, 1.6, 3.2, 4.8]
    bw, bh, row = 1.38, 0.34, 0.42
    pos = {
        "get_key": (0, 9), "open_door": (0, 8), "go_to_goal": (0, 7), "drop_held": (0, 6),
        "pickup_box": (0, 5), "move_blocker": (0, 4),
        "get_key_yellow": (0, 2.4), "get_key_blue": (0, 1.4), "open_door_tk": (0, 0.4),
        "unlock_door": (1, 8.5), "unlock_matching": (1, 1.4),
        "solve_doorkey": (2, 7.75), "unlock_pickup": (2, 6.0), "blocked_pickup_a": (2, 4.2),
        "blocked_pickup_b": (3, 3.0),
    }
    planned = {"move_blocker": ("move_blocker", "{0,5}", "planned"),
               "blocked_pickup_a": ("blocked_pickup_a", "{2,3}", "planned (T9a)"),
               "blocked_pickup_b": ("blocked_pickup_b", "{3,1}", "planned (T9b)")}
    edges = [
        ("unlock_door", "get_key"), ("unlock_door", "open_door"),
        ("solve_doorkey", "unlock_door"), ("solve_doorkey", "go_to_goal"),
        ("unlock_pickup", "unlock_door"), ("unlock_pickup", "drop_held"),
        ("unlock_pickup", "pickup_box"),
        ("unlock_matching", "get_key_yellow"), ("unlock_matching", "get_key_blue"),
        ("unlock_matching", "open_door_tk"),
        ("blocked_pickup_a", "move_blocker"), ("blocked_pickup_a", "unlock_door"),
        ("blocked_pickup_a", "pickup_box"),
        ("blocked_pickup_b", "move_blocker"), ("blocked_pickup_b", "unlock_pickup"),
    ]

    def label(k):
        if k in planned:
            return planned[k]
        m = metas[k]
        nm, name = m["id"].split(" ", 1)
        hs = m.get("heldout")
        if hs:
            return name, nm, f"{hs['successes']}/{hs['n_episodes']}"
        row_ = m["acceptance"]["b_argmax_handoff"]["successes"]  # frozen under decision 0014
        return name, nm, f"{row_}*"

    def xy(k):
        c, r = pos[k]
        return col_x[c], r * row

    H = 10 * row + 0.25
    fig = plt.figure(figsize=(WIDTH, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-0.02, WIDTH)
    ax.set_ylim(-0.2, H - 0.2)
    ax.axis("off")
    for c, x0 in enumerate(col_x):
        ax.text(x0 + bw / 2, 9.75 * row, f"Level {c}", ha="center", va="bottom", fontsize=9.5,
                weight="bold")
    # spread several edges over a box side so they do not meet in one point
    def ports(groups, key_other):
        off = {}
        for node, lst in groups.items():
            lst = sorted(lst, key=lambda e: -xy(key_other(e))[1])
            n = len(lst)
            for i, e in enumerate(lst):
                off[e] = 0.0 if n == 1 else (0.5 - i / (n - 1)) * bh * 0.6
        return off

    outgoing, incoming = {}, {}
    for e in edges:
        outgoing.setdefault(e[0], []).append(e)
        incoming.setdefault(e[1], []).append(e)
    out_off = ports(outgoing, lambda e: e[1])
    in_off = ports(incoming, lambda e: e[0])
    for a, b_ in edges:
        xa, ya = xy(a)
        xb, yb = xy(b_)
        ya += out_off[(a, b_)]
        yb += in_off[(a, b_)]
        dashed = a not in frozen or b_ not in frozen
        ax.add_patch(FancyArrowPatch((xa, ya), (xb + bw, yb), arrowstyle="-|>",
                                     mutation_scale=7, lw=0.8, shrinkA=0, shrinkB=0,
                                     color=PLAN_EDGE if dashed else INK,
                                     ls=(0, (3, 2)) if dashed else "-", zorder=1))
    for k in pos:
        x0, y0 = xy(k)
        if k in frozen:
            fc, ec, ls = FROZEN_FILL, FROZEN_EDGE, "-"
        elif k == "unlock_matching":
            fc, ec, ls = "white", PLAN_EDGE, ":"
        else:
            fc, ec, ls = "white", PLAN_EDGE, (0, (3, 2))
        ax.add_patch(FancyBboxPatch((x0, y0 - bh / 2), bw, bh, boxstyle="square,pad=0",
                                    fc=fc, ec=ec, ls=ls, lw=0.9, zorder=2))
        name, nm, score = label(k)
        ax.text(x0 + 0.06, y0 + 0.075, name, fontsize=8.5, va="center", ha="left", zorder=3)
        ax.text(x0 + 0.06, y0 - 0.085, nm, fontsize=8, va="center", ha="left", zorder=3,
                color=INK_2)
        ax.text(x0 + bw - 0.06, y0 - 0.085, score, fontsize=8, va="center", ha="right",
                zorder=3, color=INK_2)
    handles = [
        Patch(fc=FROZEN_FILL, ec=FROZEN_EDGE, lw=0.9, label=f"frozen, registered ({len(frozen)})"),
        Patch(fc="white", ec=PLAN_EDGE, ls=(0, (3, 2)), lw=0.9, label="planned (BlockedUnlockPickup)"),
        Patch(fc="white", ec=PLAN_EDGE, ls=":", lw=0.9, label="trained, not stable"),
    ]
    # legend and note in the empty lower right (levels 2-3, below the Blocked nodes)
    leg_x, leg_y = col_x[2], 2.2 * row
    ax.legend(handles=handles, loc="upper left", fontsize=8, handlelength=1.8,
              borderaxespad=0, bbox_to_anchor=(leg_x, leg_y), bbox_transform=ax.transData)
    ax.text(leg_x, 0.85 * row, "Right-hand number: held-out successes of 50 (argmax).\n"
            "* Frozen under decision 0014: deterministic success\n"
            "from the UnlockPickup handoff on 488 fresh episodes.",
            fontsize=7.5, va="top", ha="left", color=INK_2, linespacing=1.2)
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

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9), sharey=True)
    fig.subplots_adjust(left=0.1, right=0.995, top=0.8, bottom=0.15, wspace=0.08)
    panels = [(axes[0], spec2, f"(a) Specification 2 (W1a): {n2} of 10 seeds pass"),
              (axes[1], spec3, f"(b) Specification 3: {n3} of 10 seeds pass")]
    for ax, data, title in panels:
        seeds = list(data)
        xs = range(len(seeds))
        for i, s in enumerate(seeds):
            if not data[s]["passes"]:
                ax.axvspan(i - 0.45, i + 0.45, color=SHADE, zorder=0, lw=0)
        ax.axhline(0.90, color=INK, lw=0.8, ls="--", zorder=1)
        ax.scatter([x - 0.13 for x in xs], [data[s]["own"] for s in seeds], s=22, marker="o",
                   color=BLUE, zorder=3, lw=0)
        ax.scatter([x + 0.13 for x in xs], [data[s]["handoff"] for s in seeds], s=22,
                   marker="s", color=ORANGE, zorder=3, lw=0)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([str(s) for s in seeds], fontsize=8.5)
        ax.set_xlim(-0.6, len(seeds) - 0.4)
        ax.set_ylim(0.75, 1.01)
        ax.set_title(title, fontsize=9.5, loc="left", pad=4)
        ax.set_xlabel("Training seed", fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8.5, length=2.5, width=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Sampled success rate", fontsize=9)
    handles = [
        Line2D([], [], marker="o", ls="", color=BLUE, markersize=5, label="own-hook row"),
        Line2D([], [], marker="s", ls="", color=ORANGE, markersize=5, label="handoff row"),
        Line2D([], [], ls="--", color=INK, lw=0.8, label="pass threshold 0.90"),
        Patch(fc=SHADE, ec="none", label="seed fails"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8.5,
               bbox_to_anchor=(0.55, 1.0), handletextpad=0.4, columnspacing=1.4)
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
