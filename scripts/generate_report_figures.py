"""
Generate all Progress Report I figures.
Output: reports/progress1/figures/

Run from repo root:
    conda activate ens491
    python scripts/generate_report_figures.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

OUT = Path("reports/progress1/figures")
OUT.mkdir(parents=True, exist_ok=True)

BLUE   = "#2E4A7A"
ORANGE = "#D4651A"
GREEN  = "#2E7A4A"
GRAY   = "#555555"
LGRAY  = "#CCCCCC"
RED    = "#8B1A1A"
PURPLE = "#5B2D8E"

FONT = {"family": "DejaVu Sans"}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Overall Architecture / Pipeline Diagram
# ─────────────────────────────────────────────────────────────────────────────

def fig1_architecture():
    # ── Canvas ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 6.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # ── Layout constants ─────────────────────────────────────────────────────
    Y   = 2.9        # main pipeline centre-y
    BH  = 0.95       # box height
    # Box centres and half-widths along the main pipeline
    specs = {
        "obs": (0.9,  0.72),   # cx, half-w
        "ae":  (2.75, 0.78),
        "dia": (4.5,  0.0),    # diamond; handled separately
        "gru": (6.1,  0.70),
        "tlm": (8.35, 1.10),
        "mc":  (10.8, 0.97),
        "opt": (12.65,0.68),
        "col": (14.45,0.72),
    }
    # Diamond geometry
    DX, DY = 4.5, Y
    DHW, DHH = 0.42, 0.33   # half-width, half-height

    # Novel task box (centred above the diamond)
    NX, NY      = 4.5, 5.05
    NHW, NHH    = 1.85, 0.70   # half-width, half-height (full w=3.7, h=1.4)

    # ── Helpers ──────────────────────────────────────────────────────────────
    def rbox(cx, cy, hw, hh, title, sub=None, color=BLUE, fs=9.5):
        """Rounded rectangle with title and optional italic sub-label."""
        rect = FancyBboxPatch(
            (cx - hw, cy - hh), 2*hw, 2*hh,
            boxstyle="round,pad=0.07", lw=1.6,
            edgecolor=color, facecolor="white", zorder=3
        )
        ax.add_patch(rect)
        ty = cy + hh*0.25 if sub else cy
        ax.text(cx, ty, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=color, zorder=4)
        if sub:
            ax.text(cx, cy - hh*0.35, sub, ha="center", va="center",
                    fontsize=7.2, color=GRAY, style="italic",
                    multialignment="center", zorder=4)

    def arr(x0, y0, x1, y1, color=GRAY, lw=1.5):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=lw, mutation_scale=14), zorder=2)

    def elabel(x, y, txt, color=GRAY, ha="center", va="bottom"):
        ax.text(x, y, txt, ha=ha, va=va, fontsize=7.2,
                color=color, style="italic", zorder=5)

    # ── Decision diamond ─────────────────────────────────────────────────────
    def diamond(cx, cy, hw, hh, label, color=ORANGE):
        pts = np.array([[cx, cy+hh], [cx+hw, cy], [cx, cy-hh], [cx-hw, cy]])
        d = plt.Polygon(pts, closed=True, lw=1.6,
                        edgecolor=color, facecolor="white", zorder=3)
        ax.add_patch(d)
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color=color, zorder=4)

    # ── 1. Pipeline boxes ────────────────────────────────────────────────────
    def pb(key, title, sub=None, color=BLUE, fs=9.5):
        cx, hw = specs[key]
        rbox(cx, Y, hw, BH/2, title, sub, color=color, fs=fs)

    pb("obs", "Observation", "obs ∈ ℝ¹⁴⁷",   color=GRAY,   fs=9)
    pb("ae",  "Autoencoder", "novelty detector", color=ORANGE)
    diamond(DX, DY, DHW, DHH, "novel?", color=ORANGE)
    pb("gru", "GRU",                 "task identifier",   color=PURPLE)
    pb("tlm", "TaskLifecycleManager","single entry point", color=BLUE, fs=8.5)
    pb("mc",  "MetaController",      "ε-greedy selection", color=BLUE)
    pb("opt", "Option",              "wrapper",            color=GREEN)
    pb("col", "PN Column",           "{n, m}",             color=ORANGE)

    # Novel task box (centred on diamond x, well above)
    rbox(NX, NY, NHW, NHH,
         "on_novel_task_detected()",
         "open Column → train → freeze\n→ add_option()",
         color=RED, fs=8.8)

    # ── 2. Main pipeline arrows ───────────────────────────────────────────────
    # Obs → AE
    arr(specs["obs"][0]+specs["obs"][1], Y,  specs["ae"][0]-specs["ae"][1],   Y)
    elabel((specs["obs"][0]+specs["obs"][1]+specs["ae"][0]-specs["ae"][1])/2,
           Y+0.14, "obs_t")

    # AE → diamond left tip
    arr(specs["ae"][0]+specs["ae"][1], Y,  DX-DHW, Y)

    # Diamond right tip → GRU  ("False" branch)
    arr(DX+DHW, Y,  specs["gru"][0]-specs["gru"][1], Y)
    elabel((DX+DHW + specs["gru"][0]-specs["gru"][1])/2, Y+0.14, "False")

    # GRU → TLM
    arr(specs["gru"][0]+specs["gru"][1], Y,  specs["tlm"][0]-specs["tlm"][1], Y)
    elabel((specs["gru"][0]+specs["gru"][1]+specs["tlm"][0]-specs["tlm"][1])/2,
           Y+0.14, "task_id")

    # TLM → MC
    arr(specs["tlm"][0]+specs["tlm"][1], Y,  specs["mc"][0]-specs["mc"][1],  Y)

    # MC → Option
    arr(specs["mc"][0]+specs["mc"][1],  Y,  specs["opt"][0]-specs["opt"][1], Y)

    # Option → PN Column
    arr(specs["opt"][0]+specs["opt"][1], Y,  specs["col"][0]-specs["col"][1], Y)

    # PN Column → "action"
    col_right = specs["col"][0] + specs["col"][1]
    arr(col_right, Y,  col_right + 0.45, Y)
    ax.text(col_right + 0.55, Y, "action", ha="left", va="center",
            fontsize=9, color=GRAY, fontweight="bold", zorder=5)

    # ── 3. Novel-task branch: diamond top → novel box bottom ─────────────────
    arr(DX, DY+DHH,  NX, NY-NHH,  color=RED, lw=1.6)
    # "True" label — to the right of the vertical arrow, at mid-height
    mid_y = (DY+DHH + NY-NHH) / 2
    elabel(DX+0.15, mid_y, "True", color=RED, ha="left", va="center")

    # ── 4. Lateral connection (dashed, into PN Column top) ───────────────────
    col_cx, col_hw = specs["col"]
    lat_src_x = col_cx - 1.6
    lat_src_y = Y + BH/2 + 0.55
    ax.annotate("", xy=(col_cx, Y + BH/2),
                xytext=(lat_src_x, lat_src_y),
                arrowprops=dict(arrowstyle="-|>", color=LGRAY, lw=1.1,
                                linestyle="dashed", mutation_scale=11), zorder=2)
    ax.text(lat_src_x - 0.05, lat_src_y + 0.12,
            "lateral from {n, m-1}",
            ha="center", va="bottom", fontsize=7, color=GRAY, style="italic")

    # ── 5. Sub-layer recursion note (below MC) ────────────────────────────────
    mc_cx = specs["mc"][0]
    ax.annotate("", xy=(mc_cx, Y - BH/2 - 0.55),
                xytext=(mc_cx, Y - BH/2),
                arrowprops=dict(arrowstyle="-|>", color=LGRAY, lw=1.1,
                                linestyle="dashed", mutation_scale=11), zorder=2)
    ax.text(mc_cx, Y - BH/2 - 0.65,
            "sub_layer = MetaController(…)  (non-leaf: recurse deeper)",
            ha="center", va="top", fontsize=7, color=GRAY, style="italic")

    # ── 6. Legend ────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(facecolor="white", edgecolor=ORANGE, label="Task Detection (AE, PN Column)"),
        mpatches.Patch(facecolor="white", edgecolor=PURPLE, label="GRU Task Identifier"),
        mpatches.Patch(facecolor="white", edgecolor=BLUE,   label="Options Layer (TLM, MC)"),
        mpatches.Patch(facecolor="white", edgecolor=GREEN,  label="Option Wrapper"),
        mpatches.Patch(facecolor="white", edgecolor=RED,    label="Novel Task Branch"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=7.5, frameon=True,
              framealpha=0.95, edgecolor=LGRAY, ncol=3,
              bbox_to_anchor=(0.0, 0.0))

    ax.set_title(
        "Figure 1 — System Architecture: Observation-to-Action Pipeline",
        fontsize=12, fontweight="bold", pad=8, color=BLUE
    )

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig1_architecture.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / "fig1_architecture.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Fig 1: Architecture Diagram")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — PN-4 Training Curves
# ─────────────────────────────────────────────────────────────────────────────

COL1_DATA = [
    (20000, 0.89),
    (40000, 0.96),
]
COL1_STABLE_STEP = 40000
COL1_STABLE_MEAN = 0.958

COL2_DATA = [
    (20000, 0.02), (40000, 0.01), (60000, 0.01), (80000, 0.03),
    (100000, 0.05), (120000, 0.05), (140000, 0.09), (160000, 0.06),
    (180000, 0.12), (200000, 0.12), (220000, 0.12), (240000, 0.09),
    (260000, 0.10), (280000, 0.15), (300000, 0.13), (320000, 0.08),
    (340000, 0.25), (360000, 0.22), (380000, 0.17), (400000, 0.17),
    (420000, 0.20), (440000, 0.21), (460000, 0.35), (480000, 0.21),
    (500000, 0.14), (520000, 0.15), (540000, 0.16), (560000, 0.27),
    (580000, 0.24), (600000, 0.38), (620000, 0.33), (640000, 0.29),
    (660000, 0.32), (680000, 0.26), (700000, 0.27), (720000, 0.17),
    (740000, 0.36), (760000, 0.36), (780000, 0.35), (800000, 0.31),
    (820000, 0.36), (840000, 0.41), (860000, 0.36), (880000, 0.34),
    (900000, 0.35), (920000, 0.40), (940000, 0.33), (960000, 0.39),
    (980000, 0.34), (1000000, 0.36), (1020000, 0.33), (1040000, 0.35),
    (1060000, 0.30), (1080000, 0.41), (1100000, 0.35), (1120000, 0.38),
    (1140000, 0.39), (1160000, 0.40), (1180000, 0.37), (1200000, 0.46),
    (1220000, 0.24), (1240000, 0.35), (1260000, 0.43), (1280000, 0.39),
    (1300000, 0.49), (1320000, 0.32), (1340000, 0.44), (1360000, 0.38),
    (1380000, 0.48), (1400000, 0.33), (1420000, 0.37), (1440000, 0.34),
    (1460000, 0.28), (1480000, 0.36), (1500000, 0.30), (1520000, 0.42),
    (1540000, 0.33), (1560000, 0.34), (1580000, 0.19), (1600000, 0.22),
    (1620000, 0.31), (1640000, 0.39), (1660000, 0.27), (1680000, 0.37),
    (1700000, 0.27), (1720000, 0.24), (1740000, 0.32), (1760000, 0.32),
    (1780000, 0.21), (1800000, 0.28), (1820000, 0.31), (1840000, 0.23),
    (1860000, 0.35), (1880000, 0.27), (1900000, 0.36), (1920000, 0.23),
    (1940000, 0.25), (1960000, 0.30), (1980000, 0.24), (2000000, 0.24),
]


def rolling_mean(vals, window=7):
    result = []
    for i in range(len(vals)):
        lo = max(0, i - window // 2)
        hi = min(len(vals), i + window // 2 + 1)
        result.append(np.mean(vals[lo:hi]))
    return np.array(result)


def fig2_training_curves():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")

    # ── Left: Column {0,1} on Empty-8x8 ─────────────────────────────────────
    steps1 = [d[0] for d in COL1_DATA]
    rewards1 = [d[1] for d in COL1_DATA]

    ax1.plot(steps1, rewards1, "o-", color=BLUE, linewidth=2, markersize=7,
             label="Mean episode reward (50-ep window)", zorder=3)
    ax1.axhline(0.85, color=ORANGE, linestyle="--", linewidth=1.5,
                label="Stabilization threshold (0.85)", zorder=2)
    ax1.axhline(0.05, color=LGRAY, linestyle=":", linewidth=1.0,
                label="Std threshold (0.05)", zorder=2)

    ax1.axvline(COL1_STABLE_STEP, color=GREEN, linestyle="-.", linewidth=1.5, zorder=2)
    ax1.scatter([COL1_STABLE_STEP], [COL1_STABLE_MEAN], marker="*", s=220,
                color=GREEN, zorder=4, label=f"Stabilized @ {COL1_STABLE_STEP//1000}k steps\n(mean = {COL1_STABLE_MEAN:.3f})")

    ax1.fill_between([COL1_STABLE_STEP, 45000], 0, 1.05,
                     alpha=0.08, color=GREEN, label="Stable region")

    ax1.set_xlim(0, 45000)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("Training Timesteps", fontsize=10)
    ax1.set_ylabel("Mean Episode Reward", fontsize=10)
    ax1.set_title("Column {0,1}  —  MiniGrid-Empty-8x8-v0", fontsize=11, fontweight="bold", color=BLUE)
    ax1.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    ax1.grid(axis="y", color=LGRAY, linewidth=0.5, alpha=0.7)

    # ── Right: Column {0,2} on FourRooms ─────────────────────────────────────
    steps2 = np.array([d[0] for d in COL2_DATA])
    rewards2 = np.array([d[1] for d in COL2_DATA])
    rolled = rolling_mean(rewards2, window=9)

    ax2.plot(steps2, rewards2, color=BLUE, linewidth=0.7, alpha=0.35, label="Mean reward (per chunk)")
    ax2.plot(steps2, rolled, color=BLUE, linewidth=2.0, label="Rolling average (9-point)", zorder=3)
    ax2.axhline(0.85, color=ORANGE, linestyle="--", linewidth=1.5,
                label="Stabilization threshold (0.85)", zorder=2)

    # Peak annotation
    peak_step = 1300000
    peak_val  = 0.49
    ax2.scatter([peak_step], [peak_val], marker="^", s=120, color=RED, zorder=5)
    ax2.annotate(f"Peak ≈ {peak_val:.2f}\n@ {peak_step//1000}k steps",
                 xy=(peak_step, peak_val), xytext=(peak_step - 300000, peak_val + 0.12),
                 fontsize=7.5, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))

    # Budget exhausted annotation
    ax2.axvline(2000000, color=GRAY, linestyle=":", linewidth=1.2, zorder=2)
    ax2.text(1970000, 0.78, "Budget\nexhausted", ha="right", fontsize=7.5,
             color=GRAY, style="italic")
    ax2.text(1970000, 0.68, "mean = 0.240", ha="right", fontsize=7.5,
             color=RED, fontweight="bold")

    # Did not stabilize badge
    ax2.text(0.98, 0.06, "DID NOT STABILIZE", transform=ax2.transAxes,
             ha="right", va="bottom", fontsize=8.5, color=RED, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF0F0", edgecolor=RED, alpha=0.9))

    ax2.set_xlim(0, 2_100_000)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Training Timesteps", fontsize=10)
    ax2.set_ylabel("Mean Episode Reward", fontsize=10)
    ax2.set_title("Column {0,2}  —  MiniGrid-FourRooms-v0  (lateral from {0,1})",
                  fontsize=11, fontweight="bold", color=BLUE)
    ax2.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1e6):.1f}M" if x >= 1e6 else f"{int(x/1000)}k"))
    ax2.grid(axis="y", color=LGRAY, linewidth=0.5, alpha=0.7)

    fig.suptitle("Figure 2 — PN-4 Training Curves: Progressive Network Column Training",
                 fontsize=12, fontweight="bold", color=BLUE, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_training_curves.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / "fig2_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Fig 2: PN-4 Training Curves")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Experiment Pipeline Infographic
# ─────────────────────────────────────────────────────────────────────────────

def fig3_experiment_pipeline():
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def stage(cx, cy, w, h, title, body_lines, color, fontsize=9):
        rect = FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle="round,pad=0.1", linewidth=1.8,
            edgecolor=color, facecolor="#F8F8FF" if color == BLUE else "#FFF8F0"
        )
        ax.add_patch(rect)
        ax.text(cx, cy + h/2 - 0.28, title, ha="center", va="top",
                fontsize=fontsize, fontweight="bold", color=color)
        for i, line in enumerate(body_lines):
            ax.text(cx, cy + h/2 - 0.62 - i*0.30, line, ha="center", va="top",
                    fontsize=7.5, color=GRAY)

    def thick_arrow(x0, y0, x1, y1, label=""):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY,
                                   lw=2.0, mutation_scale=16))
        if label:
            ax.text((x0+x1)/2, y0 + 0.18, label, ha="center", fontsize=7.5,
                    color=GRAY, style="italic")

    Y = 2.75
    W, H = 2.4, 2.4

    # Stage 1
    stage(1.5, Y, W, H, "① Column {0,1}",
          ["env: MiniGrid-Empty-8x8-v0",
           "policy: MLP 147→64→64→7",
           "alg: PPO (SB3)",
           "threshold: mean>0.85, std<0.05",
           "budget: 500k timesteps",
           "→ STABLE @ 40k steps"],
          BLUE)

    thick_arrow(2.7, Y, 3.3, Y, "freeze")

    # Stage 2 — freeze box
    rect_fr = FancyBboxPatch((3.3, Y - 0.42), 1.4, 0.84,
        boxstyle="round,pad=0.07", linewidth=1.5,
        edgecolor=GREEN, facecolor="#F0FFF4")
    ax.add_patch(rect_fr)
    ax.text(4.0, Y + 0.10, "col1.freeze()", ha="center", fontsize=8,
            color=GREEN, fontweight="bold")
    ax.text(4.0, Y - 0.18, "requires_grad=False\n∀ params", ha="center",
            fontsize=7, color=GRAY)

    thick_arrow(4.7, Y, 5.3, Y)

    # Stage 3
    stage(6.5, Y, W, H, "② Column {0,2}",
          ["env: MiniGrid-FourRooms-v0",
           "lateral_source: {0,1}",
           "lateral: h1,h2 → Lin(64,64) → add",
           "threshold: mean>0.75, std<0.05",
           "budget: 2M timesteps",
           "→ NOT stabilized (max=0.49)"],
          ORANGE)

    thick_arrow(7.7, Y, 8.3, Y, "save")

    # Stage 4 — checkpoints
    stage(9.55, Y, 2.1, H, "③ Checkpoints",
          ["runs/pn4/col01/",
           "  model.zip (196 KB)",
           "  policy_state_dict.pt",
           "  meta.json",
           "runs/pn4/col02/",
           "  model.zip (591 KB)"],
          BLUE)

    thick_arrow(10.6, Y, 11.1, Y)

    # Stage 5 — visualization
    stage(12.5, Y, 2.4, H, "④ Visualization",
          ["200-step FourRooms rollout",
           "lateral norm: col1→col2",
           "bar chart: h1, h2 norms",
           "PCA scatter: col1 vs col2",
           "→ activations.png",
           "(non-zero lateral signal)"],
          PURPLE)

    # Title & phase labels
    ax.set_title("Figure 3 — PN-4 Experiment Pipeline: Column Training and Artifact Generation",
                 fontsize=12, fontweight="bold", color=BLUE, pad=10)

    # Phase label strip
    for x, label in [(1.5, "Train"), (6.5, "Train with Lateral"), (9.55, "Persist"), (12.5, "Analyze")]:
        ax.text(x, 4.2, label, ha="center", fontsize=8, color=GRAY, style="italic",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=LGRAY, alpha=0.3, edgecolor="none"))

    fig.tight_layout()
    fig.savefig(OUT / "fig3_experiment_pipeline.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / "fig3_experiment_pipeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Fig 3: Experiment Pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Copy existing activations.png
# ─────────────────────────────────────────────────────────────────────────────

def fig4_copy_activations():
    src = Path("runs/pn4/activations.png")
    dst = OUT / "fig4_lateral_activations.png"
    shutil.copy2(src, dst)
    print(f"[OK] Fig 4: Lateral Activations (copied from {src})")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Test Summary
# ─────────────────────────────────────────────────────────────────────────────

TEST_DATA = [
    ("Autoencoder\n(AE-1/AE-2)",          7,  "task_detection"),
    ("GRU Identifier\n(GRU-1/GRU-2)",     8,  "task_detection"),
    ("PN Column\n(PN-1/PN-3)",             9,  "continual_learning"),
    ("PN Basics\n(PN-1)",                  6,  "continual_learning"),
    ("PN Trainer\n(PN-4)",                 2,  "continual_learning"),
    ("Option Wrapper\n(OPT-2)",            7,  "options"),
    ("MetaController\n(OPT-1)",            7,  "options"),
    ("Integration\n(INT-2)",              11,  "integration"),
    ("Placeholder\n(CI)",                  1,  "ci"),
]

MODULE_COLORS = {
    "task_detection":     ORANGE,
    "continual_learning": BLUE,
    "options":            GREEN,
    "integration":        PURPLE,
    "ci":                 LGRAY,
}


def fig5_test_summary():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor("white")

    labels = [d[0] for d in TEST_DATA]
    counts = [d[1] for d in TEST_DATA]
    colors = [MODULE_COLORS[d[2]] for d in TEST_DATA]

    x = np.arange(len(labels))
    bars = ax.bar(x, counts, color=colors, edgecolor="white", linewidth=1.2,
                  width=0.6, zorder=3)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{count}", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=GRAY)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Number of Tests Passed", fontsize=10)
    ax.set_ylim(0, 14)
    ax.grid(axis="y", color=LGRAY, linewidth=0.5, alpha=0.7, zorder=0)

    total = sum(counts)
    ax.text(0.99, 0.97, f"Total: {total} / {total} PASSED",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, fontweight="bold", color=GREEN,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0FFF4",
                      edgecolor=GREEN, alpha=0.95))

    # Legend
    legend_handles = [
        mpatches.Patch(color=ORANGE, label="Task Detection (AE, GRU)"),
        mpatches.Patch(color=BLUE,   label="Continual Learning (PN Column, Trainer)"),
        mpatches.Patch(color=GREEN,  label="Options (Option, MetaController)"),
        mpatches.Patch(color=PURPLE, label="Integration (TLM end-to-end)"),
        mpatches.Patch(color=LGRAY,  label="CI Placeholder"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper left",
              framealpha=0.9, ncol=2)

    ax.set_title("Figure 5 — Test Coverage Summary: 58 / 58 Tests Passed  (pytest, Python 3.10)",
                 fontsize=11, fontweight="bold", color=BLUE, pad=10)

    ax.spines["left"].set_color(LGRAY)
    ax.spines["bottom"].set_color(LGRAY)

    fig.tight_layout()
    fig.savefig(OUT / "fig5_test_summary.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / "fig5_test_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Fig 5: Test Summary")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — Milestone Timeline
# ─────────────────────────────────────────────────────────────────────────────

def fig6_milestone_timeline():
    # ── Canvas ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 18)
    ax.set_ylim(-0.6, 7.0)
    ax.axis("off")

    # ── Vertical zone constants ───────────────────────────────────────────────
    SY   = 3.2    # spine y
    BHH  = 0.52   # box half-height
    BHW  = 1.40   # box half-width

    # Above-spine milestones: box floats 1.0 units above spine
    ACY  = SY + 1.0 + BHH   # = 4.72  (box centre)
    # Below-spine milestones: box floats 1.0 units below spine
    BCY  = SY - 1.0 - BHH   # = 1.68  (box centre)

    # Description text sits just outside the box, away from spine
    ADESC_Y = ACY + BHH + 0.15   # = 5.39 (above box top)
    BDESC_Y = BCY - BHH - 0.15   # = 1.01 (below box bottom)

    # Date label sits on the spine-side of the dot, clear of boxes
    ADATE_Y = SY - 0.35   # = 2.85  (for above milestones, below spine)
    BDATE_Y = SY + 0.35   # = 3.55  (for below milestones, above spine)

    # ── Milestone data  (x, date, title, desc, color, above) ─────────────────
    # 9 milestones evenly at x = 1,3,5,7,9,11,13,15,17 — never share x.
    milestones = [
        (1,  "Mar 2026",           "Project\nKickoff",
             "Scope defined;\nhierarchical CRL\narchitecture selected",    GRAY,   True),
        (3,  "May 4",              "Repo & CI\nSetup",
             "GitHub repo, CI\nworkflow, branch\nprotection, tests",       GRAY,   False),
        (5,  "May 23",             "Design Docs\nFinalized",
             "Module interfaces,\nrecursive hierarchy\ndesign locked",     BLUE,   True),
        (7,  "May 24 (AM)",        "All Modules\nImplemented",
             "AE, GRU, Column,\nOption, MC, TLM,\nGUI — 4 subsystems",   BLUE,   False),
        (9,  "May 24 (AM)",        "58 Tests\nPassing",
             "Unit + integration\ntests — 58/58\npassed, 0 failures",     GREEN,  True),
        (11, "May 24 (AM)",        "End-to-End\nDemo",
             "Full pipeline demo;\nmock policies;\nGUI populated",         GREEN,  False),
        (13, "May 24 (AM)",        "PN-4 Smoke\nRun",
             "5k-step chunks;\ncol01 stabilized;\nartifacts written",      ORANGE, True),
        (15, "May 24\n(overnight)","PN-4 Full\nRun",
             "Col{0,1}: stable\n@ 40k, mean=0.958;\nCol{0,2}: 2M steps", ORANGE, False),
        (17, "May 24\n(overnight)","Checkpoints &\nVisualization",
             "6 artifact files;\nactivations.png\ngenerated",              PURPLE, True),
    ]

    # ── Spine ─────────────────────────────────────────────────────────────────
    ax.plot([0.5, 17.5], [SY, SY], color=LGRAY, lw=2.5, zorder=1)

    # ── Draw each milestone ───────────────────────────────────────────────────
    for x, date, title, desc, color, above in milestones:
        cy     = ACY   if above else BCY
        desc_y = ADESC_Y if above else BDESC_Y
        date_y = ADATE_Y if above else BDATE_Y

        # Spine dot
        ax.scatter([x], [SY], s=90, color=color, zorder=5,
                   edgecolors="white", linewidths=1.5)

        # Stem: from just past the spine dot to the near edge of the box
        s0 = SY + 0.16 if above else SY - 0.16
        s1 = cy - BHH  if above else cy + BHH
        ax.plot([x, x], [s0, s1], color=color, lw=1.5, zorder=2)

        # Box
        rect = FancyBboxPatch(
            (x - BHW, cy - BHH), 2*BHW, 2*BHH,
            boxstyle="round,pad=0.08", lw=1.6,
            edgecolor=color, facecolor="white", zorder=3
        )
        ax.add_patch(rect)
        ax.text(x, cy, title, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=color,
                multialignment="center", zorder=4)

        # Description (3 lines max, fontsize 7, outside the box away from spine)
        va_desc = "bottom" if above else "top"
        ax.text(x, desc_y, desc, ha="center", va=va_desc,
                fontsize=7, color=GRAY, multialignment="center", zorder=4)

        # Date (between spine and box, italic, small)
        va_date = "top" if above else "bottom"
        ax.text(x, date_y, date, ha="center", va=va_date,
                fontsize=7.2, color=GRAY, style="italic",
                multialignment="center", zorder=4)

    # ── Phase colour segments on spine ────────────────────────────────────────
    phase_segs = [
        (0.5,  2.5,  GRAY,   "Phase 0"),
        (2.5,  12.5, BLUE,   "Phase 1: Design & Implementation"),
        (12.5, 17.5, ORANGE, "Phase 1: PN-4 Experiments"),
    ]
    for x0, x1, c, lbl in phase_segs:
        ax.plot([x0, x1], [SY, SY], color=c, lw=4, alpha=0.25, zorder=0)
        ax.text((x0+x1)/2, SY - 0.65, lbl, ha="center", va="top",
                fontsize=7.2, color=c, fontweight="bold", style="italic")

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(facecolor="white", edgecolor=GRAY,   label="Phase 0: Setup"),
        mpatches.Patch(facecolor="white", edgecolor=BLUE,   label="Design & Implementation"),
        mpatches.Patch(facecolor="white", edgecolor=GREEN,  label="Verification & Demo"),
        mpatches.Patch(facecolor="white", edgecolor=ORANGE, label="PN-4 Experiments"),
        mpatches.Patch(facecolor="white", edgecolor=PURPLE, label="Artifacts & Completion"),
    ]
    ax.legend(handles=handles, loc="lower center", fontsize=7.8, frameon=True,
              framealpha=0.95, edgecolor=LGRAY, ncol=5,
              bbox_to_anchor=(0.5, -0.04))

    ax.set_title("Figure 6 — Development Milestone Timeline (Progress Report I)",
                 fontsize=12, fontweight="bold", color=BLUE, pad=10)

    fig.tight_layout(pad=0.5)
    fig.savefig(OUT / "fig6_milestone_timeline.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / "fig6_milestone_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Fig 6: Milestone Timeline")


# ─────────────────────────────────────────────────────────────────────────────
# SUPPLEMENTARY — PN-4 Result Summary Table (not counted as main figure)
# ─────────────────────────────────────────────────────────────────────────────

def table_pn4_results():
    fig, ax = plt.subplots(figsize=(10, 3.2))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    col_labels = ["Column", "Environment", "Step Budget", "Final Mean Reward", "Stabilized?", "Checkpoint"]
    rows = [
        ["{0, 1}", "MiniGrid-Empty-8x8-v0",  "500k (@ 40k)",  "0.958",  "Yes  ✓", "runs/pn4/col01/"],
        ["{0, 2}", "MiniGrid-FourRooms-v0",   "2M (exhausted)", "0.240", "No   ✗", "runs/pn4/col02/"],
    ]

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 2.2)

    # Header style
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor(BLUE)
        cell.set_text_props(color="white", fontweight="bold")

    # Row colours
    row_bg = ["#EEF4FF", "#FFF4EE"]
    for i in range(1, 3):
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(row_bg[i - 1])

    # Highlight stable/not-stable cells
    table[1, 4].set_text_props(color=GREEN, fontweight="bold")
    table[2, 4].set_text_props(color=RED, fontweight="bold")

    ax.set_title("PN-4 Result Summary Table",
                 fontsize=11, fontweight="bold", color=BLUE, pad=8, y=0.95)

    fig.tight_layout()
    fig.savefig(OUT / "table_pn4_results.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / "table_pn4_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Supplementary: PN-4 Result Summary Table")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nGenerating figures -> {OUT.resolve()}\n")
    fig1_architecture()
    fig2_training_curves()
    fig3_experiment_pipeline()
    fig4_copy_activations()
    fig5_test_summary()
    fig6_milestone_timeline()
    table_pn4_results()
    print(f"\nAll figures saved to {OUT.resolve()}")
