"""
Minimal Streamlit GUI — ENS491 Progress Report I

Architectural flow display:
  - AE reconstruction error + novelty decision
  - GRU predicted task + confidence + probability bars
  - Column hierarchy (n, m, frozen, lateral connection)
  - Option selection history
  - Current action / reward (live mode only)

Modes:
  Demo mode (default): loads runs/demo_state.pkl as a plain dict of metrics.
  Live mode (sidebar toggle): imports TaskLifecycleManager and polls get_state().
"""

from __future__ import annotations

import os
import pickle
import random
import time
from pathlib import Path
from typing import Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ENS491 — Hierarchical CRL Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEMO_STATE_PATH = Path(__file__).parent.parent.parent / "runs" / "demo_state.pkl"

# ---------------------------------------------------------------------------
# Demo state loader
# ---------------------------------------------------------------------------

def _load_demo_state() -> dict:
    """Load plain-dict demo state. Returns a minimal default if file is missing."""
    if DEMO_STATE_PATH.exists():
        with open(DEMO_STATE_PATH, "rb") as f:
            return pickle.load(f)
    # Minimal default for first launch
    return {
        "ae_error": 0.031,
        "ae_threshold": 0.05,
        "is_novel": False,
        "gru_task_id": 0,
        "gru_confidence": 0.83,
        "gru_all_probs": [0.83, 0.17],
        "columns": [
            {"n": 0, "m": 1, "frozen": True, "lateral_source": None},
            {"n": 0, "m": 2, "frozen": False, "lateral_source": "{0,1}"},
        ],
        "meta_controllers": [
            {"layer": 1, "n_options": 1, "step_count": 42},
        ],
        "selection_history": [
            {"mc_step": i, "option_id": [0, 1], "reward": round(random.uniform(0.0, 1.0), 3)}
            for i in range(1, 11)
        ],
        "active_option_id": [0, 1],
        "active_option_steps": 47,
        "last_action": 2,
        "last_reward": 0.95,
        "is_training": False,
        "training_column": None,
    }


# ---------------------------------------------------------------------------
# Live state (optional)
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_tlm():
    from src.options.task_lifecycle import TaskLifecycleManager
    return TaskLifecycleManager()


def _live_state_dict(tlm) -> dict:
    s = tlm.get_state()
    mc = tlm.meta_controllers.get(1)
    history = mc.selection_history[-20:] if mc else []
    active_id = s.active_option.option_id if s.active_option else None
    active_steps = s.active_option.step_count if s.active_option else 0
    training_col = (
        f"{{{s.active_column.n},{s.active_column.m}}}"
        if s.active_column else None
    )
    cols = [
        {
            "n": c.n, "m": c.m, "frozen": c.frozen,
            "lateral_source": (
                f"{{{c.lateral_source.n},{c.lateral_source.m}}}"
                if c.lateral_source else None
            ),
        }
        for c in tlm.columns.values()
    ]
    return {
        "ae_error": s.ae_last_error,
        "ae_threshold": tlm.ae.threshold,
        "is_novel": s.ae_last_error > tlm.ae.threshold,
        "gru_task_id": s.current_task_id,
        "gru_confidence": s.gru_confidence,
        "gru_all_probs": [],
        "columns": cols,
        "meta_controllers": [
            {"layer": n, "n_options": mc.n_options, "step_count": mc._step_count}
            for n, mc in tlm.meta_controllers.items()
        ],
        "selection_history": history,
        "active_option_id": active_id,
        "active_option_steps": active_steps,
        "last_action": None,
        "last_reward": None,
        "is_training": s.is_training,
        "training_column": training_col,
    }


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_ae(d: dict) -> None:
    st.subheader("Autoencoder — Novelty Detection")
    col1, col2, col3 = st.columns(3)
    col1.metric("Reconstruction Error", f"{d['ae_error']:.4f}")
    col2.metric("Threshold", f"{d['ae_threshold']:.4f}")
    novelty_label = "YES — Novel" if d["is_novel"] else "NO — Familiar"
    novelty_color = "🔴" if d["is_novel"] else "🟢"
    col3.metric("Is Novel?", f"{novelty_color} {novelty_label}")


def _render_gru(d: dict) -> None:
    st.subheader("GRU — Task Identification")
    col1, col2 = st.columns(2)
    task_label = str(d["gru_task_id"]) if d["gru_task_id"] != -1 else "−1 (uncertain)"
    col1.metric("Predicted Task ID", task_label)
    col2.metric("Confidence", f"{d['gru_confidence']:.2%}")
    probs = d.get("gru_all_probs", [])
    if probs:
        import pandas as pd
        df = pd.DataFrame(
            {"Task": [f"Task {i}" for i in range(len(probs))], "P": probs}
        ).set_index("Task")
        st.bar_chart(df)


def _render_columns(d: dict) -> None:
    st.subheader("Progressive Network — Column Hierarchy")
    cols = d.get("columns", [])
    if not cols:
        st.info("No columns created yet.")
        return
    for c in cols:
        frozen_icon = "🔒 frozen" if c["frozen"] else "🟡 training"
        lateral = f"  ← {c['lateral_source']}" if c["lateral_source"] else ""
        st.markdown(f"- **`{{{c['n']},{c['m']}}}`** — {frozen_icon}{lateral}")
    if d.get("is_training") and d.get("training_column"):
        st.info(f"Training in progress: {d['training_column']}")


def _render_meta_controllers(d: dict) -> None:
    st.subheader("Meta-Controllers")
    mcs = d.get("meta_controllers", [])
    if not mcs:
        st.info("No MetaControllers active.")
        return
    for mc in mcs:
        st.markdown(
            f"- **Layer n={mc['layer']}** — {mc['n_options']} option(s), "
            f"{mc['step_count']} selections"
        )


def _render_option_history(d: dict) -> None:
    st.subheader("Option Selection History (last 20)")
    history = d.get("selection_history", [])
    if not history:
        st.info("No selections yet.")
        return
    import pandas as pd
    rows = []
    for h in history[-20:]:
        oid = h.get("option_id")
        oid_str = f"{{{oid[0]},{oid[1]}}}" if oid else "—"
        rows.append({
            "MC Step": h.get("mc_step", "—"),
            "Option": oid_str,
            "Reward": h.get("reward") if h.get("reward") is not None else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _render_current(d: dict) -> None:
    st.subheader("Current Step")
    col1, col2, col3 = st.columns(3)
    oid = d.get("active_option_id")
    oid_str = f"{{{oid[0]},{oid[1]}}}" if oid else "None"
    col1.metric("Active Option", oid_str)
    col2.metric("Steps into Option", d.get("active_option_steps", 0))
    action = d.get("last_action")
    col3.metric("Last Action", str(action) if action is not None else "—")


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("ENS491 — Hierarchical Continual RL Agent")
    st.caption(
        "Architectural feasibility demonstration | Progress Report I | "
        "Systematic experiments and ablations come after this integration milestone."
    )

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        live_mode = st.toggle("Live Mode", value=False)
        if live_mode:
            refresh_interval = st.slider("Refresh (s)", 1, 10, 2)
        st.divider()
        st.markdown("**Stack**")
        st.markdown("- PyTorch 2.x + SB3\n- MiniGrid / Gymnasium\n- Progressive Networks")

    # Load state
    if live_mode:
        try:
            tlm = _get_tlm()
            d = _live_state_dict(tlm)
        except Exception as e:
            st.error(f"Live mode error: {e}")
            d = _load_demo_state()
    else:
        d = _load_demo_state()

    # Render panels
    _render_ae(d)
    st.divider()
    _render_gru(d)
    st.divider()

    left, right = st.columns([1, 1])
    with left:
        _render_columns(d)
        _render_meta_controllers(d)
    with right:
        _render_option_history(d)
        _render_current(d)

    # Auto-refresh in live mode
    if live_mode:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
