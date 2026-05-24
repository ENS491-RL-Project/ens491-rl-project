# -*- coding: utf-8 -*-
"""
Demo run -- shows the full AE -> GRU -> MetaController -> Option -> Action pipeline
working end-to-end with a mock trained column (no RL training required).

Run from the repo root:
    python scripts/demo_run.py

After this finishes, launch the GUI to see the populated state:
    streamlit run src/gui/app.py
"""

from __future__ import annotations

import pickle
import random
import sys
from pathlib import Path

import torch

# Make sure src/ is importable when run from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.continual_learning.column import Column
from src.options.task_lifecycle import TaskLifecycleManager
from src.types import ColumnOutput


# ---------------------------------------------------------------------------
# Mock trained policy -- same pattern as the unit tests
# ---------------------------------------------------------------------------

class _MockMlpExtractor:
    def __init__(self):
        self._acts = {0: torch.randn(64), 1: torch.randn(64)}

    def get_activations(self):
        return self._acts


class _MockPolicy:
    """Mimics a trained SB3 ActorCriticPolicy. Returns deterministic outputs."""

    def __init__(self, action: int = 2):
        self._action = action
        self.mlp_extractor = _MockMlpExtractor()

    def predict(self, obs, deterministic=True):
        # Slight randomness so the demo looks alive
        return random.choice([self._action, self._action, (self._action + 1) % 7]), None

    def predict_values(self, obs):
        return torch.tensor([[round(random.uniform(0.7, 0.95), 3)]])

    def parameters(self):
        return iter([])

    def named_parameters(self):
        return iter([])


def _make_trained_column(n: int, m: int, lateral=None, action: int = 2) -> Column:
    col = Column(n=n, m=m, lateral_source=lateral)
    col.policy = _MockPolicy(action=action)
    return col


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

STEPS = 150
SEPARATOR = "-" * 60


def run_demo():
    print(SEPARATOR)
    print("  ENS491 Hierarchical CRL -- Pipeline Demo")
    print(SEPARATOR)

    # ── Build the system ──────────────────────────────────────────────────
    tlm = TaskLifecycleManager()

    # Suppress AE novelty so we can exercise the full MC -> Option path
    tlm.ae.threshold = 1e6
    print("\n[Setup] AE threshold set high -- novelty suppressed for this demo.")

    # Simulate two tasks already learned and stabilised
    col1 = _make_trained_column(n=0, m=1, action=2)           # Task 0: Empty-8x8
    col2 = _make_trained_column(n=0, m=2, lateral=col1, action=5)  # Task 1: FourRooms
    tlm.columns[(0, 1)] = col1
    tlm.columns[(0, 2)] = col2
    tlm._n_tasks_seen = 2

    # Register both tasks in the GRU head
    tlm.gru.register_new_task(0)
    tlm.gru.register_new_task(1)

    # Stabilise both -- this freezes them and adds options to MC at n=1
    tlm.on_column_stabilized(col1)
    tlm.on_column_stabilized(col2)

    mc = tlm.meta_controllers[1]
    print(f"[Setup] {len(tlm.columns)} columns stabilised -> {mc.n_options} options in MC(n=1)")
    print(f"[Setup] Columns: {list(tlm.columns.keys())}")
    print()

    # ── Run the pipeline ──────────────────────────────────────────────────
    print(f"Running {STEPS} steps...\n")
    print(f"{'Step':>5}  {'AE err':>8}  {'GRU task':>9}  {'Conf':>6}  {'Option':>8}  {'Action':>7}  {'Reward':>7}")
    print(SEPARATOR)

    history_for_gui = []
    cumulative_reward = 0.0

    for step in range(1, STEPS + 1):
        obs = torch.randn(147)

        # AE
        ae_out = tlm.ae.forward(obs)

        # GRU
        tlm.gru.observe(obs)
        gru_out = tlm.gru.forward(tlm.gru.get_sequence())

        # Pipeline step
        action = tlm.step(obs)

        # Synthetic reward signal (ramps up to simulate learning)
        reward = round(min(0.95, 0.3 + step / STEPS * 0.7 + random.uniform(-0.05, 0.05)), 3)
        cumulative_reward += reward
        tlm.on_reward(reward, done=False)

        active_opt = tlm._active_option
        opt_id = str(active_opt.option_id) if active_opt else "None"
        task_label = str(gru_out.task_id) if gru_out.task_id != -1 else "?"

        if step <= 20 or step % 25 == 0 or step == STEPS:
            print(
                f"{step:>5}  {ae_out.reconstruction_error:>8.4f}  "
                f"{task_label:>9}  {gru_out.confidence:>6.2%}  "
                f"{opt_id:>8}  {action:>7}  {reward:>7.3f}"
            )

        history_for_gui.append({
            "mc_step": step,
            "option_id": list(active_opt.option_id) if active_opt else None,
            "reward": reward,
        })

    print(SEPARATOR)
    print(f"\n[Result] Mean reward: {cumulative_reward / STEPS:.3f}")
    print(f"[Result] Total MC selections: {mc._step_count}")
    print(f"[Result] Selection history entries: {len(mc.selection_history)}")

    # ── Save state for the GUI ────────────────────────────────────────────
    state = tlm.get_state()
    demo_dict = {
        "ae_error": tlm._last_ae_error,
        "ae_threshold": tlm.ae.threshold,
        "is_novel": False,
        "gru_task_id": state.current_task_id,
        "gru_confidence": state.gru_confidence,
        "gru_all_probs": tlm.gru.forward(tlm.gru.get_sequence()).all_probs.tolist(),
        "columns": [
            {
                "n": c.n,
                "m": c.m,
                "frozen": c.frozen,
                "lateral_source": f"{{{c.lateral_source.n},{c.lateral_source.m}}}"
                if c.lateral_source else None,
            }
            for c in tlm.columns.values()
        ],
        "meta_controllers": [
            {"layer": n, "n_options": mc_.n_options, "step_count": mc_._step_count}
            for n, mc_ in tlm.meta_controllers.items()
        ],
        "selection_history": history_for_gui,
        "active_option_id": list(state.active_option.option_id) if state.active_option else None,
        "active_option_steps": state.active_option.step_count if state.active_option else 0,
        "last_action": action,
        "last_reward": reward,
        "is_training": False,
        "training_column": None,
    }

    out_path = Path(__file__).parent.parent / "runs" / "demo_state.pkl"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(demo_dict, f)

    print(f"\n[GUI]  State saved -> {out_path}")
    print("[GUI]  Launch the GUI to see it:")
    print("       streamlit run src/gui/app.py")
    print()


if __name__ == "__main__":
    run_demo()
