from __future__ import annotations

from typing import TYPE_CHECKING

from src.types import Action, ColumnIdx, LayerIdx, Observation, OptionStepOutput

if TYPE_CHECKING:
    from src.continual_learning.column import Column


class Option:
    """
    Sutton et al. (1999) option wrapping a frozen Column.

    Termination strategy for Progress I: fixed step limit (MAX_STEPS).
    Termination is checked at the end of each step — terminated=True signals
    the MetaController to regain control.

    option_id: (n, m) tuple matching the wrapped column's coordinates.
    """

    MAX_STEPS = 200

    def __init__(self, column: Column, option_id: tuple[LayerIdx, ColumnIdx]) -> None:
        self.column = column
        self.option_id = option_id
        self.step_count: int = 0

    def step(self, obs: Observation) -> OptionStepOutput:
        col_out = self.column.forward(obs)
        self.step_count += 1
        terminated = self.step_count >= self.MAX_STEPS
        return OptionStepOutput(
            action=col_out.action,
            terminated=terminated,
            info={"step_count": self.step_count, "value": col_out.value},
        )

    def reset(self) -> None:
        self.step_count = 0

    def can_initiate(self, obs: Observation) -> bool:
        # Full state space initiation set (universal).
        return True

    def __repr__(self) -> str:
        n, m = self.option_id
        return f"Option({{{n},{m}}} steps={self.step_count}/{self.MAX_STEPS})"
