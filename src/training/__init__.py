"""v2 training: device selection, the evaluation criterion, the node trainer."""

from src.training.device import available_accelerator, describe, resolve_device
from src.training.evaluation import (
    EvalResult,
    EvaluationCallback,
    StabilityCriterion,
    evaluate_node,
    heldout_seeds,
    run_episode,
    stopping_seeds,
    wilson_lower,
)

__all__ = [
    "resolve_device", "available_accelerator", "describe",
    "evaluate_node", "run_episode", "EvalResult", "StabilityCriterion",
    "EvaluationCallback", "wilson_lower", "stopping_seeds", "heldout_seeds",
]
