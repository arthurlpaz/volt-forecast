"""Evaluation and champion selection on the sealed test split.

    load roster -> align on shared rows -> per-horizon RMSE/MAE/MAPE -> champion

Scores every registered model on the same target hours, in MW, and crowns the
learned model with the lowest mean-over-horizons RMSE. The seasonal naive is
the bar, not a candidate. Registry promotion is milestone 11's job.
"""

from energycast.evaluation.evaluator import (
    NAIVE,
    EvaluationError,
    EvaluationReport,
    Evaluator,
    ModelScore,
)
from energycast.evaluation.metrics import (
    MetricError,
    mae_per_horizon,
    mape_per_horizon,
    rmse_per_horizon,
)
from energycast.evaluation.pipeline import load_roster, main
from energycast.evaluation.reporting import log_evaluation

__all__ = [
    "NAIVE",
    "EvaluationError",
    "EvaluationReport",
    "Evaluator",
    "MetricError",
    "ModelScore",
    "load_roster",
    "log_evaluation",
    "mae_per_horizon",
    "main",
    "mape_per_horizon",
    "rmse_per_horizon",
]
