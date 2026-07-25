"""Hang the evaluation metrics on the runs the models were trained in.

Reopens each learned model's training run by id and logs its per-horizon test
metrics there, so training and evaluation of one model live in one run. The
naive has no run and is skipped. Registry stages are left untouched: promotion
is milestone 11's job.
"""

from __future__ import annotations

import mlflow

from energycast.evaluation.evaluator import EvaluationReport
from energycast.utils import get_logger

logger = get_logger(__name__)


def log_evaluation(report: EvaluationReport) -> None:
    for name, score in report.scores.items():
        if score.run_id is None:
            continue
        metrics = {}
        for h in range(len(score.rmse)):
            metrics[f"test_rmse_h{h + 1}"] = float(score.rmse[h])
            metrics[f"test_mae_h{h + 1}"] = float(score.mae[h])
            metrics[f"test_mape_h{h + 1}"] = float(score.mape[h])
        metrics["test_rmse_mean"] = score.rmse_mean
        metrics["test_mae_mean"] = float(score.mae.mean())
        metrics["test_mape_mean"] = float(score.mape.mean())

        with mlflow.start_run(run_id=score.run_id):
            mlflow.log_metrics(metrics)
            mlflow.set_tag("is_champion", str(name == report.champion).lower())

    logger.info(
        "logged evaluation to training runs",
        extra={"event": "evaluation_logged", "champion": report.champion},
    )
