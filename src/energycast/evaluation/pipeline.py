"""End-to-end evaluation: load the roster, score it, hang metrics on the runs."""

from __future__ import annotations

from energycast.evaluation.evaluator import EvaluationReport, Evaluator
from energycast.evaluation.reporting import log_evaluation
from energycast.models import build_from_settings
from energycast.training import (
    ExperimentTracker,
    LoadedModel,
    load_registered,
    prepare_data,
    registered_name,
)
from energycast.training.pipeline import load_splits
from energycast.utils import get_logger

logger = get_logger(__name__)


def _roster_names() -> list[str]:
    return ["lstm", *build_from_settings().keys()]


def load_roster() -> dict[str, LoadedModel]:
    return {name: load_registered(registered_name(name)) for name in _roster_names()}


def main() -> EvaluationReport:
    ExperimentTracker.from_settings()
    prepared = prepare_data(load_splits())
    report = Evaluator(prepared).evaluate(load_roster())
    log_evaluation(report)
    logger.info(
        "evaluation run complete",
        extra={
            "event": "evaluation_complete",
            "champion": report.champion,
            "rows": report.n_rows,
        },
    )
    return report
