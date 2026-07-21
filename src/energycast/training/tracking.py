"""A thin seam over MLflow so no model module has to import it.

Centralises the tracking URI and experiment so a run is opened the same way
everywhere, and tags every run with the commit it came from.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from contextlib import contextmanager

import mlflow

from energycast.utils import get_logger

logger = get_logger(__name__)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


class ExperimentTracker:
    """Opens MLflow runs against one tracking URI and experiment."""

    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    @classmethod
    def from_settings(cls) -> ExperimentTracker:
        from energycast.config import get_settings

        mlf = get_settings().base.mlflow
        return cls(tracking_uri=mlf.tracking_uri, experiment_name=mlf.experiment_name)

    @contextmanager
    def run(self, run_name: str) -> Iterator[mlflow.ActiveRun]:
        with mlflow.start_run(run_name=run_name) as active:
            mlflow.set_tag("git_sha", _git_sha())
            logger.info(
                "opened mlflow run",
                extra={"event": "run_opened", "run_name": run_name, "run_id": active.info.run_id},
            )
            yield active

    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)
