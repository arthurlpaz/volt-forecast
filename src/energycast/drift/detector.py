"""Input drift detection: has the recent measured distribution left the train one.

Drift is a distributional comparison, not a time-series test. The reference is
the training distribution the model was fitted on; the current window is the
most recent measured hours. Only the MW-magnitude columns are compared, never
the deterministic calendar features, which a sub-season window would flag as
drift purely from the window covering fewer months than train.

`detect_drift` is the Evidently wrapper and knows nothing about where the data
came from; `DriftDetector` builds the reference and current frames and persists
the run. Evidently is isolated behind `detect_drift` so a library change lands
in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from energycast.config import Settings, get_settings
from energycast.data import DataSplits
from energycast.drift.store import SQLiteDriftStore
from energycast.features import build_features
from energycast.training import load_splits
from energycast.training.pipeline import _scale_columns
from energycast.utils import get_logger

logger = get_logger(__name__)


class DriftError(ValueError):
    """Raised when a drift run has no data to compare on one of its sides."""


@dataclass(frozen=True)
class DriftResult:
    """The dataset-level drift signal the retraining trigger reads, per run."""

    method: str
    n_columns: int
    n_drifted: int
    share: float
    threshold: float
    drifted: bool
    per_column: dict[str, float]
    reference_rows: int
    current_rows: int


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    method: str,
    drift_share: float,
) -> tuple[DriftResult, Report]:
    """Compare two column-aligned frames and return the drift signal and report."""
    if reference.empty or current.empty:
        raise DriftError(
            "Drift needs a non-empty reference and current window; got "
            f"{len(reference)} reference and {len(current)} current rows."
        )

    report = Report(metrics=[DataDriftPreset(stattest=method, drift_share=drift_share)])
    report.run(reference_data=reference, current_data=current)
    metrics = {metric["metric"]: metric["result"] for metric in report.as_dict()["metrics"]}
    dataset = metrics["DatasetDriftMetric"]
    per_column = {
        column: float(info["drift_score"])
        for column, info in metrics["DataDriftTable"]["drift_by_columns"].items()
    }
    result = DriftResult(
        method=method,
        n_columns=int(dataset["number_of_columns"]),
        n_drifted=int(dataset["number_of_drifted_columns"]),
        share=float(dataset["share_of_drifted_columns"]),
        threshold=float(dataset["drift_share"]),
        drifted=bool(dataset["dataset_drift"]),
        per_column=per_column,
        reference_rows=len(reference),
        current_rows=len(current),
    )
    return result, report


class DriftDetector:
    """Builds the train reference and recent current window, then scores drift."""

    def __init__(self, store: SQLiteDriftStore, settings: Settings | None = None) -> None:
        self.store = store
        settings = settings or get_settings()
        self.drift = settings.base.drift
        self.target = settings.data.source.target_column
        self._splits: DataSplits | None = None
        self._reference_frame: pd.DataFrame | None = None

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> DriftDetector:
        settings = settings or get_settings()
        return cls(SQLiteDriftStore(settings.base.drift.database_path), settings)

    def run(self) -> DriftResult:
        reference = self._reference()
        current = self._current()
        result, report = detect_drift(
            reference,
            current,
            method=self.drift.method,
            drift_share=self.drift.drift_share,
        )
        run_at = datetime.now(UTC)
        self._save_snapshot(report, run_at)
        self.store.record_run(result, run_at, current.index.min(), current.index.max())
        logger.info(
            "drift scored",
            extra={
                "event": "drift_scored",
                "share": result.share,
                "drifted": result.drifted,
                "current_rows": result.current_rows,
            },
        )
        return result

    def close(self) -> None:
        self.store.close()

    def _magnitude(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame[_scale_columns(frame, self.target)]

    def _reference(self) -> pd.DataFrame:
        if self._reference_frame is None:
            self._reference_frame = self._magnitude(build_features(self._load().train))
        return self._reference_frame

    def _current(self) -> pd.DataFrame:
        features = self._magnitude(build_features(self._load().test))
        if features.empty:
            raise DriftError("The current split produced no complete feature rows to score.")
        cutoff = features.index.max() - pd.Timedelta(days=self.drift.window_days)
        window = features[features.index >= cutoff]
        if window.empty:
            raise DriftError(
                f"No measured hours fall within the last {self.drift.window_days} days."
            )
        return window

    def _load(self) -> DataSplits:
        if self._splits is None:
            self._splits = load_splits()
        return self._splits

    def _save_snapshot(self, report: Report, run_at: datetime) -> None:
        directory = Path(self.drift.snapshot_dir)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = run_at.strftime("%Y%m%dT%H%M%SZ")
        report.save_html(str(directory / f"drift-{stamp}.html"))
