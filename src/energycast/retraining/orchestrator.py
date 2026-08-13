"""The retraining cycle: read the signals, decide, retrain, compare, promote.

The three trigger signals already exist (drift from M10, rolling error from M9,
new observations counted against the last cutoff). When any trips, the champion
architecture is retrained as a challenger, both are scored on the same recent
rows with the M7 evaluator, and the challenger takes the `champion` alias only if
it wins by the configured margin. Every cycle is logged, retrain or not.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import mlflow
import pandas as pd

from energycast.config import Settings, get_settings
from energycast.data import DataSplits
from energycast.drift.store import SQLiteDriftStore
from energycast.evaluation import Evaluator
from energycast.models import build_from_settings
from energycast.monitoring import MonitoringError, SQLitePredictionStore, rolling_report
from energycast.retraining.promotion import beats_champion
from energycast.retraining.store import SQLiteRetrainingStore
from energycast.retraining.triggers import RetrainDecision, TriggerSignals, evaluate_triggers
from energycast.training import (
    CHALLENGER,
    CHAMPION,
    ExperimentTracker,
    TrainingPipeline,
    load_registered,
    load_splits,
    prepare_data,
    registered_name,
    resolve_version,
    set_alias,
)
from energycast.utils import get_logger

logger = get_logger(__name__)


class RetrainingError(RuntimeError):
    """Raised when the champion architecture cannot be retrained as a challenger."""


@dataclass(frozen=True)
class RetrainingOutcome:
    """What one cycle did: whether it fired, and whether a challenger was promoted."""

    triggered: bool
    reasons: list[str]
    champion_before: str
    champion_after: str
    promoted: bool
    challenger_version: str | None


class RetrainingOrchestrator:
    """Runs one retraining cycle end to end, or just reports the trigger decision."""

    def __init__(
        self,
        drift_store: SQLiteDriftStore,
        monitoring_store: SQLitePredictionStore,
        retraining_store: SQLiteRetrainingStore,
        settings: Settings | None = None,
    ) -> None:
        self.drift_store = drift_store
        self.monitoring_store = monitoring_store
        self.retraining_store = retraining_store
        settings = settings or get_settings()
        self.settings = settings
        self.config = settings.base.retraining
        self.champion = settings.base.serving.champion_model
        self.horizon = settings.model.sequence.prediction_horizon
        self._splits: DataSplits | None = None

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> RetrainingOrchestrator:
        settings = settings or get_settings()
        base = settings.base
        return cls(
            SQLiteDriftStore(base.drift.database_path),
            SQLitePredictionStore(base.monitoring.database_path),
            SQLiteRetrainingStore(base.retraining.database_path),
            settings,
        )

    def check(self) -> tuple[TriggerSignals, RetrainDecision]:
        signals = self._collect_signals()
        return signals, evaluate_triggers(signals, self.config)

    def run_cycle(self) -> RetrainingOutcome:
        mlflow.set_tracking_uri(self.settings.base.mlflow.tracking_uri)
        signals, decision = self.check()
        run_at = datetime.now(UTC)
        data_cutoff = self._full_index().max()
        before = self.champion

        if not decision.should_retrain:
            self.retraining_store.record_cycle(run_at, decision, data_cutoff, before, before, False)
            return RetrainingOutcome(False, decision.reasons, before, before, False, None)

        name = registered_name(self.champion)
        incumbent_version = resolve_version(name, CHAMPION)
        prepared = prepare_data(self._load(), self.settings)
        challenger_version = self._train_challenger(prepared)
        promoted, after = self._promote(prepared, name, incumbent_version, challenger_version)

        self.retraining_store.record_cycle(run_at, decision, data_cutoff, before, after, promoted)
        logger.info(
            "retraining cycle ran",
            extra={
                "event": "retraining_cycle",
                "reasons": decision.reasons,
                "challenger_version": challenger_version,
                "promoted": promoted,
                "champion_after": after,
            },
        )
        return RetrainingOutcome(
            True, decision.reasons, before, after, promoted, challenger_version
        )

    def close(self) -> None:
        self.drift_store.close()
        self.monitoring_store.close()
        self.retraining_store.close()

    def _collect_signals(self) -> TriggerSignals:
        latest = self.drift_store.latest()
        return TriggerSignals(
            drifted=latest.drifted if latest is not None else None,
            rolling_rmse=self._rolling_rmse(),
            new_observations=self._new_observations(),
        )

    def _rolling_rmse(self) -> float | None:
        try:
            report = rolling_report(
                self.monitoring_store, self.champion, self.horizon, self.config.rolling_window_days
            )
        except MonitoringError:
            return None
        return float(report.rmse.mean())

    def _new_observations(self) -> int:
        marker = self.retraining_store.last_trained_cutoff()
        if marker is None:
            return 0
        return int((self._full_index() > pd.Timestamp(marker)).sum())

    def _train_challenger(self, prepared) -> str:
        pipeline = TrainingPipeline(prepared, ExperimentTracker.from_settings(), self.settings)
        if self.champion == "lstm":
            return pipeline.train_lstm()
        baselines = build_from_settings()
        if self.champion not in baselines:
            raise RetrainingError(
                f"Champion {self.champion!r} is neither the LSTM nor a configured baseline."
            )
        return pipeline.train_baseline(self.champion, baselines[self.champion])

    def _promote(
        self, prepared, name: str, incumbent_version: str, challenger_version: str
    ) -> tuple[bool, str]:
        evaluator = Evaluator(prepared, self.settings)
        champion_rmse = self._score(evaluator, load_registered(name, version=incumbent_version))
        challenger_rmse = self._score(evaluator, load_registered(name, version=challenger_version))
        if beats_champion(champion_rmse, challenger_rmse, self.config.promotion_margin):
            set_alias(name, CHAMPION, challenger_version)
            set_alias(name, CHALLENGER, incumbent_version)
            return True, f"{self.champion}@v{challenger_version}"
        set_alias(name, CHAMPION, incumbent_version)
        set_alias(name, CHALLENGER, challenger_version)
        return False, f"{self.champion}@v{incumbent_version}"

    def _score(self, evaluator: Evaluator, loaded) -> float:
        return evaluator.evaluate({self.champion: loaded}).scores[self.champion].rmse_mean

    def _full_index(self) -> pd.DatetimeIndex:
        splits = self._load()
        return splits.train.index.union(splits.validation.index).union(splits.test.index)

    def _load(self) -> DataSplits:
        if self._splits is None:
            self._splits = load_splits()
        return self._splits


def main() -> RetrainingOutcome:
    """Run one cycle on the configured data source; the entrypoint a scheduler calls."""
    orchestrator = RetrainingOrchestrator.from_settings()
    try:
        outcome = orchestrator.run_cycle()
    finally:
        orchestrator.close()
    logger.info(
        "retraining complete",
        extra={
            "event": "retraining_complete",
            "triggered": outcome.triggered,
            "promoted": outcome.promoted,
        },
    )
    return outcome
