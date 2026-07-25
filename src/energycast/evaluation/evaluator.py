"""Score the registered roster on the sealed test split, on the same rows.

The M6 pipeline left the test set untouched; this is where it is opened. Every
model is measured on the exact same target hours: the sequence windows are the
most restrictive set, so the tabular baselines and the seasonal-naive bar are
aligned onto the window timestamps before anyone is scored. Learned models
predict in z-units and are inverse-transformed to MW first; the naive and the
ground truth are read straight from the clean MW series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from energycast.config import Settings, get_settings
from energycast.evaluation.metrics import mae_per_horizon, mape_per_horizon, rmse_per_horizon
from energycast.models import SeasonalNaiveModel
from energycast.training import LoadedModel, PreparedData
from energycast.utils import get_logger

logger = get_logger(__name__)

NAIVE = "seasonal_naive"


class EvaluationError(RuntimeError):
    """Raised when the roster cannot be scored on a common set of rows."""


@dataclass(frozen=True)
class ModelScore:
    """One model's per-horizon error in MW, plus the run it was trained in."""

    name: str
    kind: str
    rmse: np.ndarray
    mae: np.ndarray
    mape: np.ndarray
    run_id: str | None

    @property
    def rmse_mean(self) -> float:
        return float(self.rmse.mean())


@dataclass(frozen=True)
class EvaluationReport:
    """Every model's score on one set of rows, and the champion among the learned ones."""

    scores: dict[str, ModelScore]
    champion: str
    target_timestamps: pd.DatetimeIndex

    @property
    def n_rows(self) -> int:
        return len(self.target_timestamps)


class Evaluator:
    """Aligns the roster on shared rows and scores each model per horizon."""

    def __init__(self, prepared: PreparedData, settings: Settings | None = None) -> None:
        self.prepared = prepared
        self.settings = settings or get_settings()
        self.target = self.settings.data.source.target_column
        self.horizon = self.settings.model.sequence.prediction_horizon
        self.freq = self.settings.data.validation.expected_frequency

    def evaluate(self, loaded: dict[str, LoadedModel]) -> EvaluationReport:
        sequence = self.prepared.sequence["test"]
        tabular = self.prepared.tabular["test"]
        common = sequence.target_timestamps.intersection(tabular.target_timestamps)
        if not len(common):
            raise EvaluationError("The sequence and tabular test sets share no target hour.")

        y_true = self._ground_truth(common)
        scores: dict[str, ModelScore] = {}
        for name, model in loaded.items():
            dataset = sequence if model.meta.sequence_length is not None else tabular
            predictions = self._predict_learned(model, dataset, common)
            scores[name] = self._score(name, model.meta.kind, model.run_id, y_true, predictions)

        scores[NAIVE] = self._score(NAIVE, NAIVE, None, y_true, self._predict_naive(common))
        champion = min(
            (s for s in scores.values() if s.name != NAIVE), key=lambda s: s.rmse_mean
        ).name

        logger.info(
            "evaluated roster",
            extra={
                "event": "roster_evaluated",
                "rows": len(common),
                "champion": champion,
                "rmse_mean": {name: score.rmse_mean for name, score in scores.items()},
            },
        )
        return EvaluationReport(scores=scores, champion=champion, target_timestamps=common)

    def _ground_truth(self, common: pd.DatetimeIndex) -> np.ndarray:
        series = self.prepared.target_series
        step = pd.Timedelta(1, unit=self.freq)
        columns = [series.reindex(common + h * step).to_numpy() for h in range(self.horizon)]
        y_true = np.column_stack(columns)
        if np.isnan(y_true).any():
            raise EvaluationError("A shared target hour has no measured value in the clean series.")
        return y_true

    def _predict_learned(self, model: LoadedModel, dataset, common: pd.DatetimeIndex) -> np.ndarray:
        rows = dataset.target_timestamps.get_indexer(common)
        predictions = np.asarray(model.model.predict(dataset.X))[rows]
        return model.scaler.inverse_transform(predictions, self.target)

    def _predict_naive(self, common: pd.DatetimeIndex) -> np.ndarray:
        anchors = common - pd.Timedelta(1, unit=self.freq)
        naive = SeasonalNaiveModel.from_settings(self.prepared.target_series)
        return naive.predict(pd.DataFrame(index=anchors))

    def _score(
        self,
        name: str,
        kind: str,
        run_id: str | None,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> ModelScore:
        return ModelScore(
            name=name,
            kind=kind,
            rmse=rmse_per_horizon(y_true, y_pred),
            mae=mae_per_horizon(y_true, y_pred),
            mape=mape_per_horizon(y_true, y_pred),
            run_id=run_id,
        )
