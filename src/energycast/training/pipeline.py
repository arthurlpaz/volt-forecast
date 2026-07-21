"""The training pipeline: the M5 benchmark skeleton, made reproducible.

    load -> clean -> validate -> split -> features -> scale -> windows/targets -> fit

Data preparation is split from model fitting so the same prepared splits feed
every model in the roster, and so tests can inject splits without the real CSV.
The scaler is fitted on train only and the lag features drop the leading hours
of incomplete history before it ever sees them.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from energycast.config import Settings, get_settings
from energycast.data import (
    ChronologicalSplitter,
    CSVDataLoader,
    DataSplits,
    SchemaValidator,
    TimeSeriesCleaner,
)
from energycast.features import (
    CalendarFeatureBuilder,
    HorizonTargetBuilder,
    LagFeatureBuilder,
    SequenceBuilder,
    SequenceDataset,
    SeriesScaler,
    TabularDataset,
)
from energycast.models import LSTMForecaster, SklearnBaseline, build_from_settings
from energycast.training.registry import ModelMeta, log_and_register, lstm_hyperparameters
from energycast.training.tracking import ExperimentTracker
from energycast.utils import get_logger

logger = get_logger(__name__)

_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class PreparedData:
    """Every model's inputs, drawn once from the same scaled splits."""

    scaler: SeriesScaler
    sequence: dict[str, SequenceDataset]
    tabular: dict[str, TabularDataset]
    target_series: pd.Series


def load_splits() -> DataSplits:
    """load -> clean -> validate -> split. The only source-specific step is load."""
    frame = CSVDataLoader.from_settings().load()
    frame = TimeSeriesCleaner.from_settings().clean(frame)
    SchemaValidator.from_settings().validate(frame)
    return ChronologicalSplitter.from_settings().split(frame)


def _features_for_split(frame: pd.DataFrame) -> pd.DataFrame:
    calendar = CalendarFeatureBuilder().build(frame)
    lagged = LagFeatureBuilder.from_settings().build(calendar)
    return lagged.dropna()


def _scale_columns(frame: pd.DataFrame, target: str) -> list[str]:
    """The MW-magnitude columns: the target and its lags. Calendar features are left as they are."""
    return [target] + [c for c in frame.columns if c.startswith(("lag_", "rolling_"))]


def prepare_data(splits: DataSplits, settings: Settings | None = None) -> PreparedData:
    """Build the scaled sequence and tabular datasets for every split.

    Features are built per split and the scaler is fitted on train only, so no
    statistic and no window lookback crosses a split boundary.
    """
    settings = settings or get_settings()
    target = settings.data.source.target_column

    features = {name: _features_for_split(getattr(splits, name)) for name in _SPLITS}
    columns = _scale_columns(features["train"], target)
    scaler = SeriesScaler.from_settings(columns=columns).fit(features["train"])
    scaled = {name: scaler.transform(frame) for name, frame in features.items()}

    sequence_builder = SequenceBuilder.from_settings()
    target_builder = HorizonTargetBuilder.from_settings()
    sequence = {name: sequence_builder.build(frame) for name, frame in scaled.items()}
    tabular = {name: target_builder.build(frame) for name, frame in scaled.items()}

    target_series = pd.concat([getattr(splits, name)[target] for name in _SPLITS]).sort_index()
    return PreparedData(
        scaler=scaler, sequence=sequence, tabular=tabular, target_series=target_series
    )


class TrainingPipeline:
    """Fits each model on the prepared splits and registers it with its scaler."""

    def __init__(
        self,
        prepared: PreparedData,
        tracker: ExperimentTracker,
        settings: Settings | None = None,
    ) -> None:
        self.prepared = prepared
        self.tracker = tracker
        self.settings = settings or get_settings()

    def train_lstm(self) -> str:
        train, validation = self.prepared.sequence["train"], self.prepared.sequence["validation"]
        model = LSTMForecaster.from_settings()
        with self.tracker.run("lstm"):
            self.tracker.log_params(
                {
                    "kind": "lstm",
                    "n_features": train.X.shape[2],
                    "train_windows": len(train),
                    "sequence_length": self.settings.model.sequence.sequence_length,
                    **lstm_hyperparameters(model),
                }
            )
            model.fit(
                train.X,
                train.y,
                validation.X,
                validation.y,
                epoch_callback=self._log_epoch,
            )
            meta = ModelMeta(
                kind="lstm",
                name=model.name,
                feature_names=train.feature_names,
                target_column=self.settings.data.source.target_column,
                prediction_horizon=self.settings.model.sequence.prediction_horizon,
                sequence_length=self.settings.model.sequence.sequence_length,
                hyperparameters=lstm_hyperparameters(model),
            )
            return log_and_register(model, self.prepared.scaler, meta, f"energycast-{model.name}")

    def train_baseline(self, name: str, model: SklearnBaseline) -> str:
        train = self.prepared.tabular["train"]
        with self.tracker.run(name):
            self.tracker.log_params(
                {
                    "kind": "sklearn",
                    "model": name,
                    "n_features": train.X.shape[1],
                    "rows": len(train),
                }
            )
            model.fit(train.X, train.y)
            meta = ModelMeta(
                kind="sklearn",
                name=name,
                feature_names=list(train.X.columns),
                target_column=self.settings.data.source.target_column,
                prediction_horizon=self.settings.model.sequence.prediction_horizon,
                sequence_length=None,
                hyperparameters={},
            )
            return log_and_register(model, self.prepared.scaler, meta, f"energycast-{name}")

    def run_all(self) -> dict[str, str]:
        versions = {"lstm": self.train_lstm()}
        for name, model in build_from_settings().items():
            versions[name] = self.train_baseline(name, model)
        return versions

    def _log_epoch(self, epoch: int, train_loss: float, val_loss: float) -> None:
        self.tracker.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)


def main() -> dict[str, str]:
    """End-to-end training run on the configured data source."""
    prepared = prepare_data(load_splits())
    pipeline = TrainingPipeline(prepared, ExperimentTracker.from_settings())
    versions = pipeline.run_all()
    logger.info("training run complete", extra={"event": "run_all_complete", "versions": versions})
    return versions
