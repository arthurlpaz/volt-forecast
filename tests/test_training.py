"""Training pipeline, MLflow tracking and the model registry.

The registry needs a database-backed tracking store, so every run here points
MLflow at a sqlite file inside tmp_path and works from that directory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from energycast.data import ChronologicalSplitter
from energycast.models import LSTMForecaster, linear_regression
from energycast.training import (
    ExperimentTracker,
    ModelMeta,
    TrainingPipeline,
    load_registered,
    log_and_register,
    lstm_hyperparameters,
    prepare_data,
)

TARGET = "PJME_MW"


def _hourly_frame(hours: int, start: str = "2015-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=hours, freq="h")
    t = np.arange(hours)
    signal = 20000 + 5000 * np.sin(2 * np.pi * t / 24) + 2000 * np.sin(2 * np.pi * t / 168)
    return pd.DataFrame({TARGET: signal}, index=index)


def _splits(hours: int = 2600):
    return ChronologicalSplitter(0.7, 0.15, 0.15).split(_hourly_frame(hours))


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return ExperimentTracker(f"sqlite:///{tmp_path}/mlflow.db", "test-experiment")


def _tiny_windows(seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((24, 5, 3)).astype(np.float32)
    y = rng.standard_normal((24, 2)).astype(np.float32)
    return x, y


def _tiny_lstm() -> LSTMForecaster:
    return LSTMForecaster(
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
        learning_rate=0.01,
        batch_size=8,
        max_epochs=3,
        early_stopping_patience=3,
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=2,
        seed=0,
    )


class TestPrepareData:
    def test_builds_sequence_and_tabular_for_every_split(self):
        prepared = prepare_data(_splits())

        for name in ("train", "validation", "test"):
            assert len(prepared.sequence[name]) > 0
            assert len(prepared.tabular[name]) > 0
            assert prepared.sequence[name].X.shape[1] == 168
            assert prepared.sequence[name].y.shape[1] == 24

    def test_scaler_touches_only_magnitude_columns(self):
        prepared = prepare_data(_splits())

        fitted = set(prepared.scaler.feature_names)
        assert TARGET in fitted
        assert all(c.startswith(("lag_", "rolling_")) or c == TARGET for c in fitted)
        assert not any(c.startswith(("hour", "month", "day", "is_holiday")) for c in fitted)

    def test_target_is_standardised_in_the_windows(self):
        # A standardised target sits near zero mean; raw MW would be ~20000.
        prepared = prepare_data(_splits())
        target_channel = prepared.sequence["train"].feature_names.index(TARGET)
        assert abs(prepared.sequence["train"].X[:, :, target_channel].mean()) < 1.0


class TestBaselineRegistration:
    def test_trains_registers_and_reloads_identically(self, tracker):
        prepared = prepare_data(_splits())
        pipeline = TrainingPipeline(prepared, tracker)

        model = linear_regression()
        version = pipeline.train_baseline("linear_regression", model)
        assert version == "1"

        reloaded = load_registered("energycast-linear_regression")
        test_x = prepared.tabular["test"].X
        np.testing.assert_allclose(reloaded.model.predict(test_x), model.predict(test_x), rtol=1e-6)
        assert reloaded.meta.kind == "sklearn"
        assert reloaded.meta.feature_names == list(test_x.columns)


class TestLSTMRegistration:
    def test_epochs_logged_and_model_reloads_bit_identical(self, tracker):
        x, y = _tiny_windows()
        x_val, y_val = _tiny_windows(seed=1)
        model = _tiny_lstm()

        with tracker.run("lstm") as active:
            model.fit(x, y, x_val, y_val, epoch_callback=tracker_epoch(tracker))
            meta = ModelMeta(
                kind="lstm",
                name=model.name,
                feature_names=["f0", "f1", "f2"],
                target_column="f0",
                prediction_horizon=2,
                sequence_length=5,
                hyperparameters=lstm_hyperparameters(model),
            )
            version = log_and_register(model, _fitted_scaler(), meta, "energycast-lstm")
            run_id = active.info.run_id

        history = MlflowClient().get_metric_history(run_id, "val_loss")
        assert len(history) == 3

        reloaded = load_registered("energycast-lstm", version)
        np.testing.assert_array_equal(reloaded.model.predict(x), model.predict(x))


def tracker_epoch(tracker):
    def callback(epoch: int, train_loss: float, val_loss: float) -> None:
        tracker.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

    return callback


def _fitted_scaler():
    from energycast.features import SeriesScaler

    frame = pd.DataFrame({"f0": np.arange(10.0), "f1": np.arange(10.0), "f2": np.arange(10.0)})
    return SeriesScaler(columns=["f0"]).fit(frame)
