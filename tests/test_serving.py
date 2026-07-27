"""The forecast path (features -> scale -> predict -> inverse) and the API.

The synthetic history is the same periodic MW signal the evaluation tests use;
here it only has to land the forecast in a plausible MW range and pin the shape,
the anchor (t+1 .. t+24) and the refusal to forecast from an incomplete history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from energycast.config import get_settings
from energycast.data import ChronologicalSplitter
from energycast.models import linear_regression
from energycast.serving import Forecaster, create_app
from energycast.serving.forecaster import ForecastError
from energycast.training import (
    ExperimentTracker,
    LoadedModel,
    ModelMeta,
    TrainingPipeline,
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


class _Stub:
    def __init__(self, horizon: int = 24) -> None:
        self.horizon = horizon
        self.seen = None

    def predict(self, X) -> np.ndarray:  # noqa: N803
        self.seen = X
        rows = len(X) if hasattr(X, "__len__") else X.shape[0]
        return np.zeros((rows, self.horizon))


def _loaded(kind: str, sequence_length, scaler, feature_names) -> tuple[LoadedModel, _Stub]:
    stub = _Stub()
    meta = ModelMeta(
        kind=kind,
        name="stub",
        feature_names=feature_names,
        target_column=TARGET,
        prediction_horizon=24,
        sequence_length=sequence_length,
        hyperparameters={},
    )
    return LoadedModel(model=stub, scaler=scaler, meta=meta, run_id="stub"), stub


@pytest.fixture
def prepared():
    return prepare_data(_splits())


class TestForecaster:
    def test_baseline_forecast_is_24_hours_in_mw_from_the_last_hour(self, prepared):
        feature_names = prepared.tabular["test"].feature_names
        loaded, stub = _loaded("sklearn", None, prepared.scaler, feature_names)
        history = _hourly_frame(400)

        forecast = Forecaster(loaded).predict(history)

        assert forecast.values.shape == (24,)
        assert isinstance(stub.seen, pd.DataFrame) and len(stub.seen) == 1
        assert forecast.timestamps[0] == history.index[-1] + pd.Timedelta(hours=1)
        assert forecast.timestamps[-1] == history.index[-1] + pd.Timedelta(hours=24)
        assert (5000 < forecast.values).all() and (forecast.values < 40000).all()

    def test_lstm_receives_a_single_lookback_window(self, prepared):
        feature_names = prepared.sequence["test"].feature_names
        loaded, stub = _loaded("lstm", 168, prepared.scaler, feature_names)

        forecast = Forecaster(loaded).predict(_hourly_frame(400))

        assert stub.seen.shape == (1, 168, len(feature_names))
        assert forecast.values.shape == (24,)

    def test_history_shorter_than_the_lookback_is_refused(self, prepared):
        feature_names = prepared.tabular["test"].feature_names
        loaded, _ = _loaded("sklearn", None, prepared.scaler, feature_names)

        with pytest.raises(ForecastError):
            Forecaster(loaded).predict(_hourly_frame(50))

    def test_gap_before_the_last_hour_is_refused(self, prepared):
        # Drop the hour right before the last one: the last hour's lag_1h has no
        # value, so its feature row is incomplete and cannot be forecast from.
        feature_names = prepared.tabular["test"].feature_names
        loaded, _ = _loaded("sklearn", None, prepared.scaler, feature_names)
        history = _hourly_frame(400)
        gapped = history.drop(history.index[-2])

        with pytest.raises(ForecastError):
            Forecaster(loaded).predict(gapped)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENERGYCAST_BASE__MLFLOW__TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setenv("ENERGYCAST_BASE__SERVING__CHAMPION_MODEL", "linear_regression")
    get_settings.cache_clear()
    settings = get_settings()

    tracker = ExperimentTracker(settings.base.mlflow.tracking_uri, "serving-test")
    TrainingPipeline(prepare_data(_splits()), tracker).train_baseline(
        "linear_regression", linear_regression()
    )

    with TestClient(create_app(settings)) as test_client:
        yield test_client
    get_settings.cache_clear()


def _history_payload(hours: int = 400) -> dict:
    frame = _hourly_frame(hours)
    return {
        "history": [
            {"timestamp": ts.isoformat(), "value": float(value)}
            for ts, value in frame[TARGET].items()
        ]
    }


class TestApp:
    def test_health_names_the_loaded_champion(self, client):
        body = client.get("/health").json()

        assert body["status"] == "ok"
        assert body["champion"] == "linear_regression"
        assert body["run_id"]

    def test_predict_returns_twenty_four_hours(self, client):
        response = client.post("/predict", json=_history_payload())

        assert response.status_code == 200
        body = response.json()
        assert body["model"] == "linear_regression"
        assert len(body["forecast"]) == 24
        assert all(point["prediction_mw"] > 0 for point in body["forecast"])

    def test_predict_rejects_a_history_too_short(self, client):
        response = client.post("/predict", json=_history_payload(hours=10))

        assert response.status_code == 422

    def test_predict_unknown_model_is_not_found(self, client):
        payload = _history_payload()
        payload["model"] = "nonexistent"

        response = client.post("/predict", json=payload)

        assert response.status_code == 404
