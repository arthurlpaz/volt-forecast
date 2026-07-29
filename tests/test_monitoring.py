"""The prediction store, the rolling metrics, and the /actuals + /metrics API.

A forecast is scored only once every hour it predicted has an actual, so the
tests pin the deferred join: log a forecast, reconcile some or all of its hours,
and check which anchors the rolling window keeps and which it drops.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from energycast.config import get_settings
from energycast.data import ChronologicalSplitter
from energycast.models import linear_regression
from energycast.monitoring import MonitoringError, SQLitePredictionStore, rolling_report
from energycast.serving import create_app
from energycast.serving.forecaster import Forecast
from energycast.training import ExperimentTracker, TrainingPipeline, prepare_data

TARGET = "PJME_MW"
HORIZON = 24


def _forecast(anchor: str, model: str = "lstm", base: float = 20000.0) -> Forecast:
    anchor_ts = pd.Timestamp(anchor)
    timestamps = pd.date_range(anchor_ts + pd.Timedelta(hours=1), periods=HORIZON, freq="h")
    values = base + np.arange(HORIZON, dtype=float)
    return Forecast(
        model=model,
        kind="lstm",
        run_id="run-1",
        anchor=anchor_ts,
        timestamps=timestamps,
        values=values,
    )


def _actuals(forecast: Forecast, offset: float = 0.0, drop_last: bool = False) -> dict:
    pairs = list(zip(forecast.timestamps, forecast.values, strict=True))
    if drop_last:
        pairs = pairs[:-1]
    return {ts: value + offset for ts, value in pairs}


class TestStore:
    def test_record_and_reconcile_an_anchor(self):
        store = SQLitePredictionStore(":memory:")
        forecast = _forecast("2018-01-10 00:00")
        store.record_forecast(forecast)

        reconciled = store.record_actuals(_actuals(forecast))

        assert reconciled == HORIZON
        window = store.reconciled_window("lstm", window_days=30)
        assert len(window) == HORIZON
        assert window["actual_mw"].notna().all()

    def test_actuals_for_unseen_hours_reconcile_nothing(self):
        store = SQLitePredictionStore(":memory:")
        store.record_forecast(_forecast("2018-01-10 00:00"))

        reconciled = store.record_actuals({pd.Timestamp("2001-01-01 00:00"): 1.0})

        assert reconciled == 0

    def test_reserving_an_anchor_keeps_its_reconciled_actuals(self):
        # An anchor served twice must not lose the actuals already matched to it:
        # the upsert updates the prediction and leaves actual_mw untouched.
        store = SQLitePredictionStore(":memory:")
        forecast = _forecast("2018-01-10 00:00")
        store.record_forecast(forecast)
        store.record_actuals(_actuals(forecast))

        store.record_forecast(forecast)

        window = store.reconciled_window("lstm", window_days=30)
        assert len(window) == HORIZON
        assert window["actual_mw"].notna().all()

    def test_window_drops_anchors_older_than_the_span(self):
        store = SQLitePredictionStore(":memory:")
        old = _forecast("2017-12-01 00:00")
        recent = _forecast("2018-01-10 00:00")
        for forecast in (old, recent):
            store.record_forecast(forecast)
            store.record_actuals(_actuals(forecast))

        window = store.reconciled_window("lstm", window_days=30)

        assert set(window["anchor"]) == {recent.anchor.isoformat()}


class TestRollingReport:
    def test_perfect_reconciliation_scores_zero(self):
        store = SQLitePredictionStore(":memory:")
        forecast = _forecast("2018-01-10 00:00")
        store.record_forecast(forecast)
        store.record_actuals(_actuals(forecast))

        report = rolling_report(store, "lstm", HORIZON, window_days=30)

        assert report.n_anchors == 1
        assert report.rmse.shape == (HORIZON,)
        assert report.rmse.max() == pytest.approx(0.0, abs=1e-6)
        assert report.mape.max() == pytest.approx(0.0, abs=1e-6)

    def test_partly_reconciled_anchor_is_excluded(self):
        # One anchor is missing its 24th actual; with no fully reconciled anchor
        # left, there is nothing to score.
        store = SQLitePredictionStore(":memory:")
        forecast = _forecast("2018-01-10 00:00")
        store.record_forecast(forecast)
        store.record_actuals(_actuals(forecast, drop_last=True))

        with pytest.raises(MonitoringError):
            rolling_report(store, "lstm", HORIZON, window_days=30)

    def test_constant_error_surfaces_as_that_error(self):
        store = SQLitePredictionStore(":memory:")
        forecast = _forecast("2018-01-10 00:00")
        store.record_forecast(forecast)
        store.record_actuals(_actuals(forecast, offset=100.0))

        report = rolling_report(store, "lstm", HORIZON, window_days=30)

        assert report.rmse.max() == pytest.approx(100.0)
        assert report.mae.max() == pytest.approx(100.0)


def _hourly_frame(hours: int, start: str = "2015-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=hours, freq="h")
    t = np.arange(hours)
    signal = 20000 + 5000 * np.sin(2 * np.pi * t / 24) + 2000 * np.sin(2 * np.pi * t / 168)
    return pd.DataFrame({TARGET: signal}, index=index)


def _splits(hours: int = 2600):
    return ChronologicalSplitter(0.7, 0.15, 0.15).split(_hourly_frame(hours))


def _history_payload(hours: int = 400) -> dict:
    frame = _hourly_frame(hours)
    return {
        "history": [
            {"timestamp": ts.isoformat(), "value": float(value)}
            for ts, value in frame[TARGET].items()
        ]
    }


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENERGYCAST_BASE__MLFLOW__TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setenv("ENERGYCAST_BASE__SERVING__CHAMPION_MODEL", "linear_regression")
    get_settings.cache_clear()
    settings = get_settings()

    tracker = ExperimentTracker(settings.base.mlflow.tracking_uri, "monitoring-test")
    TrainingPipeline(prepare_data(_splits()), tracker).train_baseline(
        "linear_regression", linear_regression()
    )

    with TestClient(create_app(settings)) as test_client:
        yield test_client
    get_settings.cache_clear()


class TestApp:
    def test_metrics_after_reconciling_a_served_forecast(self, client):
        served = client.post("/predict", json=_history_payload()).json()
        actuals = {
            "actuals": [
                {"timestamp": point["timestamp"], "value": point["prediction_mw"]}
                for point in served["forecast"]
            ]
        }

        reconciled = client.post("/actuals", json=actuals).json()
        metrics = client.get("/metrics").json()

        assert reconciled["reconciled"] == HORIZON
        assert metrics["model"] == "linear_regression"
        assert metrics["n_anchors"] == 1
        assert len(metrics["per_horizon"]) == HORIZON
        assert metrics["rmse_mean"] == pytest.approx(0.0, abs=1e-6)

    def test_metrics_before_any_actuals_is_not_found(self, client):
        client.post("/predict", json=_history_payload())

        assert client.get("/metrics").status_code == 404
