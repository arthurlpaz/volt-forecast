"""The drift detector, the drift store, and the /drift route.

Drift is a distributional comparison against the training reference, so the
tests pin the Evidently wrapper on synthetic frames (a shifted column drifts, an
identical one does not), the store round-trip, and the route wiring behind a
stub detector so no test recomputes drift against the real dataset.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from energycast.config import get_settings
from energycast.data import ChronologicalSplitter
from energycast.drift.detector import DriftError, DriftResult, detect_drift
from energycast.drift.store import SQLiteDriftStore
from energycast.models import linear_regression
from energycast.serving import create_app
from energycast.serving.app import _drift
from energycast.training import ExperimentTracker, TrainingPipeline, prepare_data

TARGET = "PJME_MW"
REFERENCE_ROWS = 800
CURRENT_ROWS = 300


def _frame(rows: int, means: dict[str, float], *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2018-01-01", periods=rows, freq="h")
    return pd.DataFrame(
        {name: rng.normal(mean, 5000, rows) for name, mean in means.items()}, index=index
    )


def _result(share: float = 0.5, drifted: bool = True) -> DriftResult:
    return DriftResult(
        method="wasserstein",
        n_columns=2,
        n_drifted=1,
        share=share,
        threshold=0.5,
        drifted=drifted,
        per_column={"PJME_MW": 3.0, "lag_24h": 0.01},
        reference_rows=REFERENCE_ROWS,
        current_rows=CURRENT_ROWS,
    )


class TestDetectDrift:
    def test_identical_distributions_do_not_drift(self):
        means = {"PJME_MW": 30000.0, "lag_24h": 30000.0}
        reference = _frame(REFERENCE_ROWS, means, seed=1)
        current = _frame(CURRENT_ROWS, means, seed=2)

        result, _ = detect_drift(reference, current, method="wasserstein", drift_share=0.5)

        assert result.share == 0.0
        assert result.n_drifted == 0
        assert not result.drifted

    def test_every_shifted_column_drifts(self):
        reference = _frame(REFERENCE_ROWS, {"PJME_MW": 30000.0, "lag_24h": 30000.0}, seed=1)
        current = _frame(CURRENT_ROWS, {"PJME_MW": 55000.0, "lag_24h": 55000.0}, seed=2)

        result, _ = detect_drift(reference, current, method="wasserstein", drift_share=0.5)

        assert result.n_drifted == result.n_columns == 2
        assert result.share == 1.0
        assert result.drifted

    def test_a_single_shifted_column_is_isolated(self):
        reference = _frame(REFERENCE_ROWS, {"PJME_MW": 30000.0, "lag_24h": 30000.0}, seed=1)
        current = _frame(CURRENT_ROWS, {"PJME_MW": 55000.0, "lag_24h": 30000.0}, seed=2)

        result, _ = detect_drift(reference, current, method="wasserstein", drift_share=0.5)

        assert result.n_drifted == 1
        assert result.per_column["PJME_MW"] > result.per_column["lag_24h"]

    def test_an_empty_side_is_refused(self):
        reference = _frame(REFERENCE_ROWS, {"PJME_MW": 30000.0}, seed=1)
        empty = reference.iloc[:0]

        with pytest.raises(DriftError, match="non-empty"):
            detect_drift(reference, empty, method="wasserstein", drift_share=0.5)


class TestStore:
    def test_record_then_read_back_the_latest_run(self):
        store = SQLiteDriftStore(":memory:")
        run_at = datetime(2018, 8, 3, 0, 0, tzinfo=UTC)

        store.record_run(
            _result(share=0.75, drifted=True),
            run_at,
            pd.Timestamp("2018-07-04"),
            pd.Timestamp("2018-08-03"),
        )
        latest = store.latest()

        assert latest is not None
        assert latest.share == 0.75
        assert latest.drifted is True
        assert latest.method == "wasserstein"
        assert latest.window_end == pd.Timestamp("2018-08-03").isoformat()
        store.close()

    def test_latest_reflects_the_most_recent_run(self):
        store = SQLiteDriftStore(":memory:")
        window = (pd.Timestamp("2018-07-04"), pd.Timestamp("2018-08-03"))

        store.record_run(
            _result(share=0.1, drifted=False), datetime(2018, 8, 1, tzinfo=UTC), *window
        )
        store.record_run(
            _result(share=0.9, drifted=True), datetime(2018, 8, 2, tzinfo=UTC), *window
        )

        latest = store.latest()

        assert latest is not None
        assert latest.share == 0.9
        assert latest.drifted is True
        store.close()

    def test_latest_is_none_before_any_run(self):
        store = SQLiteDriftStore(":memory:")

        assert store.latest() is None
        store.close()


class _StubDetector:
    """Stands in for the real detector so the route test skips the dataset load."""

    def __init__(self, result: DriftResult) -> None:
        self.result = result

    def run(self) -> DriftResult:
        return self.result


def _hourly_frame(hours: int, start: str = "2015-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=hours, freq="h")
    t = np.arange(hours)
    signal = 20000 + 5000 * np.sin(2 * np.pi * t / 24) + 2000 * np.sin(2 * np.pi * t / 168)
    return pd.DataFrame({TARGET: signal}, index=index)


def _splits(hours: int = 2600):
    return ChronologicalSplitter(0.7, 0.15, 0.15).split(_hourly_frame(hours))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENERGYCAST_BASE__MLFLOW__TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setenv("ENERGYCAST_BASE__SERVING__CHAMPION_MODEL", "linear_regression")
    get_settings.cache_clear()
    settings = get_settings()

    tracker = ExperimentTracker(settings.base.mlflow.tracking_uri, "drift-test")
    TrainingPipeline(prepare_data(_splits()), tracker).train_baseline(
        "linear_regression", linear_regression()
    )

    app = create_app(settings)
    app.dependency_overrides[_drift] = lambda: _StubDetector(_result(share=0.75, drifted=True))
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()


class TestApp:
    def test_drift_route_returns_the_detector_result(self, client):
        body = client.get("/drift").json()

        assert body["drifted"] is True
        assert body["share"] == 0.75
        assert body["method"] == "wasserstein"
        assert {entry["column"] for entry in body["per_column"]} == {"PJME_MW", "lag_24h"}
