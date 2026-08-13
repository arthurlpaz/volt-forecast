"""The retrain trigger, promotion decision, store, registry aliases, and routes.

The pure decision is pinned on plain signals; the store and registry aliases on a
round-trip; the orchestrator check on seeded stores with no training; and one
end-to-end cycle retrains a baseline champion on synthetic data and exercises the
compare-and-promote path.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from energycast.config import get_settings
from energycast.config.settings import RetrainingConfig
from energycast.data import ChronologicalSplitter
from energycast.drift.detector import DriftResult
from energycast.drift.store import SQLiteDriftStore
from energycast.models import linear_regression
from energycast.monitoring import SQLitePredictionStore
from energycast.retraining import (
    RetrainingOrchestrator,
    SQLiteRetrainingStore,
    TriggerSignals,
    beats_champion,
    evaluate_triggers,
)
from energycast.retraining.orchestrator import RetrainingOutcome
from energycast.retraining.schemas import RetrainCheckResponse
from energycast.retraining.triggers import RetrainDecision
from energycast.serving import create_app
from energycast.serving.app import _retraining
from energycast.training import (
    CHALLENGER,
    CHAMPION,
    ExperimentTracker,
    TrainingPipeline,
    alias_version,
    prepare_data,
    registered_name,
    resolve_version,
)

TARGET = "PJME_MW"


def _config(**overrides) -> RetrainingConfig:
    base = {
        "database_path": ":memory:",
        "drift_triggers": True,
        "rolling_window_days": 30,
        "rmse_threshold": 2500.0,
        "new_observations_threshold": 720,
        "promotion_margin": 0.0,
    }
    return RetrainingConfig(**{**base, **overrides})


def _signals(drifted=None, rolling_rmse=None, new_observations=0) -> TriggerSignals:
    return TriggerSignals(
        drifted=drifted, rolling_rmse=rolling_rmse, new_observations=new_observations
    )


def _drift_result(drifted: bool = True) -> DriftResult:
    return DriftResult(
        method="wasserstein",
        n_columns=11,
        n_drifted=11 if drifted else 0,
        share=1.0 if drifted else 0.0,
        threshold=0.5,
        drifted=drifted,
        per_column={"PJME_MW": 1.5},
        reference_rows=90000,
        current_rows=720,
    )


class TestEvaluateTriggers:
    def test_no_signal_does_not_retrain(self):
        decision = evaluate_triggers(_signals(), _config())

        assert decision.should_retrain is False
        assert decision.reasons == []

    def test_drift_alone_trips_it(self):
        decision = evaluate_triggers(_signals(drifted=True), _config())

        assert decision.should_retrain is True
        assert decision.reasons == ["drift"]

    def test_drift_is_ignored_when_disabled(self):
        decision = evaluate_triggers(_signals(drifted=True), _config(drift_triggers=False))

        assert decision.should_retrain is False

    def test_rolling_error_over_threshold_trips_it(self):
        decision = evaluate_triggers(_signals(rolling_rmse=3000.0), _config())

        assert decision.should_retrain is True
        assert "rmse" in decision.reasons[0]

    def test_new_observations_over_threshold_trips_it(self):
        decision = evaluate_triggers(_signals(new_observations=1000), _config())

        assert decision.should_retrain is True
        assert "new_observations" in decision.reasons[0]

    def test_several_conditions_are_all_reported(self):
        decision = evaluate_triggers(
            _signals(drifted=True, rolling_rmse=3000.0, new_observations=1000), _config()
        )

        assert decision.should_retrain is True
        assert len(decision.reasons) == 3


class TestBeatsChampion:
    def test_a_lower_rmse_wins(self):
        assert beats_champion(2000.0, 1900.0, margin=0.0) is True

    def test_a_tie_does_not_win(self):
        assert beats_champion(2000.0, 2000.0, margin=0.0) is False

    def test_the_margin_must_be_cleared(self):
        assert beats_champion(2000.0, 1850.0, margin=0.1) is False  # needs < 1800
        assert beats_champion(2000.0, 1750.0, margin=0.1) is True


class TestStore:
    def test_a_triggered_cycle_sets_the_cutoff(self):
        store = SQLiteRetrainingStore(":memory:")
        decision = RetrainDecision(should_retrain=True, reasons=["drift"])

        store.record_cycle(
            datetime(2018, 8, 3, tzinfo=UTC),
            decision,
            pd.Timestamp("2018-08-03"),
            "lstm",
            "lstm@v2",
            True,
        )

        assert store.last_trained_cutoff() == pd.Timestamp("2018-08-03").isoformat()
        latest = store.latest()
        assert latest is not None
        assert latest.triggered is True
        assert latest.promoted is True
        assert latest.champion_after == "lstm@v2"
        store.close()

    def test_a_skipped_cycle_leaves_no_cutoff(self):
        store = SQLiteRetrainingStore(":memory:")
        decision = RetrainDecision(should_retrain=False, reasons=[])

        store.record_cycle(
            datetime(2018, 8, 1, tzinfo=UTC),
            decision,
            pd.Timestamp("2018-08-01"),
            "lstm",
            "lstm",
            False,
        )

        assert store.last_trained_cutoff() is None
        assert store.latest() is not None
        store.close()

    def test_empty_store_reads_none(self):
        store = SQLiteRetrainingStore(":memory:")

        assert store.latest() is None
        assert store.last_trained_cutoff() is None
        store.close()


def _hourly_frame(hours: int, start: str = "2015-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=hours, freq="h")
    t = np.arange(hours)
    signal = 20000 + 5000 * np.sin(2 * np.pi * t / 24) + 2000 * np.sin(2 * np.pi * t / 168)
    return pd.DataFrame({TARGET: signal}, index=index)


def _splits(hours: int = 2600):
    return ChronologicalSplitter(0.7, 0.15, 0.15).split(_hourly_frame(hours))


@pytest.fixture
def mlflow_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENERGYCAST_BASE__MLFLOW__TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setenv("ENERGYCAST_BASE__SERVING__CHAMPION_MODEL", "linear_regression")
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


class TestRegistryAlias:
    def test_alias_round_trips_and_resolves(self, mlflow_env):
        settings = mlflow_env
        tracker = ExperimentTracker(settings.base.mlflow.tracking_uri, "alias-test")
        version = TrainingPipeline(prepare_data(_splits()), tracker).train_baseline(
            "linear_regression", linear_regression()
        )
        name = registered_name("linear_regression")

        from energycast.training import set_alias

        set_alias(name, CHAMPION, version)

        assert alias_version(name, CHAMPION) == version
        assert resolve_version(name, CHAMPION) == version
        assert alias_version(name, CHALLENGER) is None
        assert resolve_version(name, CHALLENGER) == version  # falls back to latest


class TestCheck:
    def test_a_drifted_run_makes_check_advise_retraining(self):
        get_settings.cache_clear()
        drift = SQLiteDriftStore(":memory:")
        drift.record_run(
            _drift_result(drifted=True),
            datetime(2018, 8, 3, tzinfo=UTC),
            pd.Timestamp("2018-07-04"),
            pd.Timestamp("2018-08-03"),
        )
        orchestrator = RetrainingOrchestrator(
            drift, SQLitePredictionStore(":memory:"), SQLiteRetrainingStore(":memory:")
        )

        signals, decision = orchestrator.check()

        assert signals.drifted is True
        assert signals.rolling_rmse is None  # nothing reconciled to score
        assert decision.should_retrain is True
        assert decision.reasons == ["drift"]
        orchestrator.close()

    def test_quiet_stores_advise_no_retraining(self):
        get_settings.cache_clear()
        orchestrator = RetrainingOrchestrator(
            SQLiteDriftStore(":memory:"),
            SQLitePredictionStore(":memory:"),
            SQLiteRetrainingStore(":memory:"),
        )

        _, decision = orchestrator.check()

        assert decision.should_retrain is False
        orchestrator.close()


class TestRunCycle:
    def test_a_triggered_cycle_trains_and_records_a_challenger(self, mlflow_env, monkeypatch):
        settings = mlflow_env
        splits = _splits()
        tracker = ExperimentTracker(settings.base.mlflow.tracking_uri, "cycle-test")
        TrainingPipeline(prepare_data(splits), tracker).train_baseline(
            "linear_regression", linear_regression()
        )
        monkeypatch.setattr("energycast.retraining.orchestrator.load_splits", lambda: splits)

        drift = SQLiteDriftStore(":memory:")
        drift.record_run(
            _drift_result(drifted=True),
            datetime(2018, 8, 3, tzinfo=UTC),
            pd.Timestamp("2018-07-04"),
            pd.Timestamp("2018-08-03"),
        )
        retraining_store = SQLiteRetrainingStore(":memory:")
        orchestrator = RetrainingOrchestrator(
            drift, SQLitePredictionStore(":memory:"), retraining_store, settings
        )

        outcome = orchestrator.run_cycle()

        assert outcome.triggered is True
        assert "drift" in outcome.reasons
        assert outcome.challenger_version == "2"
        # Retraining identical data reproduces the model, so it does not beat itself.
        assert outcome.promoted is False
        name = registered_name("linear_regression")
        # The champion alias must pin the incumbent, never fall back to the latest
        # version, which is now the unpromoted challenger.
        assert alias_version(name, CHAMPION) == "1"
        assert alias_version(name, CHALLENGER) == "2"
        assert retraining_store.latest().triggered is True
        orchestrator.close()


class _StubOrchestrator:
    """Returns a canned check so the route test skips signal collection."""

    def check(self):
        signals = TriggerSignals(drifted=True, rolling_rmse=3100.0, new_observations=50)
        return signals, RetrainDecision(
            should_retrain=True, reasons=["drift", "rmse 3100.0 > 2500.0"]
        )


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENERGYCAST_BASE__MLFLOW__TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setenv("ENERGYCAST_BASE__SERVING__CHAMPION_MODEL", "linear_regression")
    get_settings.cache_clear()
    settings = get_settings()

    tracker = ExperimentTracker(settings.base.mlflow.tracking_uri, "retraining-route-test")
    TrainingPipeline(prepare_data(_splits()), tracker).train_baseline(
        "linear_regression", linear_regression()
    )

    app = create_app(settings)
    app.dependency_overrides[_retraining] = _StubOrchestrator
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()


class TestApp:
    def test_check_route_reports_the_decision(self, client):
        body = client.get("/retrain/check").json()

        assert body["should_retrain"] is True
        assert body["drifted"] is True
        assert body["rolling_rmse"] == 3100.0
        assert "drift" in body["reasons"]


def test_check_response_maps_a_decision():
    signals = TriggerSignals(drifted=False, rolling_rmse=None, new_observations=10)
    decision = RetrainDecision(should_retrain=False, reasons=[])

    body = RetrainCheckResponse.from_check(signals, decision)

    assert body.should_retrain is False
    assert body.rolling_rmse is None
    assert body.new_observations == 10


def test_outcome_is_frozen():
    outcome = RetrainingOutcome(True, ["drift"], "lstm", "lstm@v2", True, "2")

    assert outcome.promoted is True
