"""Per-horizon metrics, roster alignment on shared rows, and champion choice.

The synthetic signal is exactly periodic with a 168-hour season (24 divides
168), so a correctly aligned seasonal naive scores ~0 MW here: that is what
pins down both the season and the one-hour anchor shift the naive needs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from energycast.data import ChronologicalSplitter
from energycast.evaluation import (
    NAIVE,
    EvaluationError,
    Evaluator,
    MetricError,
    ModelScore,
    log_evaluation,
    mae_per_horizon,
    mape_per_horizon,
    rmse_per_horizon,
)
from energycast.models import linear_regression
from energycast.training import (
    ExperimentTracker,
    LoadedModel,
    ModelMeta,
    TrainingPipeline,
    load_registered,
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


class _Constant:
    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def predict(self, X) -> np.ndarray:  # noqa: N803
        return self._values


def _stub(kind: str, values: np.ndarray, scaler) -> LoadedModel:
    meta = ModelMeta(
        kind=kind,
        name="stub",
        feature_names=[],
        target_column=TARGET,
        prediction_horizon=24,
        sequence_length=None,
        hyperparameters={},
    )
    return LoadedModel(model=_Constant(values), scaler=scaler, meta=meta, run_id="stub")


class TestMetrics:
    def test_per_horizon_returns_one_score_per_column(self):
        y_true = np.array([[100.0, 200.0], [150.0, 250.0]])
        y_pred = np.array([[110.0, 190.0], [140.0, 260.0]])

        assert rmse_per_horizon(y_true, y_pred).shape == (2,)
        np.testing.assert_allclose(rmse_per_horizon(y_true, y_pred), [10.0, 10.0])
        np.testing.assert_allclose(mae_per_horizon(y_true, y_pred), [10.0, 10.0])

    def test_shape_mismatch_raises(self):
        with pytest.raises(MetricError):
            rmse_per_horizon(np.zeros((3, 2)), np.zeros((3, 4)))

    def test_mape_undefined_at_zero_raises(self):
        with pytest.raises(MetricError):
            mape_per_horizon(np.array([[0.0, 1.0]]), np.array([[1.0, 1.0]]))


@pytest.fixture
def evaluated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = ExperimentTracker(f"sqlite:///{tmp_path}/mlflow.db", "eval-test")
    prepared = prepare_data(_splits())
    TrainingPipeline(prepared, tracker).train_baseline("linear_regression", linear_regression())
    loaded = {"linear_regression": load_registered("energycast-linear_regression")}
    report = Evaluator(prepared).evaluate(loaded)
    return prepared, loaded, report


class TestEvaluator:
    def test_scores_every_model_on_the_shared_rows(self, evaluated):
        prepared, _, report = evaluated
        common = prepared.sequence["test"].target_timestamps.intersection(
            prepared.tabular["test"].target_timestamps
        )

        assert report.n_rows == len(common)
        assert NAIVE in report.scores
        for score in report.scores.values():
            assert score.rmse.shape == (24,)
            assert np.isfinite(score.rmse).all()

    def test_naive_is_near_perfect_on_a_periodic_signal(self, evaluated):
        # A wrong season or a missing anchor shift would break the repeat and lift this.
        _, _, report = evaluated
        assert report.scores[NAIVE].rmse_mean < 1.0

    def test_learned_predictions_are_in_mw_not_z_units(self, evaluated):
        # Forgetting the inverse transform would leave predictions near z-units
        # against a ~20000 MW truth, sending RMSE toward 20000.
        _, _, report = evaluated
        assert 0.0 < report.scores["linear_regression"].rmse_mean < 10000.0

    def test_champion_is_the_only_learned_model(self, evaluated):
        _, _, report = evaluated
        assert report.champion == "linear_regression"

    def test_champion_is_the_lowest_mean_rmse_learned_model(self):
        prepared = prepare_data(_splits())
        y_scaled = prepared.tabular["test"].y
        good = _stub("sklearn", y_scaled, prepared.scaler)
        bad = _stub("sklearn", np.zeros_like(y_scaled), prepared.scaler)

        report = Evaluator(prepared).evaluate({"bad": bad, "good": good})

        assert report.champion == "good"
        assert report.scores["good"].rmse_mean < report.scores["bad"].rmse_mean

    def test_empty_intersection_raises(self):
        prepared = prepare_data(_splits())
        object.__setattr__(prepared.sequence["test"], "target_timestamps", pd.DatetimeIndex([]))
        with pytest.raises(EvaluationError):
            Evaluator(prepared).evaluate({})


class TestLogEvaluation:
    def test_metrics_hang_on_the_training_run(self, evaluated):
        _, loaded, report = evaluated
        run_id = loaded["linear_regression"].run_id

        log_evaluation(report)

        run = MlflowClient().get_run(run_id)
        assert "test_rmse_h1" in run.data.metrics
        assert run.data.metrics["test_rmse_mean"] == pytest.approx(
            report.scores["linear_regression"].rmse_mean, rel=1e-6
        )
        assert run.data.tags["is_champion"] == "true"

    def test_naive_without_a_run_is_skipped(self):
        naive = ModelScore(NAIVE, NAIVE, np.ones(24), np.ones(24), np.ones(24), run_id=None)
        report = _report_with({NAIVE: naive})
        log_evaluation(report)


def _report_with(scores: dict[str, ModelScore]):
    from energycast.evaluation import EvaluationReport

    return EvaluationReport(
        scores=scores, champion=NAIVE, target_timestamps=pd.DatetimeIndex([pd.Timestamp("2015")])
    )
