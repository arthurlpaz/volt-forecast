"""Tests for milestone 4 — horizon targets, the naive bar, and the baselines.

Synthetic frames cover the contracts; a final integration test runs the real
PJME file when it is present. The real file is gitignored, so that test skips
on a fresh clone and in CI rather than failing.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.multioutput import MultiOutputRegressor

from energycast.config import get_settings
from energycast.data import ChronologicalSplitter, CSVDataLoader, TimeSeriesCleaner
from energycast.features import (
    CalendarFeatureBuilder,
    HorizonTargetBuilder,
    LagFeatureBuilder,
    SequenceBuilder,
    TargetError,
)
from energycast.models import (
    LSTMError,
    LSTMForecaster,
    Model,
    NaiveModelError,
    SeasonalNaiveModel,
    SklearnBaseline,
    build_from_settings,
    lightgbm,
    linear_regression,
    random_forest,
    xgboost,
)

TARGET_COL = "PJME_MW"
DATETIME_COL = "Datetime"
REAL_DATA = Path("data/raw/PJME_hourly.csv")


def _hourly_frame(hours: int, start: str = "2020-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=hours, freq="h")
    return pd.DataFrame({TARGET_COL: np.arange(hours, dtype=float)}, index=index)


def _builder(horizon: int = 3) -> HorizonTargetBuilder:
    return HorizonTargetBuilder(TARGET_COL, horizon)


class TestHorizonTargetBuilder:
    def test_target_holds_the_hours_that_follow(self):
        dataset = _builder().build(_hourly_frame(100))

        assert dataset.y[0] == pytest.approx([1.0, 2.0, 3.0])
        assert dataset.y[10] == pytest.approx([11.0, 12.0, 13.0])

    def test_shapes_follow_the_horizon(self):
        dataset = _builder(horizon=24).build(_hourly_frame(100))

        assert dataset.y.shape == (100 - 24, 24)
        assert len(dataset.X) == 100 - 24

    def test_rows_whose_future_runs_past_the_end_are_dropped(self):
        dataset = _builder().build(_hourly_frame(100))

        assert dataset.X.index.max() == pd.Timestamp("2020-01-05 00:00")
        assert len(dataset) == 97

    def test_target_timestamps_mark_the_first_predicted_hour(self):
        dataset = _builder().build(_hourly_frame(100))

        assert dataset.target_timestamps[0] == pd.Timestamp("2020-01-01 01:00")

    def test_horizon_is_measured_in_time_not_rows(self):
        # The mirror of the 690-row bug: shift(-3) counts rows, so with an hour
        # missing it pairs t with a target 4 hours away while claiming 3.
        frame = _hourly_frame(100).drop(index=pd.Timestamp("2020-01-01 05:00"))
        dataset = _builder().build(frame)

        assert pd.Timestamp("2020-01-01 02:00") not in dataset.X.index
        row = dataset.X.index.get_loc(pd.Timestamp("2020-01-01 06:00"))
        assert dataset.y[row] == pytest.approx([7.0, 8.0, 9.0])

    def test_row_whose_horizon_crosses_a_gap_is_dropped(self):
        frame = _hourly_frame(100).drop(index=pd.Timestamp("2020-01-01 05:00"))
        dataset = _builder().build(frame)

        for missed in ("2020-01-01 02:00", "2020-01-01 03:00", "2020-01-01 04:00"):
            assert pd.Timestamp(missed) not in dataset.X.index
        assert pd.Timestamp("2020-01-01 01:00") in dataset.X.index

    def test_features_keep_the_current_hour_value(self):
        # At hour t the value of t is measured, so it is legitimate input for
        # t+1 — and the strongest single feature available.
        dataset = _builder().build(_hourly_frame(100))

        assert TARGET_COL in dataset.feature_names
        assert dataset.X[TARGET_COL].iloc[0] == 0.0

    def test_nan_is_rejected_rather_than_paired(self):
        frame = _hourly_frame(100)
        frame.iloc[0, 0] = np.nan

        with pytest.raises(TargetError, match="contain NaN"):
            _builder().build(frame)

    def test_frame_shorter_than_the_horizon_is_a_clear_error(self):
        with pytest.raises(TargetError, match="gap-free"):
            _builder(horizon=24).build(_hourly_frame(10))

    def test_unsorted_index_is_rejected(self):
        with pytest.raises(TargetError, match="unsorted"):
            _builder().build(_hourly_frame(100).iloc[::-1])

    def test_non_datetime_index_is_rejected(self):
        with pytest.raises(TargetError, match="DatetimeIndex"):
            _builder().build(pd.DataFrame({TARGET_COL: [1.0] * 100}))

    def test_missing_target_column_is_rejected(self):
        with pytest.raises(TargetError, match="not found"):
            _builder().build(_hourly_frame(100).rename(columns={TARGET_COL: "other"}))

    def test_from_settings_wires_the_horizon(self):
        settings = get_settings()
        builder = HorizonTargetBuilder.from_settings()

        assert builder.target_column == settings.data.source.target_column
        assert builder.prediction_horizon == settings.model.sequence.prediction_horizon
        assert builder.frequency == "h"


class TestSeasonalNaiveModel:
    def test_predicts_the_same_hour_one_season_back(self):
        frame = _hourly_frame(400)
        model = SeasonalNaiveModel(frame[TARGET_COL], prediction_horizon=3, season_length=168)
        X = frame.iloc[200:210]

        predictions = model.predict(X)
        # Row 200 predicts hours 201..203, each read from 168 hours earlier.
        assert predictions[0] == pytest.approx([201 - 168, 202 - 168, 203 - 168])

    def test_satisfies_the_model_protocol(self):
        frame = _hourly_frame(400)
        model = SeasonalNaiveModel(frame[TARGET_COL], prediction_horizon=3)

        assert isinstance(model, Model)
        assert model.fit(frame, np.zeros((len(frame), 3))) is model

    def test_shape_matches_the_horizon(self):
        frame = _hourly_frame(400)
        model = SeasonalNaiveModel(frame[TARGET_COL], prediction_horizon=24)

        assert model.predict(frame.iloc[200:250]).shape == (50, 24)

    def test_season_shorter_than_the_horizon_is_refused(self):
        # season == horizon would read the very hour being predicted.
        with pytest.raises(NaiveModelError, match="must exceed"):
            SeasonalNaiveModel(
                _hourly_frame(400)[TARGET_COL], prediction_horizon=24, season_length=24
            )

    def test_hours_without_a_season_of_history_are_refused(self):
        frame = _hourly_frame(400)
        model = SeasonalNaiveModel(frame[TARGET_COL], prediction_horizon=3, season_length=168)

        with pytest.raises(NaiveModelError, match="does not reach far enough"):
            model.predict(frame.iloc[:10])

    def test_non_datetime_index_is_rejected(self):
        model = SeasonalNaiveModel(_hourly_frame(400)[TARGET_COL], prediction_horizon=3)

        with pytest.raises(NaiveModelError, match="DatetimeIndex"):
            model.predict(pd.DataFrame({TARGET_COL: [1.0] * 10}))

    def test_from_settings_wires_the_horizon(self):
        model = SeasonalNaiveModel.from_settings(_hourly_frame(400)[TARGET_COL])

        assert model.prediction_horizon == get_settings().model.sequence.prediction_horizon
        assert model.season_length == 168
        assert model.name == "seasonal_naive_168h"


def _tiny_training_set(horizon: int = 3) -> tuple[pd.DataFrame, np.ndarray]:
    dataset = HorizonTargetBuilder(TARGET_COL, horizon).build(
        CalendarFeatureBuilder(include_holidays=False).build(_hourly_frame(300))
    )
    return dataset.X, dataset.y


class TestBaselines:
    @pytest.mark.parametrize(
        "factory",
        [
            pytest.param(lambda: linear_regression(), id="linear_regression"),
            pytest.param(
                lambda: random_forest(0, 1, n_estimators=5, max_depth=3), id="random_forest"
            ),
            pytest.param(lambda: xgboost(0, 1, n_estimators=5, max_depth=3), id="xgboost"),
            pytest.param(lambda: lightgbm(0, 1, n_estimators=5), id="lightgbm"),
        ],
    )
    def test_every_baseline_predicts_the_full_horizon(self, factory):
        X, y = _tiny_training_set()
        model = factory()

        assert isinstance(model, Model)
        assert model.fit(X, y) is model

        predictions = model.predict(X)
        assert predictions.shape == y.shape
        assert np.isfinite(predictions).all()

    def test_baseline_carries_a_name_for_the_registry(self):
        assert linear_regression().name == "linear_regression"
        assert xgboost(0, 1, n_estimators=5).name == "xgboost"

    def test_wrapper_passes_hyperparameters_through(self):
        model = random_forest(42, 1, n_estimators=7, max_depth=3)

        assert model.estimator.n_estimators == 7
        assert model.estimator.random_state == 42

    def test_lightgbm_is_wrapped_for_multi_output(self):
        # LightGBM has no native multi-output. Without the wrapper it raises on
        # a 2D y instead of fitting one model per horizon step.
        model = lightgbm(0, 1, n_estimators=5)
        assert isinstance(model.estimator, MultiOutputRegressor)

        X, y = _tiny_training_set()
        with pytest.raises(TypeError, match="Wrong type"):
            lgb.LGBMRegressor(n_estimators=5, verbose=-1).fit(X, y)
        assert model.fit(X, y).predict(X).shape == y.shape

    def test_xgboost_uses_native_multi_output(self):
        model = xgboost(0, 1, n_estimators=5)
        assert model.estimator.multi_strategy == "one_output_per_tree"

    def test_build_from_settings_returns_every_configured_baseline(self):
        models = build_from_settings()
        settings = get_settings()

        assert set(models) == {"linear_regression", "random_forest", "xgboost", "lightgbm"}
        assert all(isinstance(m, SklearnBaseline) for m in models.values())
        assert models["random_forest"].estimator.n_estimators == (
            settings.model.baselines.random_forest.hyperparameters()["n_estimators"]
        )
        assert models["xgboost"].estimator.random_state == settings.model.baselines.random_state


def _seasonal_windows(
    hours: int = 500, sequence_length: int = 24, horizon: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    index = pd.date_range("2020-01-01", periods=hours, freq="h")
    signal = np.sin(2 * np.pi * np.arange(hours) / 24.0)
    frame = pd.DataFrame({TARGET_COL: signal.astype("float32")}, index=index)
    dataset = SequenceBuilder(sequence_length, horizon, TARGET_COL).build(frame)
    return dataset.X, dataset.y


def _forecaster(horizon: int = 3, **overrides) -> LSTMForecaster:
    defaults = dict(
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
        learning_rate=0.01,
        batch_size=32,
        max_epochs=40,
        early_stopping_patience=40,
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=5,
        seed=42,
    )
    defaults.update(overrides)
    return LSTMForecaster(**defaults)


class TestLSTMForecaster:
    def test_predicts_the_full_horizon_and_satisfies_the_protocol(self):
        X, y = _seasonal_windows()
        model = _forecaster()

        assert isinstance(model, Model)
        assert model.fit(X, y) is model

        predictions = model.predict(X)
        assert predictions.shape == y.shape
        assert np.isfinite(predictions).all()

    def test_learns_a_seasonal_signal_better_than_its_mean(self):
        X, y = _seasonal_windows()
        predictions = _forecaster().fit(X, y).predict(X)

        fitted_rmse = float(np.sqrt(np.mean((predictions - y) ** 2)))
        mean_rmse = float(np.sqrt(np.mean((y.mean() - y) ** 2)))
        assert fitted_rmse < mean_rmse

    def test_same_seed_reproduces_predictions(self):
        X, y = _seasonal_windows()

        first = _forecaster().fit(X, y).predict(X)
        second = _forecaster().fit(X, y).predict(X)
        assert np.array_equal(first, second)

    def test_explicit_validation_split_is_used(self):
        X, y = _seasonal_windows()
        cut = 400
        model = _forecaster().fit(X[:cut], y[:cut], X[cut:], y[cut:])

        assert model.predict(X[cut:]).shape == y[cut:].shape

    def test_predict_before_fit_is_refused(self):
        with pytest.raises(LSTMError, match="Call fit before predict"):
            _forecaster().predict(_seasonal_windows()[0])

    def test_two_dimensional_input_is_refused(self):
        X, y = _seasonal_windows()

        with pytest.raises(LSTMError, match="sequence_length, n_features"):
            _forecaster().fit(X.reshape(len(X), -1), y)

    def test_too_few_windows_to_hold_out_a_tail_is_a_clear_error(self):
        X, y = _seasonal_windows()

        with pytest.raises(LSTMError, match="cannot be split"):
            _forecaster(validation_fraction=0.0).fit(X, y)

    def test_from_settings_wires_the_config(self):
        settings = get_settings().model.lstm
        model = LSTMForecaster.from_settings()

        assert model.name == "lstm"
        assert model.hidden_size == settings.hidden_size
        assert model.seed == settings.seed
        assert model.lr_scheduler_patience == settings.lr_scheduler.patience


@pytest.mark.skipif(not REAL_DATA.exists(), reason="PJME data not downloaded (gitignored)")
class TestRealPJMEData:
    """The comparison milestone 7 will run, on the file that motivated it."""

    @staticmethod
    def _splits():
        raw = CSVDataLoader(REAL_DATA, DATETIME_COL, TARGET_COL).load()
        cleaned = TimeSeriesCleaner(DATETIME_COL, TARGET_COL).clean(raw)
        return cleaned, ChronologicalSplitter.from_settings().split(cleaned)

    def test_horizon_target_drops_rows_around_the_real_gaps(self):
        cleaned, splits = self._splits()
        prepared = (
            LagFeatureBuilder.from_settings()
            .build(CalendarFeatureBuilder().build(splits.test))
            .dropna()
        )
        dataset = HorizonTargetBuilder.from_settings().build(prepared)

        # 24 rows lost to the end of the split, plus rows whose 24-hour future
        # crosses one of the real gaps.
        assert len(dataset) < len(prepared) - 24
        assert dataset.y.shape[1] == 24

    def test_naive_beats_nothing_but_bounds_everything(self):
        cleaned, splits = self._splits()
        prepared = (
            LagFeatureBuilder.from_settings()
            .build(CalendarFeatureBuilder().build(splits.test))
            .dropna()
        )
        dataset = HorizonTargetBuilder.from_settings().build(prepared)

        naive = SeasonalNaiveModel.from_settings(cleaned[TARGET_COL])
        rmse = float(np.sqrt(np.mean((naive.predict(dataset.X) - dataset.y) ** 2)))

        # Measured 4,760.9 MW. A naive built by repeating lag_168h across the
        # horizon scores 7,532 — a floor 1.58x too generous to mean anything.
        assert rmse == pytest.approx(4760.9, abs=50)

    def test_linear_baseline_beats_the_naive_bar(self):
        cleaned, splits = self._splits()
        cal, lag = CalendarFeatureBuilder(), LagFeatureBuilder.from_settings()
        builder = HorizonTargetBuilder.from_settings()

        train = builder.build(lag.build(cal.build(splits.train)).dropna())
        test = builder.build(lag.build(cal.build(splits.test)).dropna())

        model = linear_regression().fit(train.X, train.y)
        rmse = float(np.sqrt(np.mean((model.predict(test.X) - test.y) ** 2)))

        naive = SeasonalNaiveModel.from_settings(cleaned[TARGET_COL])
        naive_rmse = float(np.sqrt(np.mean((naive.predict(test.X) - test.y) ** 2)))

        assert rmse < naive_rmse
        assert rmse == pytest.approx(2699.3, abs=100)
