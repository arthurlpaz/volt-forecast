"""Tests for milestone 3 — calendar features, lags, scaling, sequences.

Synthetic frames cover the contracts; a final integration test runs the real
PJME file when it is present. The real file is gitignored, so that test skips
on a fresh clone and in CI rather than failing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energycast.config import get_settings
from energycast.data import ChronologicalSplitter, CSVDataLoader, TimeSeriesCleaner
from energycast.features import (
    CalendarFeatureBuilder,
    LagFeatureBuilder,
    ScalerError,
    SequenceBuilder,
    SequenceError,
    SeriesScaler,
)

DATETIME_COL = "Datetime"
TARGET_COL = "PJME_MW"
REAL_DATA = Path("data/raw/PJME_hourly.csv")


def _hourly_frame(hours: int, start: str = "2020-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=hours, freq="h")
    return pd.DataFrame({TARGET_COL: np.arange(hours, dtype=float)}, index=index)


class TestCalendarFeatureBuilder:
    def test_adds_raw_and_cyclical_columns(self):
        out = CalendarFeatureBuilder().build(_hourly_frame(48))

        for column in ("hour", "dayofweek", "month", "hour_sin", "hour_cos", "is_weekend"):
            assert column in out.columns

    def test_cyclical_encoding_puts_hour_23_next_to_hour_0(self):
        out = CalendarFeatureBuilder().build(_hourly_frame(48))
        point = out[["hour_sin", "hour_cos"]].to_numpy()

        def distance(a: int, b: int) -> float:
            return float(np.linalg.norm(point[a] - point[b]))

        assert distance(23, 0) == pytest.approx(distance(0, 1))
        assert abs(out["hour"].iloc[23] - out["hour"].iloc[0]) == 23

    def test_cyclical_pairs_stay_on_the_unit_circle(self):
        out = CalendarFeatureBuilder().build(_hourly_frame(24 * 400))

        for column in ("hour", "dayofweek", "month"):
            radius = out[f"{column}_sin"] ** 2 + out[f"{column}_cos"] ** 2
            assert radius.to_numpy() == pytest.approx(1.0)

    def test_month_cycle_wraps_december_onto_january(self):
        index = pd.DatetimeIndex(["2020-12-15", "2021-01-15", "2020-06-15"])
        frame = pd.DataFrame({TARGET_COL: [1.0, 2.0, 3.0]}, index=index)
        point = CalendarFeatureBuilder().build(frame)[["month_sin", "month_cos"]].to_numpy()

        december_to_january = np.linalg.norm(point[0] - point[1])
        december_to_june = np.linalg.norm(point[0] - point[2])
        assert december_to_january < december_to_june

    def test_weekend_flag_matches_the_calendar(self):
        out = CalendarFeatureBuilder().build(_hourly_frame(24 * 9, start="2020-01-04"))

        assert out["is_weekend"].iloc[0] == 1  # Saturday
        assert out["is_weekend"].iloc[48] == 0  # Monday

    def test_flags_christmas_as_a_holiday(self):
        out = CalendarFeatureBuilder().build(_hourly_frame(24 * 3, start="2020-12-24"))

        assert out.loc["2020-12-25 12:00", "is_holiday"] == 1
        assert out.loc["2020-12-24 12:00", "is_holiday"] == 0

    def test_holidays_can_be_switched_off(self):
        out = CalendarFeatureBuilder(include_holidays=False).build(_hourly_frame(24))
        assert "is_holiday" not in out.columns

    def test_empty_frame_does_not_ask_the_holiday_calendar_for_a_nat_range(self):
        empty = _hourly_frame(0)
        out = CalendarFeatureBuilder().build(empty)

        assert len(out) == 0
        assert "is_holiday" in out.columns

    def test_target_is_left_untouched(self):
        frame = _hourly_frame(48)
        out = CalendarFeatureBuilder().build(frame)

        pd.testing.assert_series_equal(out[TARGET_COL], frame[TARGET_COL])

    def test_non_datetime_index_is_rejected(self):
        with pytest.raises(TypeError, match="DatetimeIndex"):
            CalendarFeatureBuilder().build(pd.DataFrame({TARGET_COL: [1.0]}))


def _lagger(lags: list[int] | None = None, windows: list[int] | None = None) -> LagFeatureBuilder:
    return LagFeatureBuilder(TARGET_COL, lags or [1, 24], windows or [24])


class TestLagFeatureBuilder:
    def test_lag_reads_the_value_that_many_hours_back(self):
        out = _lagger().build(_hourly_frame(100))

        assert out["lag_1h"].iloc[50] == out[TARGET_COL].iloc[49]
        assert out["lag_24h"].iloc[50] == out[TARGET_COL].iloc[26]

    def test_history_before_the_frame_starts_is_nan_not_filled(self):
        out = _lagger().build(_hourly_frame(100))

        assert out["lag_24h"].iloc[:24].isna().all()
        assert out["lag_24h"].iloc[24:].notna().all()

    def test_lag_across_a_gap_is_measured_in_time_not_rows(self):
        # shift() counts rows, so with an hour missing it hands back a value
        # 25 hours old while claiming it is 24.
        frame = _hourly_frame(100).drop(index=pd.Timestamp("2020-01-02 00:00"))
        out = _lagger().build(frame)

        row = out.loc["2020-01-03 00:00"]
        assert row["lag_24h"] != frame[TARGET_COL].shift(24).loc["2020-01-03 00:00"]
        assert np.isnan(row["lag_24h"])  # that hour was dropped, so it is unknown

    def test_lag_landing_on_a_present_hour_still_resolves(self):
        frame = _hourly_frame(100).drop(index=pd.Timestamp("2020-01-02 00:00"))
        out = _lagger().build(frame)

        assert out.loc["2020-01-03 01:00", "lag_24h"] == frame.loc["2020-01-02 01:00", TARGET_COL]

    def test_rolling_mean_excludes_the_current_hour(self):
        out = _lagger(windows=[3]).build(_hourly_frame(100))

        expected = np.mean([47.0, 48.0, 49.0])
        assert out["rolling_mean_3h"].iloc[50] == pytest.approx(expected)

    def test_rolling_window_overlapping_a_gap_is_nan(self):
        frame = _hourly_frame(100).drop(index=pd.Timestamp("2020-01-02 00:00"))
        out = _lagger(windows=[3]).build(frame)

        assert np.isnan(out.loc["2020-01-02 02:00", "rolling_mean_3h"])
        assert not np.isnan(out.loc["2020-01-02 05:00", "rolling_mean_3h"])

    def test_rolling_std_is_added(self):
        out = _lagger(windows=[3]).build(_hourly_frame(100))
        assert out["rolling_std_3h"].iloc[50] == pytest.approx(np.std([47.0, 48.0, 49.0], ddof=1))

    def test_unsorted_index_is_rejected(self):
        with pytest.raises(ValueError, match="Refusing to lag an unsorted index"):
            _lagger().build(_hourly_frame(100).iloc[::-1])

    def test_missing_target_column_is_rejected(self):
        with pytest.raises(KeyError, match="not found"):
            _lagger().build(_hourly_frame(50).rename(columns={TARGET_COL: "other"}))

    def test_non_datetime_index_is_rejected(self):
        with pytest.raises(TypeError, match="DatetimeIndex"):
            _lagger().build(pd.DataFrame({TARGET_COL: [1.0]}))


class TestSeriesScaler:
    def test_standardises_to_zero_mean_unit_variance(self):
        scaled = SeriesScaler().fit_transform(_hourly_frame(100))

        assert scaled[TARGET_COL].mean() == pytest.approx(0.0, abs=1e-12)
        assert scaled[TARGET_COL].std() == pytest.approx(1.0)

    def test_validation_is_scaled_with_train_statistics(self):
        train, validation = _hourly_frame(100), _hourly_frame(50, start="2020-01-10")
        scaler = SeriesScaler().fit(train)

        scaled = scaler.transform(validation)
        expected = (validation[TARGET_COL] - train[TARGET_COL].mean()) / train[TARGET_COL].std()
        pd.testing.assert_series_equal(scaled[TARGET_COL], expected)
        # Fitting on validation too would have centred it; it must not be centred.
        assert abs(scaled[TARGET_COL].mean()) > 0.1

    def test_inverse_transform_round_trips(self):
        frame = _hourly_frame(100)
        scaler = SeriesScaler().fit(frame)

        scaled = scaler.transform(frame)[TARGET_COL].to_numpy()
        assert scaler.inverse_transform(scaled, TARGET_COL) == pytest.approx(
            frame[TARGET_COL].to_numpy()
        )

    def test_inverse_transform_accepts_model_output_shape(self):
        scaler = SeriesScaler().fit(_hourly_frame(100))
        predictions = np.zeros((5, 24))

        restored = scaler.inverse_transform(predictions, TARGET_COL)
        assert restored.shape == (5, 24)
        assert restored == pytest.approx(_hourly_frame(100)[TARGET_COL].mean())

    def test_only_named_columns_are_scaled(self):
        frame = _hourly_frame(100)
        frame["is_weekend"] = 1.0
        scaled = SeriesScaler(columns=[TARGET_COL]).fit_transform(frame)

        assert scaled["is_weekend"].eq(1.0).all()

    def test_transform_before_fit_is_rejected(self):
        with pytest.raises(ScalerError, match="not fitted"):
            SeriesScaler().transform(_hourly_frame(10))

    def test_inverse_transform_before_fit_is_rejected(self):
        with pytest.raises(ScalerError, match="not fitted"):
            SeriesScaler().inverse_transform(np.zeros(3), TARGET_COL)

    def test_constant_column_is_rejected_rather_than_divided_by_zero(self):
        frame = _hourly_frame(100)
        frame["flat"] = 1.0

        with pytest.raises(ScalerError, match="constant"):
            SeriesScaler().fit(frame)

    def test_fitting_an_absent_column_is_rejected(self):
        with pytest.raises(ScalerError, match="absent column"):
            SeriesScaler(columns=["nope"]).fit(_hourly_frame(10))

    def test_transforming_without_a_fitted_column_is_rejected(self):
        scaler = SeriesScaler().fit(_hourly_frame(10))

        with pytest.raises(ScalerError, match="absent"):
            scaler.transform(_hourly_frame(10).rename(columns={TARGET_COL: "other"}))

    def test_inverse_transform_of_an_unfitted_column_is_rejected(self):
        scaler = SeriesScaler(columns=[TARGET_COL]).fit(_hourly_frame(10))

        with pytest.raises(ScalerError, match="was not fitted"):
            scaler.inverse_transform(np.zeros(3), "other")

    def test_feature_names_reports_what_was_fitted(self):
        scaler = SeriesScaler(columns=[TARGET_COL]).fit(_hourly_frame(10))
        assert scaler.feature_names == [TARGET_COL]


def _builder(length: int = 4, horizon: int = 2) -> SequenceBuilder:
    return SequenceBuilder(length, horizon, TARGET_COL)


class TestSequenceBuilder:
    def test_window_shapes_follow_the_config(self):
        frame = _hourly_frame(100)
        frame["extra"] = 1.0
        dataset = _builder().build(frame)

        assert dataset.X.shape == (100 - 6 + 1, 4, 2)
        assert dataset.y.shape == (100 - 6 + 1, 2)
        assert dataset.feature_names == [TARGET_COL, "extra"]

    def test_x_holds_lookback_and_y_holds_the_horizon_that_follows(self):
        dataset = _builder().build(_hourly_frame(100))

        assert dataset.X[0, :, 0] == pytest.approx([0.0, 1.0, 2.0, 3.0])
        assert dataset.y[0] == pytest.approx([4.0, 5.0])
        assert dataset.X[1, :, 0] == pytest.approx([1.0, 2.0, 3.0, 4.0])

    def test_windows_are_float32_for_torch(self):
        dataset = _builder().build(_hourly_frame(100))

        assert dataset.X.dtype == np.float32
        assert dataset.y.dtype == np.float32

    def test_target_timestamps_mark_the_first_predicted_hour(self):
        dataset = _builder().build(_hourly_frame(100))

        assert dataset.target_timestamps[0] == pd.Timestamp("2020-01-01 04:00")
        assert len(dataset.target_timestamps) == len(dataset)

    def test_no_window_spans_a_gap(self):
        frame = _hourly_frame(100).drop(index=pd.Timestamp("2020-01-02 00:00"))
        dataset = _builder().build(frame)

        for stamp in dataset.target_timestamps:
            start = stamp - pd.Timedelta(hours=4)
            end = stamp + pd.Timedelta(hours=1)
            assert len(frame.loc[start:end]) == 6

    def test_gap_reduces_the_window_count(self):
        whole = _builder().build(_hourly_frame(100))
        gapped = _builder().build(_hourly_frame(100).drop(index=pd.Timestamp("2020-01-02 00:00")))

        assert len(gapped) < len(whole)

    def test_segments_shorter_than_one_window_are_skipped(self):
        frame = _hourly_frame(100)
        # Leave a 3-hour island in front of the gap; a window needs 6.
        frame = frame.drop(index=frame.index[3:40])
        dataset = _builder().build(frame)

        assert dataset.target_timestamps.min() > pd.Timestamp("2020-01-02 00:00")

    def test_nan_is_rejected_rather_than_windowed(self):
        frame = _hourly_frame(100)
        frame.iloc[0, 0] = np.nan

        with pytest.raises(SequenceError, match="contain NaN"):
            _builder().build(frame)

    def test_no_gap_free_stretch_is_a_clear_error(self):
        with pytest.raises(SequenceError, match="No gap-free stretch"):
            _builder().build(_hourly_frame(3))

    def test_unsorted_index_is_rejected(self):
        with pytest.raises(SequenceError, match="unsorted"):
            _builder().build(_hourly_frame(100).iloc[::-1])

    def test_missing_target_column_is_rejected(self):
        with pytest.raises(SequenceError, match="not found"):
            _builder().build(_hourly_frame(100).rename(columns={TARGET_COL: "other"}))

    def test_non_datetime_index_is_rejected(self):
        with pytest.raises(SequenceError, match="DatetimeIndex"):
            _builder().build(pd.DataFrame({TARGET_COL: [1.0] * 100}))

    def test_dropped_rows_become_a_segment_boundary(self):
        # Dropping the NaN head that lag features leave must not produce a
        # window spanning the hole.
        frame = _lagger(lags=[24], windows=[24]).build(_hourly_frame(200)).dropna()
        dataset = _builder().build(frame)

        assert dataset.target_timestamps.min() >= pd.Timestamp("2020-01-02 04:00")


class TestFromSettings:
    """`from_settings()` is the path production uses; the explicit constructors
    are only used by tests. These run without the data file present."""

    def test_lag_builder_wires_feature_config(self):
        settings = get_settings()
        builder = LagFeatureBuilder.from_settings()

        assert builder.target_column == settings.data.source.target_column
        assert builder.lags == settings.model.features.lags
        assert builder.rolling_windows == settings.model.features.rolling_windows
        assert builder.frequency == "h"

    def test_sequence_builder_wires_sequence_config(self):
        settings = get_settings()
        builder = SequenceBuilder.from_settings()

        assert builder.sequence_length == settings.model.sequence.sequence_length
        assert builder.prediction_horizon == settings.model.sequence.prediction_horizon
        assert builder.target_column == settings.data.source.target_column
        assert builder.window_length == 192

    def test_scaler_accepts_the_configured_type(self):
        assert SeriesScaler.from_settings(columns=[TARGET_COL]).columns == [TARGET_COL]

    def test_scaler_rejects_a_configured_type_nobody_implemented(self):
        # Reachable once someone widens the Literal in settings.py without
        # writing the scaler.
        settings = get_settings()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(settings.model.features, "scaler", "minmax")
            with pytest.raises(ScalerError, match="Unsupported scaler"):
                SeriesScaler.from_settings()


@pytest.mark.skipif(not REAL_DATA.exists(), reason="PJME data not downloaded (gitignored)")
class TestRealPJMEData:
    """End-to-end run against the actual file, whose quirks drove this design."""

    @staticmethod
    def _cleaned() -> pd.DataFrame:
        raw = CSVDataLoader(REAL_DATA, DATETIME_COL, TARGET_COL).load()
        return TimeSeriesCleaner(DATETIME_COL, TARGET_COL).clean(raw)

    def test_positional_shift_really_does_lie_on_this_file(self):
        # If the source ever ships gap-free, the grid reindex goes untested.
        cleaned = self._cleaned()
        out = LagFeatureBuilder(TARGET_COL, [24], []).build(cleaned)

        positional = cleaned[TARGET_COL].shift(24)
        both_known = positional.notna() & out["lag_24h"].notna()
        assert int((positional != out["lag_24h"])[both_known].sum()) == 690

    def test_full_feature_pipeline_per_split(self):
        splits = ChronologicalSplitter.from_settings().split(self._cleaned())
        calendar, lagger = CalendarFeatureBuilder(), LagFeatureBuilder.from_settings()
        builder = SequenceBuilder.from_settings()

        prepared = {
            name: lagger.build(calendar.build(getattr(splits, name))).dropna()
            for name in ("train", "validation", "test")
        }
        scaler = SeriesScaler(columns=[TARGET_COL]).fit(prepared["train"])
        datasets = {
            name: builder.build(scaler.transform(frame)) for name, frame in prepared.items()
        }

        for dataset in datasets.values():
            assert dataset.X.shape[1:] == (168, len(dataset.feature_names))
            assert dataset.y.shape[1] == 24

        # No validation window may reach back into hours the model trained on.
        train_end = splits.train.index.max()
        first_val = datasets["validation"].target_timestamps.min()
        assert first_val - pd.Timedelta(hours=168) > train_end

    def test_scaler_statistics_come_from_train_alone(self):
        splits = ChronologicalSplitter.from_settings().split(self._cleaned())
        scaler = SeriesScaler(columns=[TARGET_COL]).fit(splits.train)

        scaled_test = scaler.transform(splits.test)
        # Test load really does sit below train on this file; a scaler fitted
        # on everything would have absorbed the drift that milestone 10 hunts.
        assert scaled_test[TARGET_COL].mean() == pytest.approx(-0.202, abs=0.01)
