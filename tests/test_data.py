"""Tests for milestone 2 — data ingestion, cleaning, validation, splitting.

Synthetic frames cover the contracts; a final integration test runs the real
PJME file when it is present. The real file is gitignored, so that test skips
on a fresh clone and in CI rather than failing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from energycast.config import get_settings
from energycast.data import (
    ChronologicalSplitter,
    CSVDataLoader,
    DataLoader,
    DataSourceError,
    DataValidationError,
    SchemaValidator,
    SplitError,
    TimeSeriesCleaner,
)

DATETIME_COL = "Datetime"
TARGET_COL = "PJME_MW"
REAL_DATA = Path("data/raw/PJME_hourly.csv")


def _hourly_frame(hours: int, start: str = "2020-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=hours, freq="h")
    return pd.DataFrame({TARGET_COL: range(hours)}, index=index)


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    path = tmp_path / "sample.csv"
    frame = _hourly_frame(48).reset_index(names=DATETIME_COL)
    frame.to_csv(path, index=False)
    return path


class TestCSVDataLoader:
    def test_satisfies_the_dataloader_protocol(self, csv_path: Path):
        assert isinstance(CSVDataLoader(csv_path, DATETIME_COL, TARGET_COL), DataLoader)

    def test_loads_and_parses_datetime(self, csv_path: Path):
        frame = CSVDataLoader(csv_path, DATETIME_COL, TARGET_COL).load()

        assert len(frame) == 48
        assert pd.api.types.is_datetime64_any_dtype(frame[DATETIME_COL])

    def test_missing_file_names_the_path(self, tmp_path: Path):
        loader = CSVDataLoader(tmp_path / "nope.csv", DATETIME_COL, TARGET_COL)
        with pytest.raises(DataSourceError, match="not found"):
            loader.load()

    def test_missing_column_is_reported_before_parsing(self, tmp_path: Path):
        path = tmp_path / "wrong.csv"
        pd.DataFrame({"when": ["2020-01-01"], "value": [1.0]}).to_csv(path, index=False)

        with pytest.raises(DataSourceError, match="missing required column"):
            CSVDataLoader(path, DATETIME_COL, TARGET_COL).load()

    def test_does_not_sort_or_deduplicate(self, tmp_path: Path):
        path = tmp_path / "unsorted.csv"
        pd.DataFrame(
            {DATETIME_COL: ["2020-01-02 00:00", "2020-01-01 00:00"], TARGET_COL: [2.0, 1.0]}
        ).to_csv(path, index=False)

        frame = CSVDataLoader(path, DATETIME_COL, TARGET_COL).load()
        assert not frame[DATETIME_COL].is_monotonic_increasing


class TestTimeSeriesCleaner:
    def test_sorts_an_unsorted_frame(self):
        raw = pd.DataFrame(
            {
                DATETIME_COL: pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"]),
                TARGET_COL: [3.0, 1.0, 2.0],
            }
        )
        cleaned = TimeSeriesCleaner(DATETIME_COL, TARGET_COL).clean(raw)

        assert cleaned.index.is_monotonic_increasing
        assert list(cleaned[TARGET_COL]) == [1.0, 2.0, 3.0]

    def test_duplicate_timestamps_are_averaged_not_dropped(self):
        # The real DST fall-back rows: 2014-11-02 02:00 carries both readings.
        raw = pd.DataFrame(
            {
                DATETIME_COL: pd.to_datetime(
                    ["2014-11-02 02:00", "2014-11-02 02:00", "2014-11-02 03:00"]
                ),
                TARGET_COL: [22935.0, 23755.0, 20000.0],
            }
        )
        cleaned = TimeSeriesCleaner(DATETIME_COL, TARGET_COL).clean(raw)

        assert len(cleaned) == 2
        assert cleaned.loc["2014-11-02 02:00", TARGET_COL] == pytest.approx(23345.0)

    def test_gaps_are_preserved_not_filled(self):
        raw = pd.DataFrame(
            {
                DATETIME_COL: pd.to_datetime(["2020-01-01 00:00", "2020-01-01 03:00"]),
                TARGET_COL: [1.0, 4.0],
            }
        )
        cleaned = TimeSeriesCleaner(DATETIME_COL, TARGET_COL).clean(raw)

        assert len(cleaned) == 2

    def test_returns_a_datetime_index(self):
        raw = _hourly_frame(10).reset_index(names=DATETIME_COL)
        cleaned = TimeSeriesCleaner(DATETIME_COL, TARGET_COL).clean(raw)

        assert isinstance(cleaned.index, pd.DatetimeIndex)
        assert list(cleaned.columns) == [TARGET_COL]


def _validator(
    min_rows: int = 10,
    allow_missing_target_ratio: float = 0.01,
    allow_missing_index_ratio: float = 0.01,
) -> SchemaValidator:
    return SchemaValidator(
        target_column=TARGET_COL,
        min_rows=min_rows,
        allow_missing_target_ratio=allow_missing_target_ratio,
        allow_missing_index_ratio=allow_missing_index_ratio,
        expected_frequency="h",
    )


class TestSchemaValidator:
    def test_clean_data_passes_and_reports_measurements(self):
        report = _validator().validate(_hourly_frame(100))

        assert report.is_valid
        assert report.rows == 100
        assert report.missing_target_ratio == 0.0
        assert report.missing_index_ratio == 0.0

    def test_too_few_rows_is_rejected(self):
        with pytest.raises(DataValidationError, match="at least 500 rows"):
            _validator(min_rows=500).validate(_hourly_frame(100))

    def test_excess_missing_targets_rejected(self):
        frame = _hourly_frame(100)
        frame.iloc[:5] = None  # 5% > the 1% allowance

        with pytest.raises(DataValidationError, match="missing target values"):
            _validator().validate(frame)

    def test_excess_index_gaps_rejected(self):
        frame = _hourly_frame(100).drop(index=_hourly_frame(100).index[10:30])

        with pytest.raises(DataValidationError, match="absent from the index"):
            _validator().validate(frame)

    def test_small_gap_within_allowance_passes(self):
        # The real PJME file is missing 0.02% of its hours; that must not fail.
        frame = _hourly_frame(1000).drop(index=_hourly_frame(1000).index[500:501])
        assert _validator().validate(frame).is_valid

    def test_the_two_allowances_move_independently(self):
        # The whole point of splitting the key: tolerate the source omitting
        # hours without also tolerating NaN in the hours it did ship.
        frame = _hourly_frame(1000).drop(index=_hourly_frame(1000).index[100:150])
        frame.iloc[:5] = None

        report = _validator(
            allow_missing_target_ratio=0.0, allow_missing_index_ratio=0.10
        ).validate(frame, raise_on_error=False)

        assert [p for p in report.problems if "missing target values" in p]
        assert not [p for p in report.problems if "absent from the index" in p]

    def test_unsorted_index_is_rejected(self):
        frame = _hourly_frame(100).iloc[::-1]

        with pytest.raises(DataValidationError, match="not sorted"):
            _validator().validate(frame)

    def test_remaining_duplicate_timestamps_are_reported(self):
        # The cleaner removes these, so this is the safety net for a caller
        # that validates a frame the cleaner never saw.
        index = pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-01 00:00", "2020-01-01 01:00"])
        frame = pd.DataFrame({TARGET_COL: [1.0, 2.0, 3.0]}, index=index)

        with pytest.raises(DataValidationError, match="duplicate timestamp"):
            _validator(min_rows=1).validate(frame)

    def test_all_problems_reported_in_one_run(self):
        frame = _hourly_frame(5).iloc[::-1]
        report = _validator(min_rows=500).validate(frame, raise_on_error=False)

        assert not report.is_valid
        assert len(report.problems) >= 2

    def test_non_datetime_index_is_rejected(self):
        with pytest.raises(DataValidationError, match="DatetimeIndex"):
            _validator().validate(pd.DataFrame({TARGET_COL: [1.0] * 20}))

    def test_missing_target_column_is_rejected(self):
        frame = _hourly_frame(20).rename(columns={TARGET_COL: "other"})
        with pytest.raises(DataValidationError, match="not found"):
            _validator().validate(frame)


class TestChronologicalSplitter:
    def test_splits_in_time_order_without_overlap(self):
        splits = ChronologicalSplitter(0.7, 0.15, 0.15).split(_hourly_frame(1000))

        assert len(splits.train) == 700
        assert len(splits.validation) == 150
        assert len(splits.test) == 150
        assert len(splits) == 1000

    def test_no_future_data_leaks_into_train(self):
        splits = ChronologicalSplitter(0.7, 0.15, 0.15).split(_hourly_frame(1000))

        assert splits.train.index.max() < splits.validation.index.min()
        assert splits.validation.index.max() < splits.test.index.min()

    def test_no_rows_lost_to_rounding(self):
        # 997 does not divide evenly; test takes the remainder.
        splits = ChronologicalSplitter(0.7, 0.15, 0.15).split(_hourly_frame(997))
        assert len(splits) == 997

    def test_refuses_an_unsorted_index(self):
        frame = _hourly_frame(1000).iloc[::-1]

        with pytest.raises(SplitError, match="Refusing to split an unsorted index"):
            ChronologicalSplitter(0.7, 0.15, 0.15).split(frame)

    def test_refuses_a_non_datetime_index(self):
        with pytest.raises(SplitError, match="DatetimeIndex"):
            ChronologicalSplitter(0.7, 0.15, 0.15).split(pd.DataFrame({TARGET_COL: [1.0] * 100}))

    def test_too_few_rows_to_split_is_rejected(self):
        with pytest.raises(SplitError, match="cannot be split"):
            ChronologicalSplitter(0.7, 0.15, 0.15).split(_hourly_frame(3))


class TestFromSettings:
    """`from_settings()` is the path production uses; the explicit constructors
    are only used by tests. These run without the data file present."""

    def test_loader_wires_source_config(self):
        settings = get_settings()
        loader = CSVDataLoader.from_settings()

        assert loader.datetime_column == settings.data.source.datetime_column
        assert loader.target_column == settings.data.source.target_column
        assert loader.path == Path(settings.base.paths.raw_data_dir) / settings.data.source.filename

    def test_cleaner_wires_source_config(self):
        settings = get_settings()
        cleaner = TimeSeriesCleaner.from_settings()

        assert cleaner.datetime_column == settings.data.source.datetime_column
        assert cleaner.target_column == settings.data.source.target_column

    def test_validator_wires_validation_config(self):
        settings = get_settings()
        validator = SchemaValidator.from_settings()

        assert validator.target_column == settings.data.source.target_column
        assert validator.min_rows == settings.data.validation.min_rows
        assert (
            validator.allow_missing_target_ratio
            == settings.data.validation.allow_missing_target_ratio
        )
        assert (
            validator.allow_missing_index_ratio
            == settings.data.validation.allow_missing_index_ratio
        )
        # Literal, unlike the rest: guards against "H" returning to the config.
        assert validator.expected_frequency == "h"

    def test_splitter_wires_split_config(self):
        settings = get_settings()
        splitter = ChronologicalSplitter.from_settings()

        assert splitter.train_ratio == settings.data.split.train_ratio
        assert splitter.validation_ratio == settings.data.split.validation_ratio
        assert splitter.test_ratio == settings.data.split.test_ratio


@pytest.mark.skipif(not REAL_DATA.exists(), reason="PJME data not downloaded (gitignored)")
class TestRealPJMEData:
    """End-to-end run against the actual file, whose quirks drove this design."""

    def test_full_pipeline_load_clean_validate_split(self):
        raw = CSVDataLoader(REAL_DATA, DATETIME_COL, TARGET_COL).load()
        cleaned = TimeSeriesCleaner(DATETIME_COL, TARGET_COL).clean(raw)
        report = SchemaValidator.from_settings().validate(cleaned)
        splits = ChronologicalSplitter.from_settings().split(cleaned)

        # 145,366 raw rows contain 4 DST-duplicated hours -> 145,362 unique.
        assert len(raw) == 145_366
        assert len(cleaned) == 145_362
        assert report.is_valid
        assert splits.train.index.max() < splits.test.index.min()

    def test_raw_file_really_is_unsorted(self):
        # If the source ever ships sorted, the splitter's guard goes untested.
        raw = CSVDataLoader(REAL_DATA, DATETIME_COL, TARGET_COL).load()
        assert not raw[DATETIME_COL].is_monotonic_increasing

    def test_dst_duplicates_are_averaged_away(self):
        raw = CSVDataLoader(REAL_DATA, DATETIME_COL, TARGET_COL).load()
        cleaned = TimeSeriesCleaner(DATETIME_COL, TARGET_COL).clean(raw)

        assert raw[DATETIME_COL].duplicated().sum() == 4
        assert cleaned.index.duplicated().sum() == 0
