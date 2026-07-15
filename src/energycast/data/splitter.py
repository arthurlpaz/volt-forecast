"""Chronological train/validation/test splitting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from energycast.utils import get_logger

logger = get_logger(__name__)


class SplitError(ValueError):
    """Raised when a frame cannot be split as configured."""


@dataclass(frozen=True)
class DataSplits:
    """Three chronologically ordered, non-overlapping slices of one series."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def __len__(self) -> int:
        return len(self.train) + len(self.validation) + len(self.test)


class ChronologicalSplitter:
    """Splits a time-indexed frame in time order.

    There is deliberately no shuffle option: shuffling trains on future hours
    and evaluates on past ones, which scores well and is worthless.
    """

    def __init__(self, train_ratio: float, validation_ratio: float, test_ratio: float) -> None:
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio

    @classmethod
    def from_settings(cls) -> ChronologicalSplitter:
        from energycast.config import get_settings

        split = get_settings().data.split
        return cls(
            train_ratio=split.train_ratio,
            validation_ratio=split.validation_ratio,
            test_ratio=split.test_ratio,
        )

    def split(self, frame: pd.DataFrame) -> DataSplits:
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise SplitError(
                f"Expected a DatetimeIndex, got {type(frame.index).__name__}. "
                "Run TimeSeriesCleaner before splitting."
            )
        # Refused rather than sorted here: sorting would mask a skipped clean
        # step, and the caller's own frame would stay unsorted regardless.
        if not frame.index.is_monotonic_increasing:
            raise SplitError(
                "Refusing to split an unsorted index: a positional split would put future "
                "hours in train and past hours in test. Run TimeSeriesCleaner first."
            )

        rows = len(frame)
        n_train = int(rows * self.train_ratio)
        n_validation = int(rows * self.validation_ratio)

        if min(n_train, n_validation, rows - n_train - n_validation) < 1:
            raise SplitError(
                f"{rows} row(s) cannot be split {self.train_ratio}/{self.validation_ratio}/"
                f"{self.test_ratio} without leaving a split empty"
            )

        # Test takes the remainder so no rows are lost to integer truncation.
        splits = DataSplits(
            train=frame.iloc[:n_train],
            validation=frame.iloc[n_train : n_train + n_validation],
            test=frame.iloc[n_train + n_validation :],
        )

        logger.info(
            "split data chronologically",
            extra={
                "event": "data_split",
                "train_rows": len(splits.train),
                "validation_rows": len(splits.validation),
                "test_rows": len(splits.test),
                "train_end": str(splits.train.index.max()),
                "test_start": str(splits.test.index.min()),
            },
        )
        return splits
