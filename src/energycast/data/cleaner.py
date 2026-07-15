"""Turning raw rows into a well-formed hourly series."""

from __future__ import annotations

import pandas as pd

from energycast.utils import get_logger

logger = get_logger(__name__)


class TimeSeriesCleaner:
    """Sorts, deduplicates, and indexes a raw observations frame.

    Does not fill gaps: interpolated values would be indistinguishable from
    measured ones once drift detection reads this data.
    """

    def __init__(self, datetime_column: str, target_column: str) -> None:
        self.datetime_column = datetime_column
        self.target_column = target_column

    @classmethod
    def from_settings(cls) -> TimeSeriesCleaner:
        from energycast.config import get_settings

        source = get_settings().data.source
        return cls(datetime_column=source.datetime_column, target_column=source.target_column)

    def clean(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows_in = len(frame)

        duplicate_stamps = int(frame[self.datetime_column].duplicated().sum())
        if duplicate_stamps:
            # DST fall-back repeats an hour, and both readings are real
            # measurements of it; keep="first" would discard half an observation.
            frame = frame.groupby(self.datetime_column, as_index=False)[self.target_column].mean()

        cleaned = (
            frame.sort_values(self.datetime_column)
            .set_index(self.datetime_column)
            .loc[:, [self.target_column]]
        )

        logger.info(
            "cleaned time series",
            extra={
                "event": "data_cleaned",
                "rows_in": rows_in,
                "rows_out": len(cleaned),
                "duplicate_stamps_averaged": duplicate_stamps,
            },
        )
        return cleaned
