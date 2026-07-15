"""Calendar features derived from the index."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from energycast.utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class _Cycle:
    """One calendar ordinal and the period over which it repeats."""

    column: str
    period: int
    first_value: int = 0


_CYCLES = (
    _Cycle("hour", period=24),
    _Cycle("dayofweek", period=7),
    _Cycle("month", period=12, first_value=1),
)


class CalendarFeatureBuilder:
    """Adds calendar columns read from the index.

    Reads no target value, so these columns cannot leak. Holidays are federal
    only: `USFederalHolidayCalendar` does not know Black Friday.
    """

    def __init__(self, include_holidays: bool = True) -> None:
        self.include_holidays = include_holidays

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError(
                f"Expected a DatetimeIndex, got {type(frame.index).__name__}. "
                "Run TimeSeriesCleaner before building features."
            )

        out = frame.copy()
        index = frame.index

        out["hour"] = index.hour
        out["dayofweek"] = index.dayofweek
        out["month"] = index.month

        for cycle in _CYCLES:
            turns = (out[cycle.column].to_numpy() - cycle.first_value) / cycle.period
            radians = 2 * np.pi * turns
            out[f"{cycle.column}_sin"] = np.sin(radians)
            out[f"{cycle.column}_cos"] = np.cos(radians)

        out["is_weekend"] = (index.dayofweek >= 5).astype(int)

        if self.include_holidays:
            out["is_holiday"] = self._holiday_flag(index)

        logger.info(
            "built calendar features",
            extra={
                "event": "calendar_features_built",
                "rows": len(out),
                "columns_added": len(out.columns) - len(frame.columns),
                "holiday_hours": int(out["is_holiday"].sum()) if self.include_holidays else 0,
            },
        )
        return out

    @staticmethod
    def _holiday_flag(index: pd.DatetimeIndex) -> np.ndarray:
        if index.empty:
            return np.zeros(0, dtype=int)
        holidays = USFederalHolidayCalendar().holidays(start=index.min(), end=index.max())
        return index.normalize().isin(holidays).astype(int)
