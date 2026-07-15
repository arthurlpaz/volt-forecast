"""Lag and rolling features measured in elapsed time, not row position."""

from __future__ import annotations

import pandas as pd

from energycast.utils import get_logger

logger = get_logger(__name__)


class LagFeatureBuilder:
    """Adds lag and trailing-rolling columns for the target.

    Statistics are measured in elapsed time, not row position: `shift(24)`
    counts rows, and returns a value 25 hours old around the 30 gaps in the
    PJME index. Incomplete history is left as NaN rather than filled.
    """

    def __init__(
        self,
        target_column: str,
        lags: list[int],
        rolling_windows: list[int],
        frequency: str = "h",
    ) -> None:
        self.target_column = target_column
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.frequency = frequency

    @classmethod
    def from_settings(cls) -> LagFeatureBuilder:
        from energycast.config import get_settings

        settings = get_settings()
        features = settings.model.features
        return cls(
            target_column=settings.data.source.target_column,
            lags=features.lags,
            rolling_windows=features.rolling_windows,
            frequency=settings.data.validation.expected_frequency,
        )

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError(
                f"Expected a DatetimeIndex, got {type(frame.index).__name__}. "
                "Run TimeSeriesCleaner before building features."
            )
        if self.target_column not in frame.columns:
            raise KeyError(
                f"Target column {self.target_column!r} not found. Columns: {list(frame.columns)}"
            )
        if not frame.index.is_monotonic_increasing:
            raise ValueError(
                "Refusing to lag an unsorted index: every lag would read an arbitrary hour. "
                "Run TimeSeriesCleaner first."
            )

        grid_index = pd.date_range(frame.index.min(), frame.index.max(), freq=self.frequency)
        target_on_grid = frame[self.target_column].reindex(grid_index)

        out = frame.copy()
        for lag in self.lags:
            out[f"lag_{lag}h"] = target_on_grid.shift(lag).reindex(frame.index)

        for window in self.rolling_windows:
            trailing = target_on_grid.shift(1).rolling(window, min_periods=window)
            out[f"rolling_mean_{window}h"] = trailing.mean().reindex(frame.index)
            out[f"rolling_std_{window}h"] = trailing.std().reindex(frame.index)

        added = [column for column in out.columns if column not in frame.columns]
        logger.info(
            "built lag features",
            extra={
                "event": "lag_features_built",
                "rows": len(out),
                "columns_added": len(added),
                "rows_with_incomplete_history": int(out[added].isna().any(axis=1).sum()),
            },
        )
        return out
