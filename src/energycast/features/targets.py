"""Multi-horizon targets for tabular models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from energycast.utils import get_logger

logger = get_logger(__name__)


class TargetError(ValueError):
    """Raised when a frame cannot be paired with its horizon as configured."""


@dataclass(frozen=True)
class TabularDataset:
    """Features observed at one hour, paired with the hours that follow it."""

    X: pd.DataFrame
    y: np.ndarray
    target_timestamps: pd.DatetimeIndex
    feature_names: list[str]

    def __len__(self) -> int:
        return len(self.X)


class HorizonTargetBuilder:
    """Pairs the features at hour t with the target at t+1 .. t+horizon.

    The mirror of `LagFeatureBuilder`: lags look back, targets look forward,
    and both are measured in elapsed time. `shift(-24)` counts rows just as
    `shift(24)` does, so the horizon is taken on a complete hourly grid.

    Rows whose horizon crosses a gap or runs past the end of the frame are
    dropped: their future is unknown, not zero.
    """

    def __init__(
        self,
        target_column: str,
        prediction_horizon: int,
        frequency: str = "h",
    ) -> None:
        self.target_column = target_column
        self.prediction_horizon = prediction_horizon
        self.frequency = frequency

    @classmethod
    def from_settings(cls) -> HorizonTargetBuilder:
        from energycast.config import get_settings

        settings = get_settings()
        return cls(
            target_column=settings.data.source.target_column,
            prediction_horizon=settings.model.sequence.prediction_horizon,
            frequency=settings.data.validation.expected_frequency,
        )

    def build(self, frame: pd.DataFrame) -> TabularDataset:
        self._check(frame)

        grid = pd.date_range(frame.index.min(), frame.index.max(), freq=self.frequency)
        on_grid = frame[self.target_column].reindex(grid)

        horizons = range(1, self.prediction_horizon + 1)
        targets = pd.DataFrame(
            {step: on_grid.shift(-step).reindex(frame.index) for step in horizons},
            index=frame.index,
        )

        complete = targets.notna().all(axis=1)
        X = frame.loc[complete]
        y = targets.loc[complete].to_numpy()

        if not len(X):
            raise TargetError(
                f"No row in {len(frame)} has a gap-free {self.prediction_horizon}-hour future."
            )

        dataset = TabularDataset(
            X=X,
            y=y,
            target_timestamps=X.index + pd.Timedelta(1, unit=self.frequency),
            feature_names=list(X.columns),
        )

        logger.info(
            "built horizon targets",
            extra={
                "event": "targets_built",
                "rows_in": len(frame),
                "rows_out": len(dataset),
                "prediction_horizon": self.prediction_horizon,
                "n_features": len(dataset.feature_names),
            },
        )
        return dataset

    def _check(self, frame: pd.DataFrame) -> None:
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TargetError(
                f"Expected a DatetimeIndex, got {type(frame.index).__name__}. "
                "Run TimeSeriesCleaner before building targets."
            )
        if self.target_column not in frame.columns:
            raise TargetError(
                f"Target column {self.target_column!r} not found. Columns: {list(frame.columns)}"
            )
        if not frame.index.is_monotonic_increasing:
            raise TargetError(
                "Refusing to build targets on an unsorted index: every horizon would read "
                "arbitrary hours."
            )
        if frame.isna().to_numpy().any():
            columns = frame.columns[frame.isna().any()].tolist()
            raise TargetError(
                f"Column(s) {columns} contain NaN. Drop those rows first — the gap they leave "
                "is honoured when the horizon is taken."
            )
