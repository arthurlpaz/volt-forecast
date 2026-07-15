"""Gap-aware sliding windows for sequence models."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from energycast.utils import get_logger

logger = get_logger(__name__)


class SequenceError(ValueError):
    """Raised when a frame cannot be windowed as configured."""


@dataclass(frozen=True)
class SequenceDataset:
    """Windows drawn from one frame, each from an uninterrupted run of hours."""

    X: np.ndarray
    y: np.ndarray
    target_timestamps: pd.DatetimeIndex
    feature_names: list[str]

    def __len__(self) -> int:
        return len(self.X)


class SequenceBuilder:
    """Builds (sequence_length, n_features) -> (prediction_horizon,) windows.

    No window spans a gap in the index. Build per split, after splitting.
    """

    def __init__(
        self,
        sequence_length: int,
        prediction_horizon: int,
        target_column: str,
        frequency: str = "h",
        dtype: npt.DTypeLike = np.float32,
    ) -> None:
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.target_column = target_column
        self.frequency = frequency
        self.dtype = dtype

    @classmethod
    def from_settings(cls) -> SequenceBuilder:
        from energycast.config import get_settings

        settings = get_settings()
        sequence = settings.model.sequence
        return cls(
            sequence_length=sequence.sequence_length,
            prediction_horizon=sequence.prediction_horizon,
            target_column=settings.data.source.target_column,
            frequency=settings.data.validation.expected_frequency,
        )

    @property
    def window_length(self) -> int:
        return self.sequence_length + self.prediction_horizon

    def build(self, frame: pd.DataFrame) -> SequenceDataset:
        self._check(frame)

        feature_names = list(frame.columns)
        target_position = feature_names.index(self.target_column)
        window = self.window_length

        blocks_x: list[np.ndarray] = []
        blocks_y: list[np.ndarray] = []
        stamps: list[pd.DatetimeIndex] = []
        skipped = 0

        for segment in self._segments(frame):
            if len(segment) < window:
                skipped += 1
                continue

            by_window_by_feature_by_hour = sliding_window_view(
                segment.to_numpy(dtype=self.dtype), window, axis=0
            )
            lookback = by_window_by_feature_by_hour[:, :, : self.sequence_length]
            horizon = by_window_by_feature_by_hour[:, target_position, self.sequence_length :]

            blocks_x.append(lookback.transpose(0, 2, 1))
            blocks_y.append(horizon)
            stamps.append(
                segment.index[self.sequence_length : len(segment) - self.prediction_horizon + 1]
            )

        if not blocks_x:
            raise SequenceError(
                f"No gap-free stretch of {window} hours ({self.sequence_length} lookback + "
                f"{self.prediction_horizon} horizon) exists in {len(frame)} row(s)."
            )

        dataset = SequenceDataset(
            X=np.concatenate(blocks_x),
            y=np.concatenate(blocks_y),
            target_timestamps=pd.DatetimeIndex(np.concatenate(stamps)),
            feature_names=feature_names,
        )

        logger.info(
            "built sequences",
            extra={
                "event": "sequences_built",
                "windows": len(dataset),
                "segments_too_short": skipped,
                "sequence_length": self.sequence_length,
                "prediction_horizon": self.prediction_horizon,
                "n_features": dataset.X.shape[2],
            },
        )
        return dataset

    def _check(self, frame: pd.DataFrame) -> None:
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise SequenceError(
                f"Expected a DatetimeIndex, got {type(frame.index).__name__}. "
                "Run TimeSeriesCleaner before building sequences."
            )
        if self.target_column not in frame.columns:
            raise SequenceError(
                f"Target column {self.target_column!r} not found. Columns: {list(frame.columns)}"
            )
        if not frame.index.is_monotonic_increasing:
            raise SequenceError(
                "Refusing to window an unsorted index: every window would span arbitrary hours."
            )
        if frame.isna().to_numpy().any():
            columns = frame.columns[frame.isna().any()].tolist()
            raise SequenceError(
                f"Column(s) {columns} contain NaN. Drop those rows first — the gap they leave "
                "is treated as a segment boundary, which is the intended behaviour."
            )

    def _segments(self, frame: pd.DataFrame) -> Iterator[pd.DataFrame]:
        """Split the frame wherever consecutive rows are not one hour apart."""
        step = pd.Timedelta(1, unit=self.frequency)
        breaks = (frame.index.to_series().diff() != step).to_numpy()
        for _, segment in frame.groupby(breaks.cumsum()):
            yield segment
