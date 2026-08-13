"""The end-to-end forecast path a registered model is served behind.

Wraps a `LoadedModel` (model + scaler + meta) and turns a raw MW history into a
24-hour forecast in MW, rebuilding the exact features the model was fitted on:

    raw history -> build_features -> scale -> select meta.feature_names
        -> last row (baseline) | last window (LSTM) -> predict -> inverse

The most recent hour must carry a complete feature row, and the LSTM's lookback
must be an unbroken run of hours: a gap there is refused, never filled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from energycast.config import Settings, get_settings
from energycast.features import build_features
from energycast.training import LoadedModel, load_registered, registered_name
from energycast.utils import get_logger

logger = get_logger(__name__)


class ForecastError(ValueError):
    """Raised when a history cannot produce a forecast for its most recent hour."""


@dataclass(frozen=True)
class Forecast:
    """A model's 24-hour forecast in MW, and which model produced it."""

    model: str
    kind: str
    run_id: str
    anchor: pd.Timestamp
    timestamps: pd.DatetimeIndex
    values: np.ndarray


class Forecaster:
    """Serves one registered model: raw MW history in, 24-hour MW forecast out."""

    def __init__(self, loaded: LoadedModel, settings: Settings | None = None) -> None:
        self.loaded = loaded
        settings = settings or get_settings()
        self.target = settings.data.source.target_column
        self.horizon = settings.model.sequence.prediction_horizon
        self.freq = settings.data.validation.expected_frequency
        features = settings.model.features
        self._max_lookback = max([*features.lags, *features.rolling_windows])

    @classmethod
    def from_registered(
        cls, name: str, settings: Settings | None = None, alias: str | None = None
    ) -> Forecaster:
        settings = settings or get_settings()
        return cls(load_registered(registered_name(name), alias=alias), settings)

    @property
    def name(self) -> str:
        return self.loaded.meta.name

    @property
    def sequence_length(self) -> int | None:
        return self.loaded.meta.sequence_length

    @property
    def required_history(self) -> int:
        """Raw hours needed before the most recent hour has a complete input."""
        rows = self.sequence_length if self.sequence_length is not None else 1
        return self._max_lookback + rows

    def predict(self, history: pd.DataFrame) -> Forecast:
        self._check_history(history)
        features = build_features(history)
        if not len(features) or features.index[-1] != history.index[-1]:
            raise ForecastError(
                "The most recent hour has no complete feature row; its lag lookback is "
                f"incomplete. Provide at least {self.required_history} contiguous hours ending "
                "at the hour to forecast from."
            )

        scaled = self.loaded.scaler.transform(features)[self.loaded.meta.feature_names]
        anchor = scaled.index[-1]
        inputs = self._model_inputs(scaled)
        scaled_prediction = np.asarray(self.loaded.model.predict(inputs))
        values = self.loaded.scaler.inverse_transform(scaled_prediction, self.target).reshape(-1)

        step = pd.Timedelta(1, unit=self.freq)
        timestamps = pd.date_range(anchor + step, periods=self.horizon, freq=self.freq)
        logger.info(
            "served forecast",
            extra={
                "event": "forecast_served",
                "model": self.name,
                "anchor": str(anchor),
                "history_hours": len(history),
            },
        )
        return Forecast(
            model=self.name,
            kind=self.loaded.meta.kind,
            run_id=self.loaded.run_id,
            anchor=anchor,
            timestamps=timestamps,
            values=values,
        )

    def _model_inputs(self, scaled: pd.DataFrame) -> pd.DataFrame | np.ndarray:
        if self.sequence_length is None:
            return scaled.iloc[[-1]]
        window = scaled.iloc[-self.sequence_length :]
        self._require_contiguous(window.index)
        return window.to_numpy(dtype=np.float32)[np.newaxis, ...]

    def _require_contiguous(self, index: pd.DatetimeIndex) -> None:
        if len(index) < self.sequence_length:
            raise ForecastError(
                f"The LSTM needs {self.sequence_length} contiguous hours of lookback, but only "
                f"{len(index)} complete feature rows are available."
            )
        step = pd.Timedelta(1, unit=self.freq)
        if (index[-1] - index[0]) != step * (self.sequence_length - 1):
            raise ForecastError(
                "The LSTM lookback window has a gap; the last "
                f"{self.sequence_length} hours are not contiguous."
            )

    def _check_history(self, history: pd.DataFrame) -> None:
        if not isinstance(history.index, pd.DatetimeIndex):
            raise ForecastError(f"Expected a DatetimeIndex, got {type(history.index).__name__}.")
        if self.target not in history.columns:
            raise ForecastError(
                f"History has no {self.target!r} column. Columns: {list(history.columns)}"
            )
        if not history.index.is_monotonic_increasing:
            raise ForecastError("History index must be sorted in ascending time order.")
        if history.index.has_duplicates:
            raise ForecastError("History index has duplicate timestamps.")
        if len(history) < self.required_history:
            raise ForecastError(
                f"History has {len(history)} hours; the {self.name} model needs at least "
                f"{self.required_history}."
            )
