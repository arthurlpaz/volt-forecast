"""The one path from a raw target frame to model-ready features.

Training and serving both call this, so a forecast is built on the exact
columns the model was fitted on. Rows with incomplete lag history are dropped,
not filled: the leading hours of any frame have no 168-hour lookback.
"""

from __future__ import annotations

import pandas as pd

from energycast.features.calendar import CalendarFeatureBuilder
from energycast.features.lags import LagFeatureBuilder


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """calendar + lags, then drop the leading rows with incomplete history."""
    calendar = CalendarFeatureBuilder().build(frame)
    lagged = LagFeatureBuilder.from_settings().build(calendar)
    return lagged.dropna()
