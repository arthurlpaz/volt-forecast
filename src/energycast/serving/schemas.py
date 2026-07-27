"""Request and response bodies for the forecast API."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from pydantic import BaseModel, Field

from energycast.serving.forecaster import Forecast


class Observation(BaseModel):
    """One measured hour of the target series."""

    timestamp: datetime
    value: float


class ForecastRequest(BaseModel):
    """A raw MW history and, optionally, which roster model to serve it with."""

    history: list[Observation] = Field(min_length=1)
    model: str | None = None

    def to_frame(self, target_column: str) -> pd.DataFrame:
        index = pd.DatetimeIndex([obs.timestamp for obs in self.history])
        return pd.DataFrame({target_column: [obs.value for obs in self.history]}, index=index)


class HorizonPoint(BaseModel):
    """One forecast hour."""

    timestamp: datetime
    prediction_mw: float


class ForecastResponse(BaseModel):
    """A model's 24-hour forecast and its identity."""

    model: str
    kind: str
    run_id: str
    forecast: list[HorizonPoint]

    @classmethod
    def from_forecast(cls, forecast: Forecast) -> ForecastResponse:
        points = [
            HorizonPoint(timestamp=ts, prediction_mw=float(value))
            for ts, value in zip(forecast.timestamps, forecast.values, strict=True)
        ]
        return cls(
            model=forecast.model,
            kind=forecast.kind,
            run_id=forecast.run_id,
            forecast=points,
        )
