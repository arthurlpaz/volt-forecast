"""Request and response bodies for the monitoring API."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field

from energycast.monitoring.reporting import RollingReport
from energycast.serving.schemas import Observation


class ActualsRequest(BaseModel):
    """Measured hours to reconcile against previously served forecasts."""

    actuals: list[Observation] = Field(min_length=1)

    def to_mapping(self) -> dict[pd.Timestamp, float]:
        return {pd.Timestamp(obs.timestamp): obs.value for obs in self.actuals}


class ActualsResponse(BaseModel):
    """How many stored forecast rows the actuals reconciled."""

    reconciled: int


class HorizonMetrics(BaseModel):
    """Rolling error at one forecast horizon, in MW (mape in percent)."""

    horizon: int
    rmse: float
    mae: float
    mape: float


class MetricsResponse(BaseModel):
    """A model's rolling per-horizon error over the recent reconciled window."""

    model: str
    window_days: int
    n_anchors: int
    per_horizon: list[HorizonMetrics]
    rmse_mean: float
    mae_mean: float
    mape_mean: float

    @classmethod
    def from_report(cls, report: RollingReport) -> MetricsResponse:
        per_horizon = [
            HorizonMetrics(
                horizon=h + 1,
                rmse=float(report.rmse[h]),
                mae=float(report.mae[h]),
                mape=float(report.mape[h]),
            )
            for h in range(len(report.rmse))
        ]
        return cls(
            model=report.model,
            window_days=report.window_days,
            n_anchors=report.n_anchors,
            per_horizon=per_horizon,
            rmse_mean=float(report.rmse.mean()),
            mae_mean=float(report.mae.mean()),
            mape_mean=float(report.mape.mean()),
        )
