"""Rolling accuracy of a served model, per horizon, in MW.

Reads the reconciled window from the store, keeps only the anchors whose every
horizon has a measured actual, and reuses the offline evaluation metrics to
score the predicted-versus-measured block the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from energycast.evaluation.metrics import mae_per_horizon, mape_per_horizon, rmse_per_horizon
from energycast.monitoring.store import PredictionStore


class MonitoringError(ValueError):
    """Raised when no fully reconciled forecasts are available to score."""


@dataclass(frozen=True)
class RollingReport:
    """Per-horizon rolling error of one model over the recent reconciled window."""

    model: str
    window_days: int
    n_anchors: int
    rmse: np.ndarray
    mae: np.ndarray
    mape: np.ndarray


def rolling_report(
    store: PredictionStore, model: str, horizon: int, window_days: int
) -> RollingReport:
    window = store.reconciled_window(model, window_days)
    y_true, y_pred = _complete_block(window, horizon)
    if not len(y_true):
        raise MonitoringError(
            f"No fully reconciled forecasts for {model!r} within the last {window_days} days."
        )
    return RollingReport(
        model=model,
        window_days=window_days,
        n_anchors=len(y_true),
        rmse=rmse_per_horizon(y_true, y_pred),
        mae=mae_per_horizon(y_true, y_pred),
        mape=mape_per_horizon(y_true, y_pred),
    )


def _complete_block(window: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    columns = list(range(1, horizon + 1))
    if window.empty:
        empty = np.empty((0, horizon))
        return empty, empty
    predicted = window.pivot(index="anchor", columns="horizon", values="predicted_mw")
    actual = window.pivot(index="anchor", columns="horizon", values="actual_mw")
    predicted = predicted.reindex(columns=columns)
    actual = actual.reindex(columns=columns)
    complete = actual.notna().all(axis=1)
    return actual.loc[complete].to_numpy(), predicted.loc[complete].to_numpy()
