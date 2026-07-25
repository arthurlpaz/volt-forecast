"""Per-horizon error metrics on forecasts in original units.

Each function takes two `(n_rows, horizon)` arrays and returns one score per
horizon, never the average: the best model changes with the horizon, so the
mean over 24 hours hides which model to trust at each step.
"""

from __future__ import annotations

import numpy as np


class MetricError(ValueError):
    """Raised when the two arrays cannot be compared."""


def _aligned(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise MetricError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}.")
    if y_true.ndim != 2:
        raise MetricError(f"Expected 2-D (n_rows, horizon) arrays, got {y_true.ndim}-D.")
    return y_true, y_pred


def rmse_per_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true, y_pred = _aligned(y_true, y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def mae_per_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true, y_pred = _aligned(y_true, y_pred)
    return np.mean(np.abs(y_true - y_pred), axis=0)


def mape_per_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true, y_pred = _aligned(y_true, y_pred)
    if np.any(y_true == 0):
        raise MetricError("MAPE is undefined where the true value is zero.")
    return np.mean(np.abs((y_true - y_pred) / y_true), axis=0) * 100
