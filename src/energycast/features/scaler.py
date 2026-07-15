"""Standardisation fitted on the training split only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from energycast.utils import get_logger

logger = get_logger(__name__)


class ScalerError(RuntimeError):
    """Raised when a scaler is used out of order or cannot fit a column."""


class SeriesScaler:
    """Standardises columns to zero mean and unit variance.

    Fit on train, then transform validation and test with those statistics.
    Metrics must be reported through `inverse_transform`: train sigma is
    ~6,500 MW, so an RMSE of 0.10 left in z-units reads as 0.10 but means 650.
    """

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns
        self._means: pd.Series | None = None
        self._stds: pd.Series | None = None

    @classmethod
    def from_settings(cls, columns: list[str] | None = None) -> SeriesScaler:
        from energycast.config import get_settings

        scaler = get_settings().model.features.scaler
        if scaler != "standard":
            raise ScalerError(f"Unsupported scaler {scaler!r}; only 'standard' is implemented.")
        return cls(columns=columns)

    @property
    def is_fitted(self) -> bool:
        return self._means is not None

    @property
    def feature_names(self) -> list[str]:
        self._require_fitted()
        return list(self._means.index)

    def fit(self, frame: pd.DataFrame) -> SeriesScaler:
        columns = self.columns if self.columns is not None else list(frame.columns)
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ScalerError(f"Cannot fit on absent column(s) {sorted(missing)}")

        subset = frame[columns]
        stds = subset.std()
        constant = stds[stds == 0].index.tolist()
        if constant:
            raise ScalerError(
                f"Column(s) {constant} are constant on this split; standardising them divides "
                "by zero. Drop them or exclude them from `columns`."
            )

        self._means = subset.mean()
        self._stds = stds

        logger.info(
            "fitted scaler",
            extra={
                "event": "scaler_fitted",
                "columns": columns,
                "rows": len(frame),
                "index_start": str(frame.index.min()),
                "index_end": str(frame.index.max()),
            },
        )
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        missing = set(self._means.index) - set(frame.columns)
        if missing:
            raise ScalerError(f"Cannot transform: column(s) {sorted(missing)} absent")

        out = frame.copy()
        columns = list(self._means.index)
        out[columns] = (out[columns] - self._means) / self._stds
        return out

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

    def inverse_transform(self, values: np.ndarray, column: str) -> np.ndarray:
        """Return `values` for one column in original units.

        Takes a bare array, not a frame: the LSTM predicts (n_windows, horizon).
        """
        self._require_fitted()
        if column not in self._means.index:
            raise ScalerError(f"Column {column!r} was not fitted. Fitted: {self.feature_names}")
        return np.asarray(values) * self._stds[column] + self._means[column]

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise ScalerError("Scaler is not fitted. Call fit() on the training split first.")
