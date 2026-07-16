"""The bar every learned model has to clear."""

from __future__ import annotations

import numpy as np
import pandas as pd

from energycast.utils import get_logger

logger = get_logger(__name__)


class NaiveModelError(ValueError):
    """Raised when the naive model cannot answer for the hours it was asked."""


class SeasonalNaiveModel:
    """Predicts each hour with the same hour one season earlier.

    Holds the observed series rather than learned parameters. For a target at
    t+h it reads t+h-season_length, which with a 24-hour horizon and a 168-hour
    season sits between 144 and 167 hours before t — always in the past, so it
    can be read at prediction time without knowing the future.

    Exists because 2,167 MW means nothing on its own. This is what says whether
    a gradient-booster earned its complexity.
    """

    def __init__(
        self,
        series: pd.Series,
        prediction_horizon: int,
        season_length: int = 168,
        frequency: str = "h",
    ) -> None:
        if season_length <= prediction_horizon:
            raise NaiveModelError(
                f"season_length {season_length} must exceed the {prediction_horizon}-hour "
                "horizon, or the value it reads would lie in the future being predicted."
            )
        self.name = f"seasonal_naive_{season_length}h"
        self.series = series
        self.prediction_horizon = prediction_horizon
        self.season_length = season_length
        self.frequency = frequency

    @classmethod
    def from_settings(cls, series: pd.Series, season_length: int = 168) -> SeasonalNaiveModel:
        from energycast.config import get_settings

        settings = get_settings()
        return cls(
            series=series,
            prediction_horizon=settings.model.sequence.prediction_horizon,
            season_length=season_length,
            frequency=settings.data.validation.expected_frequency,
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> SeasonalNaiveModel:
        """No parameters to learn; kept so the Model protocol holds."""
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X.index, pd.DatetimeIndex):
            raise NaiveModelError(
                f"Expected a DatetimeIndex, got {type(X.index).__name__}: the season is read "
                "from the index, not from the columns."
            )

        season = pd.Timedelta(self.season_length, unit=self.frequency)
        steps = [
            self.series.reindex(X.index + pd.Timedelta(step, unit=self.frequency) - season)
            for step in range(1, self.prediction_horizon + 1)
        ]
        predictions = np.column_stack([step.to_numpy() for step in steps])

        unknown = int(np.isnan(predictions).any(axis=1).sum())
        if unknown:
            raise NaiveModelError(
                f"{unknown} row(s) have no observation one season back — the series given to "
                "this model does not reach far enough behind the hours it was asked about."
            )

        logger.info(
            "predicted with seasonal naive",
            extra={
                "event": "naive_predicted",
                "rows": len(X),
                "season_length": self.season_length,
                "prediction_horizon": self.prediction_horizon,
            },
        )
        return predictions
