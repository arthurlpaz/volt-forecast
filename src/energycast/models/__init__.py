"""Forecasting models.

Every model here answers the same question — given the features of hour t,
what are hours t+1 .. t+24 — and returns (n_rows, prediction_horizon). The
LSTM of milestone 5 answers it too, which is what lets milestone 7 compare
them instead of comparing different questions.

`SeasonalNaiveModel` is the bar: a learned model that cannot beat last week's
same hour has not earned its complexity.
"""

from energycast.models.base import Model
from energycast.models.baselines import (
    SklearnBaseline,
    build_from_settings,
    lightgbm,
    linear_regression,
    random_forest,
    xgboost,
)
from energycast.models.lstm import LSTMError, LSTMForecaster
from energycast.models.naive import NaiveModelError, SeasonalNaiveModel

__all__ = [
    "Model",
    "LSTMError",
    "LSTMForecaster",
    "NaiveModelError",
    "SeasonalNaiveModel",
    "SklearnBaseline",
    "build_from_settings",
    "lightgbm",
    "linear_regression",
    "random_forest",
    "xgboost",
]
