"""Tabular baselines behind one interface."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

from energycast.utils import get_logger

logger = get_logger(__name__)


class SklearnBaseline:
    """Adapts an estimator with sklearn's fit/predict to the Model protocol.

    Each estimator reaches 24 outputs differently — natively for the linear
    model, the forest, and XGBoost 2.x; as 24 fitted models for LightGBM, which
    has no native multi-output. The factories below settle that per estimator so
    callers see one shape.
    """

    def __init__(self, name: str, estimator: Any) -> None:
        self.name = name
        self.estimator = estimator

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> SklearnBaseline:
        self.estimator.fit(X, y)

        logger.info(
            "fitted baseline",
            extra={
                "event": "baseline_fitted",
                "model": self.name,
                "rows": len(X),
                "n_features": X.shape[1],
                "prediction_horizon": y.shape[1],
            },
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict(X))


def linear_regression() -> SklearnBaseline:
    return SklearnBaseline("linear_regression", LinearRegression())


def random_forest(random_state: int, n_jobs: int, **hyperparameters: Any) -> SklearnBaseline:
    return SklearnBaseline(
        "random_forest",
        RandomForestRegressor(random_state=random_state, n_jobs=n_jobs, **hyperparameters),
    )


def xgboost(random_state: int, n_jobs: int, **hyperparameters: Any) -> SklearnBaseline:
    return SklearnBaseline(
        "xgboost",
        xgb.XGBRegressor(
            random_state=random_state,
            n_jobs=n_jobs,
            multi_strategy="one_output_per_tree",
            **hyperparameters,
        ),
    )


def lightgbm(random_state: int, n_jobs: int, **hyperparameters: Any) -> SklearnBaseline:
    return SklearnBaseline(
        "lightgbm",
        MultiOutputRegressor(
            lgb.LGBMRegressor(
                random_state=random_state, n_jobs=n_jobs, verbose=-1, **hyperparameters
            )
        ),
    )


def build_from_settings() -> dict[str, SklearnBaseline]:
    """Every configured baseline, keyed by name."""
    from energycast.config import get_settings

    baselines = get_settings().model.baselines
    shared = {"random_state": baselines.random_state, "n_jobs": baselines.n_jobs}

    return {
        "linear_regression": linear_regression(),
        "random_forest": random_forest(**shared, **baselines.random_forest.hyperparameters()),
        "xgboost": xgboost(**shared, **baselines.xgboost.hyperparameters()),
        "lightgbm": lightgbm(**shared, **baselines.lightgbm.hyperparameters()),
    }
