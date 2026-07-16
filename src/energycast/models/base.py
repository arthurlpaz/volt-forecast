"""The contract every forecaster satisfies."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class Model(Protocol):
    """Forecasts `prediction_horizon` hours from the features of one hour.

    A Protocol, not an ABC, so the LSTM of milestone 5 satisfies it without
    inheriting from this module.

    `predict` returns (n_rows, prediction_horizon) — the same shape for every
    model here and for the LSTM, which is what lets milestone 7 score them
    against each other rather than against different questions.
    """

    name: str

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Model: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
