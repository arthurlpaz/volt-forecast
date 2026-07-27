"""FastAPI serving for the registered forecast roster.

    raw MW history -> build_features -> scale -> predict -> inverse -> 24h MW

Serves the champion named in config, or a challenger asked for by name, loaded
from the MLflow Registry with the scaler it was fitted with.
"""

from energycast.serving.app import ForecasterCache, create_app
from energycast.serving.forecaster import Forecast, Forecaster, ForecastError
from energycast.serving.schemas import (
    ForecastRequest,
    ForecastResponse,
    HorizonPoint,
    Observation,
)

__all__ = [
    "Forecast",
    "ForecastError",
    "ForecastRequest",
    "ForecastResponse",
    "Forecaster",
    "ForecasterCache",
    "HorizonPoint",
    "Observation",
    "create_app",
]
