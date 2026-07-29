"""The FastAPI forecast service.

The champion named in config is loaded once at startup; a challenger asked for
by name is loaded lazily and cached, so no request pays the registry round-trip
twice. The model itself is loaded from the MLflow Registry, never refitted here.

Every served forecast is persisted to the prediction store; `/actuals` feeds
measured hours back in and `/metrics` rolls per-horizon error over the anchors
that have since been reconciled.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import mlflow
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from energycast.config import Settings, get_settings
from energycast.monitoring import (
    MonitoringError,
    PredictionStore,
    SQLitePredictionStore,
    rolling_report,
)
from energycast.monitoring.schemas import ActualsRequest, ActualsResponse, MetricsResponse
from energycast.serving.forecaster import Forecaster, ForecastError
from energycast.serving.schemas import ForecastRequest, ForecastResponse
from energycast.training import RegistryError
from energycast.utils import get_logger

logger = get_logger(__name__)


class ForecasterCache:
    """Holds one Forecaster per roster model, loaded on first use."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.champion = settings.base.serving.champion_model
        self._forecasters: dict[str, Forecaster] = {}

    def get(self, name: str) -> Forecaster:
        if name not in self._forecasters:
            self._forecasters[name] = Forecaster.from_registered(name, self.settings)
        return self._forecasters[name]

    @property
    def loaded(self) -> list[str]:
        return sorted(self._forecasters)


def _cache(request: Request) -> ForecasterCache:
    return request.app.state.cache


def _store(request: Request) -> PredictionStore:
    return request.app.state.store


Cache = Annotated[ForecasterCache, Depends(_cache)]
Store = Annotated[PredictionStore, Depends(_store)]


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the service; the champion is warmed in the startup lifespan."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        mlflow.set_tracking_uri(settings.base.mlflow.tracking_uri)
        cache = ForecasterCache(settings)
        cache.get(cache.champion)
        app.state.cache = cache
        app.state.store = SQLitePredictionStore(settings.base.monitoring.database_path)
        logger.info(
            "serving ready",
            extra={"event": "serving_ready", "champion": cache.champion},
        )
        yield
        app.state.store.close()

    app = FastAPI(title="EnergyCast forecast API", lifespan=lifespan)

    @app.exception_handler(RegistryError)
    async def _model_not_found(request: Request, error: RegistryError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(error)})

    @app.exception_handler(ForecastError)
    async def _unservable_history(request: Request, error: ForecastError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.exception_handler(MonitoringError)
    async def _no_metrics_yet(request: Request, error: MonitoringError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(error)})

    @app.get("/health")
    def health(cache: Cache) -> dict:
        champion = cache.get(cache.champion)
        return {
            "status": "ok",
            "champion": champion.name,
            "run_id": champion.loaded.run_id,
            "loaded": cache.loaded,
        }

    @app.post("/predict", response_model=ForecastResponse)
    def predict(body: ForecastRequest, cache: Cache, store: Store) -> ForecastResponse:
        forecaster = cache.get(body.model or cache.champion)
        forecast = forecaster.predict(body.to_frame(forecaster.target))
        store.record_forecast(forecast)
        return ForecastResponse.from_forecast(forecast)

    @app.post("/actuals", response_model=ActualsResponse)
    def actuals(body: ActualsRequest, store: Store) -> ActualsResponse:
        reconciled = store.record_actuals(body.to_mapping())
        return ActualsResponse(reconciled=reconciled)

    @app.get("/metrics", response_model=MetricsResponse)
    def metrics(cache: Cache, store: Store, model: str | None = None) -> MetricsResponse:
        report = rolling_report(
            store,
            model or cache.champion,
            settings.model.sequence.prediction_horizon,
            settings.base.monitoring.rolling_window_days,
        )
        return MetricsResponse.from_report(report)

    return app
