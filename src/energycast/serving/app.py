"""The FastAPI forecast service.

The champion named in config is loaded once at startup; a challenger asked for
by name is loaded lazily and cached, so no request pays the registry round-trip
twice. The model itself is loaded from the MLflow Registry, never refitted here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import mlflow
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from energycast.config import Settings, get_settings
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


Cache = Annotated[ForecasterCache, Depends(_cache)]


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the service; the champion is warmed in the startup lifespan."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        mlflow.set_tracking_uri(settings.base.mlflow.tracking_uri)
        cache = ForecasterCache(settings)
        cache.get(cache.champion)
        app.state.cache = cache
        logger.info(
            "serving ready",
            extra={"event": "serving_ready", "champion": cache.champion},
        )
        yield

    app = FastAPI(title="EnergyCast forecast API", lifespan=lifespan)

    @app.exception_handler(RegistryError)
    async def _model_not_found(request: Request, error: RegistryError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(error)})

    @app.exception_handler(ForecastError)
    async def _unservable_history(request: Request, error: ForecastError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(error)})

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
    def predict(body: ForecastRequest, cache: Cache) -> ForecastResponse:
        forecaster = cache.get(body.model or cache.champion)
        forecast = forecaster.predict(body.to_frame(forecaster.target))
        return ForecastResponse.from_forecast(forecast)

    return app
