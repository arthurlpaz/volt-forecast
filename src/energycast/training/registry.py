"""Persist a fitted model with everything needed to run it again.

A trained model is not enough on its own: serving it needs the scaler that put
its inputs in z-units and takes its outputs back to MW, and the feature order it
was fitted on. Those travel beside the model as run artifacts; the model itself
goes through its native MLflow flavor so the Model Registry can version it.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from energycast.features import SeriesScaler
from energycast.models import LSTMForecaster, SklearnBaseline
from energycast.utils import get_logger

logger = get_logger(__name__)

_LSTM_KWARGS = (
    "hidden_size",
    "num_layers",
    "dropout",
    "bidirectional",
    "learning_rate",
    "batch_size",
    "max_epochs",
    "early_stopping_patience",
    "lr_scheduler_factor",
    "lr_scheduler_patience",
    "seed",
    "validation_fraction",
)


class RegistryError(RuntimeError):
    """Raised when a model cannot be persisted or brought back whole."""


@dataclass(frozen=True)
class ModelMeta:
    """What a fitted model needs beside its weights to be run again."""

    kind: str
    name: str
    feature_names: list[str]
    target_column: str
    prediction_horizon: int
    sequence_length: int | None
    hyperparameters: dict


@dataclass(frozen=True)
class LoadedModel:
    """A registered model brought back with the scaler it was fitted with."""

    model: LSTMForecaster | SklearnBaseline
    scaler: SeriesScaler
    meta: ModelMeta


def log_and_register(
    model: LSTMForecaster | SklearnBaseline,
    scaler: SeriesScaler,
    meta: ModelMeta,
    registered_name: str,
) -> str:
    """Log the model, scaler and meta to the active run, then register a version.

    Assumes an MLflow run is already open. Returns the registered version.
    """
    mlflow.log_dict(asdict(meta), "meta/meta.json")
    with tempfile.TemporaryDirectory() as tmp:
        scaler_path = Path(tmp) / "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(str(scaler_path), artifact_path="preprocessing")

    if isinstance(model, LSTMForecaster):
        if model.network is None:
            raise RegistryError("Cannot register an LSTM that has not been fitted.")
        mlflow.pytorch.log_model(
            model.network, artifact_path="model", registered_model_name=registered_name
        )
    else:
        mlflow.sklearn.log_model(
            model.estimator, artifact_path="model", registered_model_name=registered_name
        )

    version = _latest_version(MlflowClient(), registered_name)
    logger.info(
        "registered model",
        extra={"event": "model_registered", "registered_name": registered_name, "version": version},
    )
    return version


def load_registered(registered_name: str, version: str | None = None) -> LoadedModel:
    """Bring a registered model back whole: model, scaler and meta."""
    client = MlflowClient()
    if version is None:
        version = _latest_version(client, registered_name)
    run_id = client.get_model_version(registered_name, version).run_id

    meta = _load_meta(run_id)
    scaler = _load_scaler(run_id)
    flavor = mlflow.pytorch if meta.kind == "lstm" else mlflow.sklearn
    loaded = flavor.load_model(f"models:/{registered_name}/{version}")

    if meta.kind == "lstm":
        model: LSTMForecaster | SklearnBaseline = LSTMForecaster(**meta.hyperparameters)
        model.network = loaded
    else:
        model = SklearnBaseline(meta.name, loaded)

    return LoadedModel(model=model, scaler=scaler, meta=meta)


def _latest_version(client: MlflowClient, registered_name: str) -> str:
    versions = client.search_model_versions(f"name='{registered_name}'")
    if not versions:
        raise RegistryError(f"No registered version found for {registered_name!r}.")
    return str(max(int(v.version) for v in versions))


def lstm_hyperparameters(model: LSTMForecaster) -> dict:
    """The constructor kwargs that rebuild an LSTMForecaster around loaded weights."""
    return {key: getattr(model, key) for key in _LSTM_KWARGS}


def _load_meta(run_id: str) -> ModelMeta:
    path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="meta/meta.json")
    with open(path, encoding="utf-8") as f:
        return ModelMeta(**json.load(f))


def _load_scaler(run_id: str) -> SeriesScaler:
    path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="preprocessing/scaler.joblib"
    )
    return joblib.load(path)
