"""Typed, validated configuration for the EnergyCast AI platform.

Design rationale
-----------------
Configuration is split into three YAML files (base / data / model) that map
onto three corresponding Pydantic models. This mirrors how the platform's
subsystems are separated (infrastructure vs. data vs. modeling), so a change
to sequence length never touches the same file as a change to the MLflow
tracking URI.

Pydantic (not Hydra) is used as the validation layer so that:
  1. Config errors fail fast, at startup, with a readable message instead of
     surfacing as a cryptic KeyError three modules deep.
  2. The same validation library is reused by the FastAPI layer (milestone 8),
     avoiding a second config paradigm in the same codebase.
  3. Secrets / environment-specific values (e.g. MLFLOW tracking URI in
     production) can override YAML defaults via environment variables,
     without editing files that are checked into version control.

Env override example:
    ENERGYCAST_MLFLOW__TRACKING_URI=postgresql://... 
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


class PathsConfig(BaseModel):
    raw_data_dir: str
    processed_data_dir: str
    artifacts_dir: str


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_format: bool = True


class MLflowConfig(BaseModel):
    tracking_uri: str
    experiment_name: str


class SourceConfig(BaseModel):
    filename: str
    datetime_column: str
    target_column: str


class ValidationConfig(BaseModel):
    min_rows: int
    allow_missing_ratio: float = Field(ge=0.0, le=1.0)
    expected_frequency: str


class SplitConfig(BaseModel):
    train_ratio: float
    validation_ratio: float
    test_ratio: float


class SequenceConfig(BaseModel):
    sequence_length: int = Field(gt=0)
    prediction_horizon: int = Field(gt=0)


class LRSchedulerConfig(BaseModel):
    type: str
    factor: float
    patience: int


class LSTMConfig(BaseModel):
    hidden_size: int = Field(gt=0)
    num_layers: int = Field(gt=0)
    dropout: float = Field(ge=0.0, lt=1.0)
    bidirectional: bool = False
    learning_rate: float = Field(gt=0.0)
    batch_size: int = Field(gt=0)
    max_epochs: int = Field(gt=0)
    early_stopping_patience: int = Field(gt=0)
    use_mixed_precision: bool = False
    lr_scheduler: LRSchedulerConfig


class BaselineModelConfig(BaseModel):
    model_config = {"extra": "allow"}


class BaselinesConfig(BaseModel):
    random_forest: BaselineModelConfig
    xgboost: BaselineModelConfig
    lightgbm: BaselineModelConfig


class BaseAppConfig(BaseModel):
    environment: Literal["development", "staging", "production"]
    paths: PathsConfig
    logging: LoggingConfig
    mlflow: MLflowConfig


class DataConfig(BaseModel):
    source: SourceConfig
    validation: ValidationConfig
    split: SplitConfig


class ModelConfig(BaseModel):
    sequence: SequenceConfig
    lstm: LSTMConfig
    baselines: BaselinesConfig


class Settings(BaseSettings):
    """Aggregated, validated settings for the whole platform.

    Values are first loaded from YAML, then this class allows environment
    variables prefixed with ENERGYCAST_ to override any field, using `__`
    as the nested delimiter (e.g. ENERGYCAST_MLFLOW__TRACKING_URI).
    """

    model_config = SettingsConfigDict(
        env_prefix="ENERGYCAST_",
        env_nested_delimiter="__",
    )

    base: BaseAppConfig
    data: DataConfig
    model: ModelConfig


def _load_yaml(filename: str) -> dict:
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and validate settings once per process.

    Cached with lru_cache so that every module in the platform shares a
    single validated Settings instance instead of re-reading and
    re-parsing YAML on every call (and so tests can call
    `get_settings.cache_clear()` to reload with different config).
    """
    raw = {
        "base": _load_yaml("base.yaml"),
        "data": _load_yaml("data.yaml"),
        "model": _load_yaml("model.yaml"),
    }
    return Settings(**raw)
