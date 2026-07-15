"""Typed, validated configuration.

Three YAML files (base / data / model) map onto three Pydantic models, so a
change to sequence length never touches the file holding the MLflow URI.

Any field can be overridden by an environment variable, `__` marking each level
of nesting:
    ENERGYCAST_BASE__LOGGING__LEVEL=DEBUG
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

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
    train_ratio: float = Field(gt=0.0, lt=1.0)
    validation_ratio: float = Field(gt=0.0, lt=1.0)
    test_ratio: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def ratios_must_sum_to_one(self) -> SplitConfig:
        """Reject splits that do not account for exactly 100% of the data.

        Summing under 1.0 silently discards rows; over 1.0 silently overlaps
        the splits and leaks training data into test. Both stay invisible
        until they surface as an implausibly good score much later.
        """
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        # Tolerance for float representation, e.g. 0.7 + 0.2 + 0.1 is
        # 0.9999999999999999, while still catching real typos.
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"train/validation/test ratios must sum to 1.0, got {total:.6g} "
                f"(train={self.train_ratio}, validation={self.validation_ratio}, "
                f"test={self.test_ratio})"
            )
        return self


class SequenceConfig(BaseModel):
    sequence_length: int = Field(gt=0)
    prediction_horizon: int = Field(gt=0)


class FeaturesConfig(BaseModel):
    lags: list[int] = Field(min_length=1)
    rolling_windows: list[int]
    # One value today. A Literal rather than a free string so that configuring
    # a scaler nobody implemented fails at load instead of being ignored.
    scaler: Literal["standard"]

    @model_validator(mode="after")
    def periods_must_be_positive(self) -> FeaturesConfig:
        if any(lag < 1 for lag in self.lags):
            raise ValueError(f"lags must be >= 1 hour, got {self.lags}")
        # A one-hour rolling window is the lag itself under a costlier name.
        if any(window < 2 for window in self.rolling_windows):
            raise ValueError(f"rolling_windows must be >= 2 hours, got {self.rolling_windows}")
        return self


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
    features: FeaturesConfig
    lstm: LSTMConfig
    baselines: BaselinesConfig


class Settings(BaseSettings):
    """Aggregated, validated settings for the whole platform."""

    model_config = SettingsConfigDict(
        env_prefix="ENERGYCAST_",
        env_nested_delimiter="__",
    )

    base: BaseAppConfig
    data: DataConfig
    model: ModelConfig

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Give environment variables priority over the YAML files.

        `get_settings` feeds YAML in as init kwargs, which pydantic-settings
        ranks above env by default, making every ENERGYCAST_* override a
        silent no-op. Sources are deep-merged, so overriding one leaf leaves
        its siblings intact.
        """
        return (env_settings, init_settings, dotenv_settings, file_secret_settings)


def _load_yaml(filename: str) -> dict:
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and validate settings once per process.

    Cached so every module shares one validated instance; tests call
    `get_settings.cache_clear()` to reload with different config.
    """
    raw = {
        "base": _load_yaml("base.yaml"),
        "data": _load_yaml("data.yaml"),
        "model": _load_yaml("model.yaml"),
    }
    return Settings(**raw)
