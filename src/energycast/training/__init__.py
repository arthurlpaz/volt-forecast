"""Training pipeline and MLflow tracking.

    load -> clean -> split -> features -> scale -> fit -> track -> register

Formalises the hand-driven M5 benchmark into a reproducible run that persists
each fitted model beside the scaler it needs to be served again.
"""

from energycast.training.pipeline import (
    PreparedData,
    TrainingPipeline,
    load_splits,
    main,
    prepare_data,
)
from energycast.training.registry import (
    LoadedModel,
    ModelMeta,
    RegistryError,
    load_registered,
    log_and_register,
    lstm_hyperparameters,
)
from energycast.training.tracking import ExperimentTracker

__all__ = [
    "ExperimentTracker",
    "LoadedModel",
    "ModelMeta",
    "PreparedData",
    "RegistryError",
    "TrainingPipeline",
    "load_registered",
    "load_splits",
    "log_and_register",
    "lstm_hyperparameters",
    "main",
    "prepare_data",
]
