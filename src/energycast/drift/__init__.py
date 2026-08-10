"""Drift detection: compare the recent measured distribution against train.

    load_splits -> build_features -> magnitude columns -> detect_drift -> record_run

The reference is the training distribution; the current window is the most recent
measured hours. Only the MW-magnitude columns are compared. Each run persists its
dataset-level signal for the retraining trigger and an HTML snapshot for humans.
"""

from energycast.drift.detector import DriftDetector, DriftError, DriftResult, detect_drift
from energycast.drift.store import DriftRun, DriftStore, SQLiteDriftStore

__all__ = [
    "DriftDetector",
    "DriftError",
    "DriftResult",
    "DriftRun",
    "DriftStore",
    "SQLiteDriftStore",
    "detect_drift",
]
