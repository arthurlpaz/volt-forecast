"""Retraining orchestration: a configurable OR of signals, not a fixed cron.

    collect signals -> evaluate_triggers -> retrain challenger -> compare -> promote

A retrain fires when drift, rolling error, or the count of new observations trips
its threshold. The champion architecture is retrained and promoted to the
`champion` alias only if it beats the incumbent on the same recent rows.
"""

from energycast.retraining.orchestrator import (
    RetrainingError,
    RetrainingOrchestrator,
    RetrainingOutcome,
    main,
)
from energycast.retraining.promotion import beats_champion
from energycast.retraining.store import RetrainingRun, RetrainingStore, SQLiteRetrainingStore
from energycast.retraining.triggers import (
    RetrainDecision,
    TriggerSignals,
    evaluate_triggers,
)

__all__ = [
    "RetrainDecision",
    "RetrainingError",
    "RetrainingOrchestrator",
    "RetrainingOutcome",
    "RetrainingRun",
    "RetrainingStore",
    "SQLiteRetrainingStore",
    "TriggerSignals",
    "beats_champion",
    "evaluate_triggers",
    "main",
]
