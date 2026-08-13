"""Persistence for retraining cycles: the audit trail and the volume baseline.

Every orchestrated cycle is logged, whether or not it retrained: what tripped
it, the champion before and after, and the data cutoff. The cutoff of the last
cycle that actually retrained is the baseline the next run counts new
observations against.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pandas as pd

if TYPE_CHECKING:
    from energycast.retraining.triggers import RetrainDecision

_RUN_COLUMNS = [
    "ran_at",
    "triggered",
    "reasons",
    "data_cutoff",
    "champion_before",
    "champion_after",
    "promoted",
]


@dataclass(frozen=True)
class RetrainingRun:
    """A persisted retraining cycle, as read back for the audit trail."""

    ran_at: str
    triggered: bool
    reasons: str
    data_cutoff: str
    champion_before: str
    champion_after: str
    promoted: bool


@runtime_checkable
class RetrainingStore(Protocol):
    """Records each cycle and reads back the last data cutoff that retrained."""

    def record_cycle(
        self,
        ran_at: datetime,
        decision: RetrainDecision,
        data_cutoff: pd.Timestamp,
        champion_before: str,
        champion_after: str,
        promoted: bool,
    ) -> None: ...

    def last_trained_cutoff(self) -> str | None: ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS retraining_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ran_at TEXT NOT NULL,
    triggered INTEGER NOT NULL,
    reasons TEXT NOT NULL,
    data_cutoff TEXT NOT NULL,
    champion_before TEXT NOT NULL,
    champion_after TEXT NOT NULL,
    promoted INTEGER NOT NULL
);
"""


class SQLiteRetrainingStore:
    """A `RetrainingStore` backed by one SQLite file, safe across threads."""

    def __init__(self, database_path: str) -> None:
        if database_path != ":memory:":
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._connection:
            self._connection.executescript(_SCHEMA)

    def record_cycle(
        self,
        ran_at: datetime,
        decision: RetrainDecision,
        data_cutoff: pd.Timestamp,
        champion_before: str,
        champion_after: str,
        promoted: bool,
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO retraining_runs
                    (ran_at, triggered, reasons, data_cutoff,
                     champion_before, champion_after, promoted)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ran_at.isoformat(),
                    int(decision.should_retrain),
                    "; ".join(decision.reasons),
                    pd.Timestamp(data_cutoff).isoformat(),
                    champion_before,
                    champion_after,
                    int(promoted),
                ),
            )

    def last_trained_cutoff(self) -> str | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT data_cutoff FROM retraining_runs WHERE triggered = 1 "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return row[0] if row is not None else None

    def latest(self) -> RetrainingRun | None:
        with self._lock:
            row = self._connection.execute(
                f"SELECT {', '.join(_RUN_COLUMNS)} FROM retraining_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        values = dict(zip(_RUN_COLUMNS, row, strict=True))
        values["triggered"] = bool(values["triggered"])
        values["promoted"] = bool(values["promoted"])
        return RetrainingRun(**values)

    def close(self) -> None:
        with self._lock:
            self._connection.close()
