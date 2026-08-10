"""Persistence for the dataset-level drift signal, one row per run.

Each drift run writes its share of drifted columns and the drift verdict here,
sibling to the monitoring store. The retraining trigger reads the latest run;
the per-column detail lives in the Evidently HTML snapshot, not this table.
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
    from energycast.drift.detector import DriftResult

_RUN_COLUMNS = [
    "run_at",
    "window_start",
    "window_end",
    "method",
    "n_columns",
    "n_drifted",
    "share",
    "threshold",
    "drifted",
    "reference_rows",
    "current_rows",
]


@dataclass(frozen=True)
class DriftRun:
    """A persisted drift run, as read back for the retraining trigger."""

    run_at: str
    window_start: str
    window_end: str
    method: str
    n_columns: int
    n_drifted: int
    share: float
    threshold: float
    drifted: bool
    reference_rows: int
    current_rows: int


@runtime_checkable
class DriftStore(Protocol):
    """Records each drift run and reads back the most recent one."""

    def record_run(
        self,
        result: DriftResult,
        run_at: datetime,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
    ) -> None: ...

    def latest(self) -> DriftRun | None: ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS drift_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT NOT NULL,
    window_start TEXT NOT NULL,
    window_end TEXT NOT NULL,
    method TEXT NOT NULL,
    n_columns INTEGER NOT NULL,
    n_drifted INTEGER NOT NULL,
    share REAL NOT NULL,
    threshold REAL NOT NULL,
    drifted INTEGER NOT NULL,
    reference_rows INTEGER NOT NULL,
    current_rows INTEGER NOT NULL
);
"""


class SQLiteDriftStore:
    """A `DriftStore` backed by one SQLite file, safe across request threads."""

    def __init__(self, database_path: str) -> None:
        if database_path != ":memory:":
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._connection:
            self._connection.executescript(_SCHEMA)

    def record_run(
        self,
        result: DriftResult,
        run_at: datetime,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO drift_runs
                    (run_at, window_start, window_end, method, n_columns, n_drifted,
                     share, threshold, drifted, reference_rows, current_rows)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_at.isoformat(),
                    pd.Timestamp(window_start).isoformat(),
                    pd.Timestamp(window_end).isoformat(),
                    result.method,
                    result.n_columns,
                    result.n_drifted,
                    result.share,
                    result.threshold,
                    int(result.drifted),
                    result.reference_rows,
                    result.current_rows,
                ),
            )

    def latest(self) -> DriftRun | None:
        with self._lock:
            row = self._connection.execute(
                f"SELECT {', '.join(_RUN_COLUMNS)} FROM drift_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        values = dict(zip(_RUN_COLUMNS, row, strict=True))
        values["drifted"] = bool(values["drifted"])
        return DriftRun(**values)

    def close(self) -> None:
        with self._lock:
            self._connection.close()
