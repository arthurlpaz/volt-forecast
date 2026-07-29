"""Persistence for served forecasts and the actuals that later score them.

A forecast is logged the moment it is served: 24 rows, one per horizon, the
actual left null. When the hours it predicted are measured, the actuals are
matched in by timestamp. Rolling metrics read back the anchors whose every hour
has been reconciled.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pandas as pd

if TYPE_CHECKING:
    from energycast.serving.forecaster import Forecast

_WINDOW_COLUMNS = ["anchor", "horizon", "predicted_mw", "actual_mw"]


@runtime_checkable
class PredictionStore(Protocol):
    """Records served forecasts, reconciles actuals, reads back scored windows."""

    def record_forecast(self, forecast: Forecast) -> None: ...

    def record_actuals(self, actuals: Mapping[pd.Timestamp, float]) -> int: ...

    def reconciled_window(self, model: str, window_days: int) -> pd.DataFrame: ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    model TEXT NOT NULL,
    run_id TEXT NOT NULL,
    anchor TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    target_timestamp TEXT NOT NULL,
    predicted_mw REAL NOT NULL,
    actual_mw REAL,
    PRIMARY KEY (model, anchor, horizon)
);
CREATE INDEX IF NOT EXISTS idx_predictions_target ON predictions (target_timestamp);
"""


class SQLitePredictionStore:
    """A `PredictionStore` backed by one SQLite file, safe across request threads."""

    def __init__(self, database_path: str) -> None:
        if database_path != ":memory:":
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._connection:
            self._connection.executescript(_SCHEMA)

    def record_forecast(self, forecast: Forecast) -> None:
        rows = [
            (
                forecast.model,
                forecast.run_id,
                forecast.anchor.isoformat(),
                horizon,
                timestamp.isoformat(),
                float(value),
            )
            for horizon, (timestamp, value) in enumerate(
                zip(forecast.timestamps, forecast.values, strict=True), start=1
            )
        ]
        with self._lock, self._connection:
            self._connection.executemany(
                """
                INSERT INTO predictions
                    (model, run_id, anchor, horizon, target_timestamp, predicted_mw)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (model, anchor, horizon) DO UPDATE SET
                    run_id = excluded.run_id,
                    target_timestamp = excluded.target_timestamp,
                    predicted_mw = excluded.predicted_mw
                """,
                rows,
            )

    def record_actuals(self, actuals: Mapping[pd.Timestamp, float]) -> int:
        reconciled = 0
        with self._lock, self._connection:
            for timestamp, value in actuals.items():
                cursor = self._connection.execute(
                    "UPDATE predictions SET actual_mw = ? WHERE target_timestamp = ?",
                    (float(value), pd.Timestamp(timestamp).isoformat()),
                )
                reconciled += cursor.rowcount
        return reconciled

    def reconciled_window(self, model: str, window_days: int) -> pd.DataFrame:
        with self._lock:
            latest = self._connection.execute(
                "SELECT MAX(anchor) FROM predictions WHERE model = ?", (model,)
            ).fetchone()[0]
            if latest is None:
                return pd.DataFrame(columns=_WINDOW_COLUMNS)
            cutoff = (pd.Timestamp(latest) - pd.Timedelta(days=window_days)).isoformat()
            rows = self._connection.execute(
                """
                SELECT anchor, horizon, predicted_mw, actual_mw
                FROM predictions
                WHERE model = ? AND actual_mw IS NOT NULL AND anchor >= ?
                ORDER BY anchor, horizon
                """,
                (model, cutoff),
            ).fetchall()
        return pd.DataFrame(rows, columns=_WINDOW_COLUMNS)

    def close(self) -> None:
        with self._lock:
            self._connection.close()
