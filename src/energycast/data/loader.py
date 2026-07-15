"""Reading raw observations from a source."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from energycast.utils import get_logger

logger = get_logger(__name__)


class DataSourceError(RuntimeError):
    """Raised when a source cannot be read or lacks the expected columns."""


@runtime_checkable
class DataLoader(Protocol):
    """Produces a raw observations frame.

    A Protocol, not an ABC, so that non-CSV sources satisfy the contract
    without inheriting from this module.
    """

    def load(self) -> pd.DataFrame: ...


class CSVDataLoader:
    """Loads hourly observations from a CSV file.

    I/O only: does not sort, deduplicate, or validate. Sorting here would
    hide a skipped `TimeSeriesCleaner` step from every caller downstream.
    """

    def __init__(self, path: Path | str, datetime_column: str, target_column: str) -> None:
        self.path = Path(path)
        self.datetime_column = datetime_column
        self.target_column = target_column

    @classmethod
    def from_settings(cls) -> CSVDataLoader:
        from energycast.config import get_settings

        settings = get_settings()
        source = settings.data.source
        return cls(
            path=Path(settings.base.paths.raw_data_dir) / source.filename,
            datetime_column=source.datetime_column,
            target_column=source.target_column,
        )

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise DataSourceError(
                f"Data file not found: {self.path}. Download it first — see README.md."
            )

        frame = pd.read_csv(self.path)

        # Before parsing, so the error names the missing column rather than
        # surfacing as a KeyError from to_datetime.
        missing = {self.datetime_column, self.target_column} - set(frame.columns)
        if missing:
            raise DataSourceError(
                f"{self.path.name} is missing required column(s) {sorted(missing)}. "
                f"Found: {list(frame.columns)}. Check `source` in configs/data.yaml."
            )

        frame[self.datetime_column] = pd.to_datetime(frame[self.datetime_column])

        logger.info(
            "loaded raw data",
            extra={"event": "data_loaded", "path": str(self.path), "rows": len(frame)},
        )
        return frame
