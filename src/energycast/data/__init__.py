"""Data ingestion, cleaning, validation, and splitting.

    load -> clean -> validate -> split

Four steps rather than one class: only the loader is source-specific, so
retraining on observations that never came from a CSV reuses the rest as-is.
"""

from energycast.data.cleaner import TimeSeriesCleaner
from energycast.data.loader import CSVDataLoader, DataLoader, DataSourceError
from energycast.data.splitter import ChronologicalSplitter, DataSplits, SplitError
from energycast.data.validator import DataValidationError, SchemaValidator, ValidationReport

__all__ = [
    "CSVDataLoader",
    "ChronologicalSplitter",
    "DataLoader",
    "DataSourceError",
    "DataSplits",
    "DataValidationError",
    "SchemaValidator",
    "SplitError",
    "TimeSeriesCleaner",
    "ValidationReport",
]
