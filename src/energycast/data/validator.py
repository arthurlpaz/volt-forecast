"""Schema and quality validation against configs/data.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from energycast.utils import get_logger

logger = get_logger(__name__)


class DataValidationError(ValueError):
    """Raised when data violates the contract declared in configs/data.yaml."""


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of a validation run, returned even when valid so callers can
    log the measured ratios rather than only knowing nothing blew up."""

    rows: int
    missing_target_ratio: float
    missing_index_ratio: float
    problems: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.problems


class SchemaValidator:
    """Validates a cleaned hourly series.

    Reports; never mutates. Cleaning belongs to `TimeSeriesCleaner`, so that a
    failure describes the data you have rather than what a repair step left.

    Missing targets and missing hours have separate allowances: a NaN reading
    and an hour the source never shipped are different failures, and one
    tolerance could not be relaxed for either without relaxing both.
    """

    def __init__(
        self,
        target_column: str,
        min_rows: int,
        allow_missing_target_ratio: float,
        allow_missing_index_ratio: float,
        expected_frequency: str,
    ) -> None:
        self.target_column = target_column
        self.min_rows = min_rows
        self.allow_missing_target_ratio = allow_missing_target_ratio
        self.allow_missing_index_ratio = allow_missing_index_ratio
        self.expected_frequency = expected_frequency

    @classmethod
    def from_settings(cls) -> SchemaValidator:
        from energycast.config import get_settings

        settings = get_settings()
        validation = settings.data.validation
        return cls(
            target_column=settings.data.source.target_column,
            min_rows=validation.min_rows,
            allow_missing_target_ratio=validation.allow_missing_target_ratio,
            allow_missing_index_ratio=validation.allow_missing_index_ratio,
            expected_frequency=validation.expected_frequency,
        )

    def validate(self, frame: pd.DataFrame, *, raise_on_error: bool = True) -> ValidationReport:
        """Check every rule before raising, so one run reports every problem."""
        if self.target_column not in frame.columns:
            raise DataValidationError(
                f"Target column {self.target_column!r} not found. Columns: {list(frame.columns)}"
            )
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise DataValidationError(
                f"Expected a DatetimeIndex, got {type(frame.index).__name__}. "
                "Run TimeSeriesCleaner before validating."
            )

        problems: list[str] = []
        rows = len(frame)

        if rows < self.min_rows:
            problems.append(f"expected at least {self.min_rows} rows, got {rows}")

        missing_target_ratio = float(frame[self.target_column].isna().mean()) if rows else 0.0
        if missing_target_ratio > self.allow_missing_target_ratio:
            problems.append(
                f"missing target values {missing_target_ratio:.4%} exceed the allowed "
                f"{self.allow_missing_target_ratio:.4%}"
            )

        if not frame.index.is_monotonic_increasing:
            problems.append("index is not sorted ascending — a positional split would leak")

        duplicates = int(frame.index.duplicated().sum())
        if duplicates:
            problems.append(f"{duplicates} duplicate timestamp(s) remain in the index")

        missing_index_ratio = 0.0
        # A gap count is only meaningful once the index is sorted and unique.
        if rows > 1 and not duplicates and frame.index.is_monotonic_increasing:
            expected_index = pd.date_range(
                frame.index.min(), frame.index.max(), freq=self.expected_frequency
            )
            gaps = len(expected_index.difference(frame.index))
            missing_index_ratio = gaps / len(expected_index)
            if missing_index_ratio > self.allow_missing_index_ratio:
                problems.append(
                    f"{gaps} hour(s) absent from the index ({missing_index_ratio:.4%}) exceed "
                    f"the allowed {self.allow_missing_index_ratio:.4%}"
                )

        report = ValidationReport(
            rows=rows,
            missing_target_ratio=missing_target_ratio,
            missing_index_ratio=missing_index_ratio,
            problems=problems,
        )

        logger.info(
            "validated data",
            extra={
                "event": "data_validated",
                "rows": rows,
                "missing_target_ratio": missing_target_ratio,
                "missing_index_ratio": missing_index_ratio,
                "problems": problems,
            },
        )

        if problems and raise_on_error:
            raise DataValidationError(
                f"{self.target_column}: data failed validation — " + "; ".join(problems)
            )
        return report
