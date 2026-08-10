"""Response bodies for the drift API."""

from __future__ import annotations

from pydantic import BaseModel

from energycast.drift.detector import DriftResult


class ColumnDrift(BaseModel):
    """One magnitude column's drift score under the configured test."""

    column: str
    score: float


class DriftResponse(BaseModel):
    """The dataset-level drift verdict for the recent window, plus per-column scores."""

    method: str
    n_columns: int
    n_drifted: int
    share: float
    threshold: float
    drifted: bool
    per_column: list[ColumnDrift]
    reference_rows: int
    current_rows: int

    @classmethod
    def from_result(cls, result: DriftResult) -> DriftResponse:
        per_column = [
            ColumnDrift(column=column, score=score)
            for column, score in sorted(result.per_column.items())
        ]
        return cls(
            method=result.method,
            n_columns=result.n_columns,
            n_drifted=result.n_drifted,
            share=result.share,
            threshold=result.threshold,
            drifted=result.drifted,
            per_column=per_column,
            reference_rows=result.reference_rows,
            current_rows=result.current_rows,
        )
