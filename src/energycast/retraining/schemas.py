"""Response body for the read-only retraining check route."""

from __future__ import annotations

from pydantic import BaseModel

from energycast.retraining.triggers import RetrainDecision, TriggerSignals


class RetrainCheckResponse(BaseModel):
    """The trigger decision and the signals it was read from, without retraining."""

    should_retrain: bool
    reasons: list[str]
    drifted: bool | None
    rolling_rmse: float | None
    new_observations: int

    @classmethod
    def from_check(cls, signals: TriggerSignals, decision: RetrainDecision) -> RetrainCheckResponse:
        return cls(
            should_retrain=decision.should_retrain,
            reasons=decision.reasons,
            drifted=signals.drifted,
            rolling_rmse=signals.rolling_rmse,
            new_observations=signals.new_observations,
        )
