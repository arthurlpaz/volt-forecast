"""The configurable OR that decides whether to retrain, never a fixed cron.

Three signals feed the decision, each already produced by an earlier milestone:
the latest drift verdict (M10), the rolling forecast error (M9), and the count of
measured hours accumulated past the champion's data cutoff. A retrain fires when
any enabled condition trips. The decision is pure so it is tested on plain
signals, with no store or model in reach.
"""

from __future__ import annotations

from dataclasses import dataclass

from energycast.config.settings import RetrainingConfig


@dataclass(frozen=True)
class TriggerSignals:
    """The three inputs to the retrain decision, as read at one point in time."""

    drifted: bool | None
    rolling_rmse: float | None
    new_observations: int


@dataclass(frozen=True)
class RetrainDecision:
    """Whether to retrain, and which conditions tripped."""

    should_retrain: bool
    reasons: list[str]


def evaluate_triggers(signals: TriggerSignals, config: RetrainingConfig) -> RetrainDecision:
    """Trip a retrain if any enabled condition exceeds its threshold."""
    reasons: list[str] = []
    if config.drift_triggers and signals.drifted:
        reasons.append("drift")
    if signals.rolling_rmse is not None and signals.rolling_rmse > config.rmse_threshold:
        reasons.append(f"rmse {signals.rolling_rmse:.1f} > {config.rmse_threshold:.1f}")
    if signals.new_observations > config.new_observations_threshold:
        reasons.append(
            f"new_observations {signals.new_observations} > {config.new_observations_threshold}"
        )
    return RetrainDecision(should_retrain=bool(reasons), reasons=reasons)
