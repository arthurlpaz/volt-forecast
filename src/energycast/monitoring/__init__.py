"""Monitoring: persist served forecasts, reconcile actuals, roll error metrics.

    serve -> record_forecast -> (hours pass) -> record_actuals -> rolling_report

A forecast can only be scored once the hours it predicted are measured, so the
store keeps predictions until their actuals arrive and metrics read back only
the anchors that are fully reconciled.
"""

from energycast.monitoring.reporting import MonitoringError, RollingReport, rolling_report
from energycast.monitoring.store import PredictionStore, SQLitePredictionStore

__all__ = [
    "MonitoringError",
    "PredictionStore",
    "RollingReport",
    "SQLitePredictionStore",
    "rolling_report",
]
