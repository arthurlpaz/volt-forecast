"""Feature engineering and sequence datasets.

    calendar + lags  ->  scale (fit on train)  ->  sequences

Built per split, after `ChronologicalSplitter`, so that no scaler statistic and
no window lookback ever crosses from one split into another.
"""

from energycast.features.calendar import CalendarFeatureBuilder
from energycast.features.engineering import build_features
from energycast.features.lags import LagFeatureBuilder
from energycast.features.scaler import ScalerError, SeriesScaler
from energycast.features.sequences import SequenceBuilder, SequenceDataset, SequenceError
from energycast.features.targets import HorizonTargetBuilder, TabularDataset, TargetError

__all__ = [
    "CalendarFeatureBuilder",
    "HorizonTargetBuilder",
    "build_features",
    "LagFeatureBuilder",
    "ScalerError",
    "SequenceBuilder",
    "SequenceDataset",
    "SequenceError",
    "SeriesScaler",
    "TabularDataset",
    "TargetError",
]
