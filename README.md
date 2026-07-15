# EnergyCast AI

Continuous learning platform for energy consumption forecasting, built on the
[PJM Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
dataset.

This is not a notebook. The goal is a production-shaped ML system: clean
architecture, typed configuration, tests, reproducibility, and a real MLOps
pipeline — training → registry → monitoring → drift detection → automated
retraining → champion/challenger promotion → serving.

## Setup

Conda owns the interpreter; Poetry owns the dependencies. Conda pins Python
3.11 (numpy 1.26 ships no wheels for 3.13) and can supply CUDA runtimes if we
ever need them, while Poetry keeps the deterministic `poetry.lock` and the
dev/prod dependency split that the project depends on.

```bash
conda create -n energycast python=3.11 -y
conda activate energycast
poetry install
pre-commit install
```

> **Always activate the conda env before running Poetry.** This project sets
> `virtualenvs.create = false` in `poetry.toml` so that Poetry installs into
> the active conda env rather than building a second, redundant venv. The
> tradeoff is that `poetry install` *without* `conda activate energycast`
> first will install into whatever Python is on your PATH — likely the system
> one. Check with `which python` if you are unsure.

torch is installed as the **CPU-only** build (see the `pytorch-cpu` source in
`pyproject.toml`). The CUDA build pulls ~5 GB of `nvidia-*` wheels that the
LSTM in milestone 5 does not need. To switch, remove the `source` key from
`torch` and re-run `poetry lock`.

## Getting the data

```bash
kaggle datasets download -d robikscube/hourly-energy-consumption -p data/raw --unzip
```

The dataset is public and CC0, so this needs **no Kaggle API token**. It
unpacks 13 CSVs; `configs/data.yaml` names only `PJME_hourly.csv`. `data/raw/`
is gitignored — the data is never committed.

## Running the tests

```bash
conda activate energycast
pytest
```

Tests that exercise the real PJME file skip automatically when it is absent, so
the suite is green on a fresh clone before you download anything.

## Configuration

All configuration lives in `configs/` as YAML and is validated at startup by
`energycast.config.get_settings()`. Nothing is hardcoded.

| File | Owns |
|---|---|
| `base.yaml` | environment, paths, logging, MLflow |
| `data.yaml` | data source, validation rules, train/val/test split |
| `model.yaml` | sequence length, horizon, features, LSTM, baselines |

Any field can be overridden by an environment variable prefixed with
`ENERGYCAST_`, using `__` as the nesting delimiter. `BASE` picks the file,
and each `__` descends one level into it:

```bash
ENERGYCAST_BASE__LOGGING__LEVEL=DEBUG pytest
```

Environment variables win over the YAML, and only the field named is replaced —
its siblings still come from the file. This is how a deployment points at its
own MLflow or log level without editing a versioned config.

## Project layout

```
src/energycast/
├── config/     # typed Settings + YAML loading
├── data/       # load -> clean -> validate -> split
├── features/   # calendar + lags -> scale -> sequences
└── utils/      # structured logging
```

## The pipeline

```
load -> clean -> validate -> split -> calendar + lags -> scale -> sequences
```

Each step is a separate class, and only the loader knows where the data came
from — retraining on observations that never touched a CSV reuses the rest
unchanged.

Two rules the code enforces rather than documents, because both failure modes
are silent and score *better* when broken:

- **The split is chronological, and never shuffled.** The raw PJME file is not
  sorted, so a positional split would train on future hours and test on past
  ones.
- **Features are built per split, after splitting.** The scaler is fitted on
  train alone, and no window takes its lookback from another split.

```python
from energycast.data import CSVDataLoader, TimeSeriesCleaner, SchemaValidator, ChronologicalSplitter
from energycast.features import CalendarFeatureBuilder, LagFeatureBuilder, SeriesScaler, SequenceBuilder

cleaned = TimeSeriesCleaner.from_settings().clean(CSVDataLoader.from_settings().load())
SchemaValidator.from_settings().validate(cleaned)
splits = ChronologicalSplitter.from_settings().split(cleaned)

calendar, lags = CalendarFeatureBuilder(), LagFeatureBuilder.from_settings()
train = lags.build(calendar.build(splits.train)).dropna()

scaler = SeriesScaler(columns=["PJME_MW"]).fit(train)
dataset = SequenceBuilder.from_settings().build(scaler.transform(train))
# dataset.X -> (92778, 168, 22)   dataset.y -> (92778, 24)
```

Report every metric through `scaler.inverse_transform(...)`. Train sigma is
~6,500 MW, so an RMSE left in z-units reads 25x better than it is.

See `CLAUDE.md` for architectural decisions and the milestone roadmap.
