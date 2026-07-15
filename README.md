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

## Running the tests

```bash
conda activate energycast
pytest
```

## Configuration

All configuration lives in `configs/` as YAML and is validated at startup by
`energycast.config.get_settings()`. Nothing is hardcoded.

| File | Owns |
|---|---|
| `base.yaml` | environment, paths, logging, MLflow |
| `data.yaml` | data source, validation rules, train/val/test split |
| `model.yaml` | sequence length, horizon, LSTM, baselines |

Any field can be overridden by an environment variable prefixed with
`ENERGYCAST_`, using `__` as the nesting delimiter:

```bash
ENERGYCAST_BASE__MLFLOW__TRACKING_URI=postgresql://... poetry run pytest
```

## Project layout

```
src/energycast/
├── config/   # typed Settings + YAML loading
└── utils/    # structured logging
```

See `CLAUDE.md` for architectural decisions and the milestone roadmap.
