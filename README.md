# EnergyCast AI

Continuous learning platform for energy consumption forecasting, built on the
[PJM Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
dataset.

This is not a notebook. The goal is a production-shaped ML system: clean
architecture, typed configuration, tests, reproducibility, and a real MLOps
pipeline — training → registry → monitoring → drift detection → automated
retraining → champion/challenger promotion → serving.

## The MLOps loop

```mermaid
flowchart LR
  csv[PJME hourly CSV] --> train[Training<br/>+ MLflow Registry]
  train --> serve[Serving<br/>models:/...@champion]
  serve -->|POST /predict| mon[(monitoring.db)]
  mon -->|POST /actuals| metrics[GET /metrics<br/>rolling error]
  csv --> drift[GET /drift<br/>train vs recent]
  drift --> check{GET /retrain/check<br/>drift OR error OR volume}
  metrics --> check
  check -->|fires| cycle[Retrain challenger]
  cycle -->|beats champion| promote[Promote @champion]
  promote --> serve
```

Each layer is named for what it does in the ML lifecycle, not for its
transport, and each is a thin, testable unit:

```
src/energycast/
├── config/      # typed Settings + YAML loading (fails fast on a bad file)
├── data/        # load -> clean -> validate -> split
├── features/    # calendar + lags -> scale -> sequences | horizon targets
├── models/      # Model protocol, seasonal naive, four baselines, LSTM
├── training/    # prepare_data + TrainingPipeline + MLflow tracking/registry
├── evaluation/  # per-horizon metrics + Evaluator + champion selection
├── serving/     # Forecaster + FastAPI app (the HTTP transport lives here)
├── monitoring/  # PredictionStore (deferred join) + rolling metrics
├── drift/       # detect_drift (Evidently) + DriftDetector
├── retraining/  # trigger OR + orchestrator + champion/challenger promotion
└── utils/       # structured JSON logging
```

## Quickstart with Docker

The fastest way to see the whole loop running. One image plays every role;
`docker-compose.yml` runs serving and the MLflow UI, and training is a one-off
command against the same image. The raw data is mounted, never baked in.

```bash
# 1. Get the data (public, CC0 — no Kaggle API token needed)
kaggle datasets download -d robikscube/hourly-energy-consumption -p data/raw --unzip

# 2. Build the image
docker compose build

# 3. Train the roster once — this populates the shared state volume with the
#    MLflow registry that serving loads its champion from
docker compose run --rm serving python -m energycast.training

# 4. Bring up serving (:8000) and the MLflow UI (:5000)
docker compose up serving mlflow-ui
```

Then the API is at `http://localhost:8000` (interactive docs at `/docs`) and the
MLflow registry at `http://localhost:5000`. A retraining cycle is another one-off:

```bash
docker compose run --rm serving python -m energycast.retraining
```

All persistent state — the MLflow store, the monitoring/drift/retraining SQLite
files, and the Evidently snapshots — lives in the `state` volume, so it survives
restarts. Serving must find a registered champion at startup, so run training
before `up`.

## Local setup

Conda owns the interpreter; Poetry owns the dependencies. Conda pins Python
3.11 (numpy 1.26 ships no wheels for 3.13) and can supply CUDA runtimes if we
ever need them, while Poetry keeps the deterministic `poetry.lock` and the
dev/prod dependency split.

```bash
conda create -n energycast python=3.11 -y
conda activate energycast
poetry install
pre-commit install
```

> **Always activate the conda env before running Poetry.** This project sets
> `virtualenvs.create = false` in `poetry.toml` so Poetry installs into the
> active conda env rather than a second venv. Without `conda activate` first,
> `poetry install` targets whatever Python is on your PATH. Check with
> `which python`.

torch is the **CPU-only** build (see the `pytorch-cpu` source in
`pyproject.toml`); the CUDA build pulls ~5 GB of `nvidia-*` wheels the LSTM does
not need. The Docker image drops conda entirely — a container has no interpreter
or CUDA job for it to do.

## Running the loop locally

```bash
python -m energycast.training      # fit the roster, register each model + scaler
python -m energycast.evaluation    # score the roster, pick the champion
python -m energycast.serving       # serve the champion (uvicorn, :8000)
python -m energycast.retraining    # one orchestration cycle: check -> retrain -> promote
```

## API reference

The FastAPI app serves the champion and exposes the whole loop over HTTP. The
champion is resolved from the `@champion` registry alias, with the
`serving.champion_model` config as the bootstrap before any promotion.

| Method & path | What it does |
|---|---|
| `GET /health` | Champion name, its run id, which models are loaded |
| `POST /predict` | 24-hour forecast from a raw MW history; `?model=` serves a challenger; every forecast is logged |
| `POST /actuals` | Feed measured hours back to reconcile logged forecasts |
| `GET /metrics` | Rolling per-horizon RMSE/MAE/MAPE over the reconciled window |
| `GET /drift` | Evidently drift of the recent window vs the training distribution, plus an HTML snapshot |
| `GET /retrain/check` | The retrain decision and the signals it was read from, without training |

## Configuration

All configuration lives in `configs/` as YAML and is validated at startup by
`energycast.config.get_settings()`. Nothing is hardcoded.

| File | Owns |
|---|---|
| `base.yaml` | environment, paths, logging, MLflow, serving, monitoring, drift, retraining |
| `data.yaml` | data source, validation rules, train/val/test split |
| `model.yaml` | sequence length, horizon, features, LSTM, baselines |

Any field can be overridden by an environment variable prefixed with
`ENERGYCAST_`, using `__` as the nesting delimiter — this is how the Docker
compose file points every store at the mounted volume without editing a file:

```bash
ENERGYCAST_BASE__LOGGING__LEVEL=DEBUG pytest
```

Environment variables win over the YAML, and only the field named is replaced —
its siblings still come from the file.

## State: four SQLite stores

Each capability owns its own store, all gitignored, all pointed at the `state`
volume in Docker:

| Store | Written by | Holds |
|---|---|---|
| `mlflow.db` | training, retraining | runs, metrics, the Model Registry + `@champion`/`@challenger` aliases |
| `monitoring.db` | serving | served forecasts and the actuals that later score them |
| `drift.db` | drift | one row per drift run: the dataset-level verdict |
| `retraining.db` | retraining | one row per cycle: what tripped it, what was promoted |

## The data pipeline

```
load -> clean -> validate -> split -> calendar + lags -> scale -> sequences
```

Each step is a separate class, and only the loader knows where the data came
from — retraining on observations that never touched a CSV reuses the rest
unchanged. Two rules the code enforces rather than documents, because both
failure modes are silent and score *better* when broken:

- **The split is chronological, and never shuffled.** The raw PJME file is not
  sorted, so a positional split would train on future hours and test on past
  ones.
- **Features are built per split, after splitting.** The scaler is fitted on
  train alone, and no window takes its lookback from another split.

Report every metric through `scaler.inverse_transform(...)`. Train sigma is
~6,500 MW, so an RMSE left in z-units reads 25x better than it is.

## Models

Every model answers one question — given the features of hour t, what are hours
t+1 .. t+24 — and returns `(n_rows, 24)`. **Predicting t+1 instead is a
different, easier question**: the same model scores RMSE 507 at t+1 against
2,605 at t+24 — 5.1x apart.

`SeasonalNaiveModel` — the same hour one week back — is the bar, not a model.
Measured on the PJME test split:

| | RMSE | vs naive |
|---|---|---|
| seasonal naive (the bar) | 4,761 | — |
| linear regression | 2,699 | +43% |
| random forest | 2,218 | +53% |
| xgboost | 2,103 | +56% |
| lightgbm | 2,094 | +56% |
| LSTM (champion) | 1,973 | +59% |

Error is not uniform across the horizon: the best model scores ~390 MW at t+1
and ~2,500 at t+24. Report per-horizon, not just the 24-hour average.

## Tests

```bash
conda activate energycast
pytest        # 212 tests
```

Tests that exercise the real PJME file skip automatically when it is absent, so
the suite is green on a fresh clone before you download anything. Lint and format
are enforced by pre-commit (`black`, `ruff`).

## Known limitations

- **Drift rebuilds its reference from the CSV at serving time.** The training
  distribution is reprocessed on each `/drift` call, so the serving container
  mounts the raw data. Snapshotting the reference at training time — the way the
  scaler travels with the model — is the cleaner shape.
- **Champion promotion is per architecture.** MLflow aliases are per registered
  name, so retraining promotes a new version of the champion's architecture; a
  cross-architecture switch would need a global champion pointer.
- **Serving caches the champion at startup.** A promotion is served only after a
  restart; there is no live reload endpoint yet.
