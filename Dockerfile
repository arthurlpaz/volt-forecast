# syntax=docker/dockerfile:1

# Builder: resolve and install the main dependencies into an in-project venv.
# Conda is deliberately absent here — it exists locally to pin the interpreter
# and supply CUDA, but the container has neither job: torch is CPU-only (pinned
# to the pytorch-cpu source in pyproject) and Python 3.11 comes from the base
# image. A slim image plus Poetry is the leaner container.
FROM python:3.11-slim AS builder

ENV POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_CREATE=true \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR /app

# Dependency layer first, so a source change does not re-resolve the tree.
COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --only main --no-root

# Then the package itself.
COPY src ./src
RUN poetry install --only main

# Runtime: carry only the venv and the code, on a clean slim base.
FROM python:3.11-slim AS runtime

# libgomp1 is the OpenMP runtime that lightgbm and xgboost link against.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY configs ./configs

EXPOSE 8000

# Serving is the default role; batch roles override the command:
#   python -m energycast.training     python -m energycast.retraining
CMD ["uvicorn", "--factory", "energycast.serving.app:create_app", \
     "--host", "0.0.0.0", "--port", "8000"]
