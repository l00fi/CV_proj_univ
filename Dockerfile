# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:/root/.local/bin:${PATH}"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install dependencies (cached layer)
COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project

# Copy project and install
COPY . .
RUN uv sync --no-dev

RUN chmod +x /app/scripts/entrypoint.sh

VOLUME ["/app/runs", "/app/dataset"]

ENV MLFLOW_TRACKING_URI=http://mlflow:5000

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["train"]
