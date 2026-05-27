# syntax=docker/dockerfile:1
# Training image: PyTorch 2.x Linux wheels bundle CUDA libs; host driver via NVIDIA Container Toolkit.
# Run: docker compose run --rm --gpus all poker-yolo train --config configs/default.yaml

FROM python:3.11-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:/root/.local/bin:${PATH}" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install dependencies (cached layer; retry on transient PyPI/CDN failures)
COPY pyproject.toml uv.lock ./
RUN for attempt in 1 2 3; do \
      uv sync --no-dev --no-install-project && break; \
      echo "uv sync attempt $attempt failed, retrying..."; \
      sleep $((attempt * 15)); \
    done

# Copy project and install
COPY . .
RUN for attempt in 1 2 3; do \
      uv sync --no-dev && break; \
      echo "uv sync attempt $attempt failed, retrying..."; \
      sleep $((attempt * 15)); \
    done

RUN sed -i 's/\r$//' /app/scripts/entrypoint.sh /app/scripts/resolve_device.py \
    && chmod +x /app/scripts/entrypoint.sh /app/scripts/resolve_device.py

VOLUME ["/app/runs", "/app/dataset"]

ENV MLFLOW_TRACKING_URI=http://mlflow:5000

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["train", "--config", "configs/default.yaml"]
