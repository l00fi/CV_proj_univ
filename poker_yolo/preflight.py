"""CPU preflight before GPU training to verify dataset and pipeline."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from poker_yolo.config import Config
from poker_yolo.reporting import log_event
from poker_yolo.train import run_training

logger = logging.getLogger(__name__)


def run_preflight_cpu_smoke(config: Config) -> tuple[Path, float] | None:
    """Run a short CPU training pass before the main GPU run (``preflight_cpu_epochs``)."""
    if config.preflight_cpu_epochs <= 0:
        return None

    preflight = replace(
        config,
        device="cpu",
        epochs=config.preflight_cpu_epochs,
        workers=0,
        batch=min(config.batch, 4),
        patience=min(config.patience, config.preflight_cpu_epochs),
        name=f"{config.name}_preflight",
        gpu_resource_fraction=None,
        mlflow_run_name=f"preflight-{config.name}",
    )

    log_event(
        "train.preflight.start",
        epochs=preflight.epochs,
        device=preflight.device,
        batch=preflight.batch,
        name=preflight.name,
    )
    logger.info(
        "Preflight: %d epoch(s) on CPU (batch=%d) before GPU training",
        preflight.epochs,
        preflight.batch,
    )

    weights, duration = run_training(preflight)

    log_event("train.preflight.complete", weights=str(weights), duration_sec=duration)
    logger.info("Preflight finished in %.1fs — weights: %s", duration, weights)
    return weights, duration
