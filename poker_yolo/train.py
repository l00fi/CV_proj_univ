from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import mlflow
from ultralytics import YOLO

from poker_yolo.callbacks import register_mlflow_callbacks, set_train_monitor
from poker_yolo.config import Config
from poker_yolo.device import apply_training_resource_limits, effective_training_workers
from poker_yolo.mlflow_phase import mlflow_phase
from poker_yolo.mlflow_utils import log_artifact_dir, log_ultralytics_results
from poker_yolo.monitoring import ResourceMonitor
from poker_yolo.reporting import get_report, log_event
from poker_yolo.ultralytics_metrics import extract_metrics

logger = logging.getLogger(__name__)


def run_training(config: Config) -> tuple[Path, float]:
    log_event(
        "train.start",
        model=config.model_weights,
        epochs=config.epochs,
        batch=config.batch,
        imgsz=config.imgsz,
    )

    with mlflow_phase(config, f"train-{config.name}"):
        train_monitor = ResourceMonitor("train")
        set_train_monitor(train_monitor)
        train_monitor.sample_once()

        model = YOLO(config.model_weights, task=config.task)
        register_mlflow_callbacks(model)

        if config.gpu_resource_fraction is not None:
            apply_training_resource_limits(config.device, config.gpu_resource_fraction)
            log_event(
                "train.resource_limit",
                fraction=config.gpu_resource_fraction,
                device=config.device,
            )

        train_kwargs = config.train_args()
        if config.gpu_resource_fraction is not None:
            adjusted_workers = effective_training_workers(
                config.workers,
                config.gpu_resource_fraction,
            )
            if adjusted_workers != train_kwargs["workers"]:
                logger.info(
                    "Training limit: DataLoader workers %s -> %s",
                    train_kwargs["workers"],
                    adjusted_workers,
                )
                train_kwargs["workers"] = adjusted_workers

        n_alb = len(train_kwargs.get("augmentations", []))
        log_event(
            "train.augmentations",
            mosaic=config.augmentations.mosaic,
            mixup=config.augmentations.mixup,
            copy_paste=config.augmentations.copy_paste,
            cutmix=config.augmentations.cutmix,
            albumentations=n_alb,
        )

        t0 = time.perf_counter()
        results = model.train(**train_kwargs)
        train_duration = time.perf_counter() - t0

        train_monitor.stop_background()
        resource_summary = train_monitor.summary()

        best_weights = Path(results.save_dir) / "weights" / "best.pt"
        last_weights = Path(results.save_dir) / "weights" / "last.pt"
        weights = best_weights if best_weights.exists() else last_weights

        results_csv = Path(results.save_dir) / "results.csv"
        if results_csv.exists():
            _log_artifact_safe(results_csv, "training")

        log_ultralytics_results(results, prefix="train_")

        report = get_report()
        if report:
            report.set_resources(resource_summary)
            report.set_metrics({"train_duration_sec": train_duration})
            train_metrics = extract_metrics(results)
            if train_metrics:
                report.set_metrics({f"train_{k}": v for k, v in train_metrics.items()})
            report.set_artifact("weights", weights)

        if best_weights.exists():
            mlflow.log_artifact(str(best_weights), artifact_path="weights")
            log_event("train.weights_saved", path=str(best_weights))

    log_event("train.complete", weights=str(weights), duration_sec=train_duration)
    return weights, train_duration


def _log_artifact_safe(path: Path, artifact_path: str) -> None:
    try:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
    except Exception as exc:
        logger.warning("Failed to log artifact %s: %s", path, exc)
