from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from poker_yolo.callbacks import register_mlflow_callbacks, set_train_monitor
from poker_yolo.config import Config
from poker_yolo.mlflow_utils import log_config, log_ultralytics_results, setup_mlflow
from poker_yolo.monitoring import ResourceMonitor
from poker_yolo.reporting import get_report, log_event

logger = logging.getLogger(__name__)


def run_training(config: Config) -> tuple[Path, float]:
    log_event(
        "train.start",
        model=config.model_weights,
        epochs=config.epochs,
        batch=config.batch,
        imgsz=config.imgsz,
    )

    setup_mlflow(config, run_name=f"train-{config.name}")
    log_config(config)

    train_monitor = ResourceMonitor("train")
    set_train_monitor(train_monitor)
    train_monitor.sample_once()

    model = YOLO(config.model_weights)
    register_mlflow_callbacks(model)

    train_kwargs = config.train_args()
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

    metrics_dir = Path(results.save_dir)
    results_csv = metrics_dir / "results.csv"
    if results_csv.exists():
        log_artifact_safe(results_csv, "training")

    log_ultralytics_results(results, prefix="train_")

    report = get_report()
    if report:
        report.set_resources(resource_summary)
        report.set_metrics({"train_duration_sec": train_duration})
        train_metrics = _extract_train_metrics(results)
        if train_metrics:
            report.set_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        report.set_artifact("weights", weights)

    if best_weights.exists():
        import mlflow

        mlflow.log_artifact(str(best_weights), artifact_path="weights")
        log_event("train.weights_saved", path=str(best_weights))

    import mlflow

    if mlflow.active_run() is not None:
        mlflow.end_run()

    log_event("train.complete", weights=str(weights), save_dir=str(metrics_dir), duration_sec=train_duration)
    return weights, train_duration


def _extract_train_metrics(results: Any) -> dict[str, float]:
    box = getattr(results, "box", None)
    if box is None:
        return {}
    metrics = {}
    for src, dst in (("map50", "map50"), ("map", "map50_95"), ("mp", "precision"), ("mr", "recall")):
        value = getattr(box, src, None)
        if value is not None:
            metrics[dst] = float(value)
    return metrics


def log_artifact_safe(path: Path, artifact_path: str) -> None:
    import mlflow

    try:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
    except Exception as exc:
        logger.warning("Failed to log artifact %s: %s", path, exc)
