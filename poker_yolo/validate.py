from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from poker_yolo.config import Config
from poker_yolo.mlflow_utils import (
    log_config,
    log_json_artifact,
    log_metrics,
    log_ultralytics_results,
    setup_mlflow,
)
from poker_yolo.monitoring import ResourceMonitor
from poker_yolo.reporting import get_report, log_event

logger = logging.getLogger(__name__)


def run_validation(config: Config, weights: Path) -> tuple[dict[str, float], float]:
    log_event(
        "validate.start",
        weights=str(weights),
        split=config.val_split,
        metric_conf=config.val_metric_conf,
        iou=config.val_iou,
    )

    setup_mlflow(config, run_name=f"validate-{config.name}")
    log_config(config)

    import mlflow

    mlflow.log_param("weights", str(weights))

    val_monitor = ResourceMonitor("val")
    val_monitor.start_background()

    model = YOLO(str(weights))
    t0 = time.perf_counter()
    results = model.val(
        data=str(config.data_yaml),
        split=config.val_split,
        imgsz=config.imgsz,
        conf=config.val_metric_conf,
        iou=config.val_iou,
        device=config.device,
        verbose=True,
    )
    val_duration = time.perf_counter() - t0
    val_monitor.stop_background()

    metrics = extract_metrics(results)
    log_ultralytics_results(results, prefix="val_")
    log_metrics(metrics)

    summary = {
        "split": config.val_split,
        "weights": str(weights),
        **metrics,
    }
    log_json_artifact(summary, "validation_summary.json")

    report = get_report()
    if report:
        report.set_metrics({f"val_{k}": v for k, v in metrics.items()})
        report.set_resources(val_monitor.summary())
        report.set_metrics({"val_duration_sec": val_duration})
        report.set_artifact("weights", weights)

    if mlflow.active_run() is not None:
        mlflow.end_run()

    log_event("validate.complete", **metrics, duration_sec=val_duration)
    logger.info("Validation metrics: %s", metrics)
    return metrics, val_duration


def extract_metrics(results: Any) -> dict[str, float]:
    box = getattr(results, "box", None)
    if box is None:
        return {}

    f1 = None
    mp, mr = getattr(box, "mp", None), getattr(box, "mr", None)
    if mp is not None and mr is not None and (mp + mr) > 0:
        f1 = 2 * mp * mr / (mp + mr)

    metrics = {
        "map50": float(getattr(box, "map50", 0.0) or 0.0),
        "map50_95": float(getattr(box, "map", 0.0) or 0.0),
        "precision": float(mp or 0.0),
        "recall": float(mr or 0.0),
    }
    if f1 is not None:
        metrics["f1"] = float(f1)
    return metrics
