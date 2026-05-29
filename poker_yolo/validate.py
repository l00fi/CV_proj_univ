from __future__ import annotations

import logging
import time
from pathlib import Path

import mlflow
from ultralytics import YOLO

from poker_yolo.config import Config
from poker_yolo.kaggle_dataset import ensure_hands_data_yaml
from poker_yolo.mlflow_phase import mlflow_phase
from poker_yolo.mlflow_utils import log_json_artifact, log_metrics, log_ultralytics_results
from poker_yolo.monitoring import ResourceMonitor
from poker_yolo.reporting import get_report, log_event
from poker_yolo.ultralytics_metrics import extract_metrics

logger = logging.getLogger(__name__)

__all__ = ["extract_metrics", "run_validation"]


def run_validation(config: Config, weights: Path) -> tuple[dict[str, float], float]:
    log_event(
        "validate.start",
        weights=str(weights),
        split=config.val_split,
        metric_conf=config.val_metric_conf,
        iou=config.val_iou,
    )

    with mlflow_phase(config, f"validate-{config.name}"):
        mlflow.log_param("weights", str(weights))

        val_monitor = ResourceMonitor("val")
        val_monitor.start_background()
        ensure_hands_data_yaml(config.val_data_yaml)

        model = YOLO(str(weights), task=config.task)
        t0 = time.perf_counter()
        val_data = str(config.dataset_root) if config.task == "classify" else str(config.data_yaml)
        results = model.val(
            task=config.task,
            data=val_data,
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
        log_json_artifact(
            {"split": config.val_split, "weights": str(weights), **metrics},
            "validation_summary.json",
        )

        report = get_report()
        if report:
            report.set_metrics({f"val_{k}": v for k, v in metrics.items()})
            report.set_resources(val_monitor.summary())
            report.set_metrics({"val_duration_sec": val_duration})
            report.set_artifact("weights", weights)

    log_event("validate.complete", **metrics, duration_sec=val_duration)
    logger.info("Validation metrics: %s", metrics)
    return metrics, val_duration
