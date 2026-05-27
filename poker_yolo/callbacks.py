from __future__ import annotations

import logging
import time
from typing import Any

import mlflow

from poker_yolo.monitoring import ResourceMonitor
from poker_yolo.reporting import get_report, log_event

logger = logging.getLogger(__name__)

_train_monitor: ResourceMonitor | None = None


def _sanitize_mlflow_key(key: str) -> str:
    return key.replace("(", "_").replace(")", "").replace(" ", "_").replace("/", "_")


def set_train_monitor(monitor: ResourceMonitor) -> None:
    global _train_monitor
    _train_monitor = monitor


def _on_train_start(trainer: Any) -> None:
    if _train_monitor:
        _train_monitor.start_background()
    log_event("train.loop_start", epochs=getattr(trainer, "epochs", None))


def _on_train_end(trainer: Any) -> None:
    if _train_monitor:
        _train_monitor.stop_background()
        report = get_report()
        if report:
            report.set_resources(_train_monitor.summary())


def mlflow_epoch_callback(trainer: Any) -> None:
    """Log per-epoch metrics from Ultralytics trainer to MLflow and run report."""
    if _train_monitor:
        _train_monitor.sample_once()

    raw_metrics = {}
    for key, value in trainer.metrics.items():
        if isinstance(value, (int, float)):
            raw_metrics[key] = float(value)

    if not raw_metrics:
        return

    step = trainer.epoch
    log_event("train.epoch", epoch=step, **raw_metrics)

    if mlflow.active_run() is None:
        return

    metrics = {f"epoch_{_sanitize_mlflow_key(k)}": v for k, v in raw_metrics.items()}
    mlflow.log_metrics(metrics, step=step)
    logger.debug("Logged epoch %d metrics to MLflow", step)


def register_mlflow_callbacks(model: Any) -> None:
    """Attach MLflow logging and resource monitoring callbacks to YOLO model."""
    model.add_callback("on_train_start", _on_train_start)
    model.add_callback("on_train_end", _on_train_end)
    model.add_callback("on_train_epoch_end", mlflow_epoch_callback)
    model.add_callback("on_fit_epoch_end", mlflow_epoch_callback)
