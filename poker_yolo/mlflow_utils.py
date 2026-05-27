from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from poker_yolo.config import Config

logger = logging.getLogger(__name__)


def setup_mlflow(config: Config, run_name: str | None = None) -> mlflow.ActiveRun:
    os.environ.setdefault("MLFLOW_TRACKING_URI", config.mlflow_tracking_uri)
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)

    _wait_for_mlflow(config.mlflow_tracking_uri)

    experiment = mlflow.set_experiment(config.mlflow_experiment)
    active_run = mlflow.start_run(
        run_name=run_name or config.mlflow_run_name or f"poker-yolo-{int(time.time())}",
        experiment_id=experiment.experiment_id,
    )
    mlflow.set_tag("project", "poker-yolo")
    return active_run


def _wait_for_mlflow(tracking_uri: str, retries: int = 30, delay: float = 2.0) -> None:
    client = MlflowClient(tracking_uri=tracking_uri)
    for attempt in range(1, retries + 1):
        try:
            client.search_experiments(max_results=1)
            logger.info("MLflow server is reachable at %s", tracking_uri)
            return
        except Exception as exc:
            logger.warning(
                "MLflow not ready (attempt %d/%d): %s",
                attempt,
                retries,
                exc,
            )
            time.sleep(delay)
    logger.warning("Proceeding without confirmed MLflow connectivity")


def log_config(config: Config) -> None:
    params = {
        "model_weights": config.model_weights,
        "imgsz": config.imgsz,
        "epochs": config.epochs,
        "batch": config.batch,
        "patience": config.patience,
        "lr0": config.lr0,
        "lrf": config.lrf,
        "weight_decay": config.weight_decay,
        "warmup_epochs": config.warmup_epochs,
        "data_yaml": str(config.data_yaml),
        **config.augmentations.to_mlflow_params(),
    }
    mlflow.log_params(params)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    clean = {k: float(v) for k, v in metrics.items() if v is not None}
    if clean:
        mlflow.log_metrics(clean, step=step)


def log_ultralytics_results(results: Any, prefix: str = "") -> None:
    if results is None:
        return

    box = getattr(results, "box", None)
    if box is None:
        return

    metric_map = {
        "map50": getattr(box, "map50", None),
        "map": getattr(box, "map", None),
        "precision": getattr(box, "mp", None),
        "recall": getattr(box, "mr", None),
    }
    logged = {}
    for name, value in metric_map.items():
        if value is not None:
            key = f"{prefix}{name}" if prefix else name
            logged[key] = float(value)

    if logged:
        mlflow.log_metrics(logged)

    maps = getattr(box, "maps", None)
    if maps is not None:
        per_class = {f"class_map_{i}": float(v) for i, v in enumerate(maps) if v is not None}
        if per_class:
            mlflow.log_metrics(per_class)


def log_artifact_dir(directory: Path, artifact_path: str | None = None) -> None:
    if directory.exists():
        mlflow.log_artifacts(str(directory), artifact_path=artifact_path)


def log_json_artifact(data: dict[str, Any], filename: str) -> None:
    path = Path(filename)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    mlflow.log_artifact(str(path))
    path.unlink(missing_ok=True)
