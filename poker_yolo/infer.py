from __future__ import annotations

import logging
import time
from pathlib import Path

from ultralytics import YOLO

from poker_yolo.config import Config
from poker_yolo.mlflow_utils import log_artifact_dir, log_config, log_metrics, setup_mlflow
from poker_yolo.reporting import get_report, log_event

logger = logging.getLogger(__name__)


def run_inference(
    config: Config,
    weights: Path,
    source: Path,
    save: bool = True,
) -> Path:
    log_event(
        "infer.start",
        weights=str(weights),
        source=str(source),
        conf=config.infer_conf,
        save=save,
    )

    setup_mlflow(config, run_name=f"infer-{config.name}")
    log_config(config)

    import mlflow

    mlflow.log_param("weights", str(weights))
    mlflow.log_param("source", str(source))

    output_dir = config.infer_save_dir / f"pred_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))

    start = time.perf_counter()
    results = model.predict(
        source=str(source),
        imgsz=config.imgsz,
        conf=config.infer_conf,
        iou=config.infer_iou,
        device=config.device,
        save=save,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
    )
    elapsed = time.perf_counter() - start

    n_images = len(results)
    latency_ms = (elapsed / max(n_images, 1)) * 1000
    fps = n_images / elapsed if elapsed > 0 else 0.0

    infer_metrics = {
        "infer_latency_ms": latency_ms,
        "infer_fps": fps,
        "infer_images": float(n_images),
    }
    log_metrics(infer_metrics)

    pred_dir = output_dir if output_dir.exists() else Path(results[0].save_dir) if results else output_dir
    if pred_dir.exists():
        log_artifact_dir(pred_dir, artifact_path="predictions")

    report = get_report()
    if report:
        report.set_metrics(infer_metrics)
        report.set_artifact("predictions", pred_dir)
        report.set_artifact("weights", weights)

    if mlflow.active_run() is not None:
        mlflow.end_run()

    log_event("infer.complete", **infer_metrics, output=str(pred_dir))
    logger.info(
        "Inference complete: %d images, %.1f ms/image, %.1f FPS -> %s",
        n_images,
        latency_ms,
        fps,
        pred_dir,
    )
    return pred_dir
