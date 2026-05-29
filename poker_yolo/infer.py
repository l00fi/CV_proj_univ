from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from ultralytics import YOLO

from poker_yolo.config import Config
from poker_yolo.infer_source import resolve_infer_source
from poker_yolo.mlflow_phase import mlflow_phase
from poker_yolo.mlflow_utils import log_artifact_dir, log_metrics
from poker_yolo.reporting import get_report, log_event

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceRun:
    output_dir: Path
    results: list[Any]


def run_inference(
    config: Config,
    weights: Path,
    source: Path,
    save: bool = True,
) -> InferenceRun:
    predict_source = resolve_infer_source(source)
    source_label = (
        str(predict_source)
        if isinstance(predict_source, Path)
        else f"{source} ({len(predict_source)} images)"
    )
    log_event(
        "infer.start",
        weights=str(weights),
        source=source_label,
        conf=config.infer_conf,
        save=save,
    )

    with mlflow_phase(config, f"infer-{config.name}"):
        mlflow.log_param("weights", str(weights))
        mlflow.log_param("source", source_label)

        output_dir = config.infer_save_dir / f"pred_{int(time.time())}"
        output_dir.mkdir(parents=True, exist_ok=True)

        model = YOLO(str(weights), task=config.task)
        t0 = time.perf_counter()
        results = model.predict(
            task=config.task,
            source=predict_source,
            imgsz=config.imgsz,
            conf=config.infer_conf,
            iou=config.infer_iou,
            device=config.device,
            save=save,
            project=str(output_dir.parent),
            name=output_dir.name,
            exist_ok=True,
        )
        elapsed = time.perf_counter() - t0
        del model

        n_images = len(results)
        latency_ms = (elapsed / max(n_images, 1)) * 1000
        fps = n_images / elapsed if elapsed > 0 else 0.0

        infer_metrics: dict[str, float] = {
            "infer_latency_ms": latency_ms,
            "infer_fps": fps,
            "infer_images": float(n_images),
        }
        top1_conf_total = 0.0
        top1_conf_count = 0
        top1_counts: dict[int, int] = {}
        for result in results:
            probs = getattr(result, "probs", None)
            if probs is None:
                continue
            top1_idx = int(getattr(probs, "top1", -1))
            top1_conf = float(getattr(probs, "top1conf", 0.0))
            if top1_idx >= 0:
                top1_counts[top1_idx] = top1_counts.get(top1_idx, 0) + 1
                top1_conf_total += top1_conf
                top1_conf_count += 1
        if top1_conf_count:
            infer_metrics["infer_top1_conf_avg"] = top1_conf_total / top1_conf_count
        for cls_id, count in sorted(top1_counts.items()):
            infer_metrics[f"infer_top1_class_{cls_id}"] = float(count)

        log_metrics(infer_metrics)
        pred_dir = output_dir if output_dir.exists() else Path(results[0].save_dir) if results else output_dir
        if pred_dir.exists():
            log_artifact_dir(pred_dir, artifact_path="predictions")

        report = get_report()
        if report:
            report.set_metrics(infer_metrics)
            report.set_artifact("predictions", pred_dir)
            report.set_artifact("weights", weights)

    log_event("infer.complete", **infer_metrics, output=str(pred_dir))
    logger.info(
        "Inference complete: %d images, %.1f ms/image, %.1f FPS -> %s",
        n_images,
        latency_ms,
        fps,
        pred_dir,
    )
    return InferenceRun(pred_dir, list(results))
