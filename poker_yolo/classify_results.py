"""Parse YOLO classification predict/val results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO

from poker_yolo.config import Config


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def extract_top_classes(result: Any, limit: int = 5) -> list[dict[str, Any]]:
    """Top-k class predictions from a single Ultralytics classify result."""
    probs = getattr(result, "probs", None)
    if probs is None:
        return []
    names = getattr(result, "names", {}) or {}
    top_ids = as_list(getattr(probs, "top5", None))[:limit]
    top_confs = [float(v) for v in as_list(getattr(probs, "top5conf", None))[:limit]]
    ranked: list[dict[str, Any]] = []
    for class_id, confidence in zip(top_ids, top_confs):
        class_id = int(class_id)
        ranked.append(
            {
                "class_id": class_id,
                "class_name": str(names.get(class_id, class_id)),
                "confidence": confidence,
            }
        )
    return ranked


def predict_top_classes(
    config: Config,
    weights: Path,
    image_path: Path,
) -> list[tuple[str, float]]:
    """Run classify predict on one image; return ``(class_name, confidence)`` pairs."""
    model = YOLO(str(weights), task=config.task)
    results = model.predict(
        task=config.task,
        source=str(image_path),
        imgsz=config.imgsz,
        conf=config.infer_conf,
        iou=config.infer_iou,
        device=config.device,
        save=False,
        verbose=False,
    )
    del model
    ranked = extract_top_classes(results[0])
    return [(str(item["class_name"]), float(item["confidence"])) for item in ranked]
