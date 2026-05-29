"""Extract metrics from Ultralytics train/val results."""

from __future__ import annotations

from typing import Any

CLASSIFY_ALIASES: tuple[tuple[str, str], ...] = (
    ("metrics/accuracy_top1", "top1"),
    ("metrics/accuracy_top5", "top5"),
    ("val/loss", "loss"),
    ("train/loss", "train_loss"),
    ("fitness", "fitness"),
)


def extract_metrics(results: Any) -> dict[str, float]:
    """Normalize classify or detect validation/train metrics from Ultralytics results."""
    data = getattr(results, "results_dict", None) or {}
    if isinstance(data, dict):
        metrics: dict[str, float] = {}
        for src, dst in CLASSIFY_ALIASES:
            value = data.get(src)
            if value is not None:
                metrics[dst] = float(value)
        if metrics:
            return metrics

    box = getattr(results, "box", None)
    if box is None:
        return {}

    mp, mr = getattr(box, "mp", None), getattr(box, "mr", None)
    f1 = None
    if mp is not None and mr is not None and (mp + mr) > 0:
        f1 = 2 * mp * mr / (mp + mr)

    detect = {
        "map50": float(getattr(box, "map50", 0.0) or 0.0),
        "map50_95": float(getattr(box, "map", 0.0) or 0.0),
        "precision": float(mp or 0.0),
        "recall": float(mr or 0.0),
    }
    if f1 is not None:
        detect["f1"] = float(f1)
    return detect


def metrics_for_mlflow(results: Any, prefix: str = "") -> dict[str, float]:
    """Flat metric dict suitable for ``mlflow.log_metrics``."""
    data = getattr(results, "results_dict", None)
    if isinstance(data, dict) and data:
        logged: dict[str, float] = {}
        for key, value in data.items():
            if value is None:
                continue
            metric_name = str(key).replace("/", "_")
            out_key = f"{prefix}{metric_name}" if prefix else metric_name
            logged[out_key] = float(value)
        if logged:
            return logged

    extracted = extract_metrics(results)
    logged = {f"{prefix}{k}" if prefix else k: v for k, v in extracted.items()}

    box = getattr(results, "box", None)
    maps = getattr(box, "maps", None) if box is not None else None
    if maps is not None:
        for i, value in enumerate(maps):
            if value is not None:
                logged[f"{prefix}class_map_{i}" if prefix else f"class_map_{i}"] = float(value)

    return logged
