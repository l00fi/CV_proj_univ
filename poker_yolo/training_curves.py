"""Parse Ultralytics ``results.csv`` for Grafana training-curve metrics."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def results_csv_path(weights: Path) -> Path:
    """``best.pt`` lives in ``.../weights/``; ``results.csv`` is in the run root."""
    return weights.resolve().parent.parent / "results.csv"


def parse_training_results_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                epoch_raw = raw.get("epoch")
                if epoch_raw is None:
                    continue
                epoch = int(float(epoch_raw))
                point: dict[str, Any] = {"epoch": epoch}
                for key, series in (
                    ("train/loss", "train_loss"),
                    ("val/loss", "val_loss"),
                    ("metrics/accuracy_top1", "top1"),
                    ("metrics/accuracy_top5", "top5"),
                ):
                    value = raw.get(key)
                    if value is not None and str(value).strip() != "":
                        point[series] = float(value)
                rows.append(point)
    except (OSError, ValueError, csv.Error) as exc:
        logger.warning("Failed to parse training results %s: %s", path, exc)
        return []
    return rows
