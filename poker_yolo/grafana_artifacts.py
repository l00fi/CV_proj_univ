"""JSON artifacts under ``runs/reports/grafana/`` for Grafana Infinity panels."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from poker_yolo.benchmark_grids import normalize_confusion_matrix, normalize_outcomes_by_hand
from poker_yolo.hands import COMBO_CLASSES

logger = logging.getLogger(__name__)

# Synthetic timestamps so Grafana timeseries panels get a proper time field (epoch is not enough).
_CURVE_TIME_BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _curves_with_time(curves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in curves:
        epoch = int(row.get("epoch", 0))
        ts = _CURVE_TIME_BASE + timedelta(hours=max(epoch - 1, 0))
        out.append({**row, "time": ts.isoformat().replace("+00:00", "Z")})
    return out


def _validate_table_json(path: Path, *, min_rows: int = 1) -> None:
    """Ensure Infinity table panels get a top-level JSON array, not ``{classes, rows}``."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(doc, dict) and "rows" in doc:
        raise ValueError(
            f"{path.name}: legacy wrapped format detected; expected a JSON array. "
            "Rebuild the Docker image and rerun train or backfill_grafana_artifacts.py."
        )
    if not isinstance(doc, list):
        raise ValueError(f"{path.name}: expected JSON array, got {type(doc).__name__}")
    if len(doc) < min_rows:
        logger.warning("%s has fewer than %d rows (%d)", path.name, min_rows, len(doc))


# Infinity JSONata: flat array OR legacy ``{rows: [...]}`` from older images.
_INFINITY_ROW_SELECTOR = '$type($) = "array" ? $ : $.rows'


def write_grafana_artifacts(
    report_dir: Path,
    *,
    outcomes_by_hand: dict[str, dict[str, int]],
    confusion: dict[str, dict[str, int]],
    training_curves: list[dict[str, Any]],
) -> Path:
    """Write ``report_dir/grafana/*.json`` consumed by Grafana Infinity datasource."""
    grafana_dir = report_dir / "grafana"
    grafana_dir.mkdir(parents=True, exist_ok=True)

    outcomes = normalize_outcomes_by_hand(outcomes_by_hand)
    outcome_rows = [
        {
            "hand_class": hand,
            "hand_index": idx,
            "correct": int(outcomes[hand]["correct"]),
            "incorrect": int(outcomes[hand]["incorrect"]),
        }
        for idx, hand in enumerate(COMBO_CLASSES)
    ]
    # Top-level JSON array — Infinity backend parser + table format (no root_selector).
    outcomes_path = grafana_dir / "benchmark_outcomes.json"
    outcomes_path.write_text(
        json.dumps(outcome_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _validate_table_json(outcomes_path, min_rows=10)

    confusion = normalize_confusion_matrix(confusion)
    confusion_rows = [
        {
            "true_class": true,
            "true_index": t_idx,
            "pred_class": pred,
            "pred_index": p_idx,
            "count": int(confusion[true][pred]),
        }
        for t_idx, true in enumerate(COMBO_CLASSES)
        for p_idx, pred in enumerate(COMBO_CLASSES)
    ]
    confusion_path = grafana_dir / "confusion_matrix.json"
    confusion_path.write_text(
        json.dumps(confusion_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _validate_table_json(confusion_path, min_rows=10)

    curves_path = grafana_dir / "training_curves.json"
    curves_with_time = _curves_with_time(training_curves)
    curves_path.write_text(
        json.dumps(curves_with_time, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if curves_with_time and "time" not in curves_with_time[0]:
        raise ValueError(f"{curves_path.name}: missing required 'time' field for Grafana timeseries")
    if not training_curves:
        logger.warning("No training curves written for Grafana (missing results.csv?)")

    return grafana_dir


def infinity_row_selector() -> str:
    """Root selector for Grafana Infinity panels (supports flat array and legacy wrap)."""
    return _INFINITY_ROW_SELECTOR
