"""Tests for Grafana JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from poker_yolo.grafana_artifacts import infinity_row_selector, write_grafana_artifacts
from poker_yolo.hands import COMBO_CLASSES


def test_write_grafana_artifacts_full_grids(tmp_path: Path) -> None:
    outcomes = {c: {"correct": 1 if c == "pair" else 0, "incorrect": 0} for c in COMBO_CLASSES}
    confusion = {t: {p: int(t == p) for p in COMBO_CLASSES} for t in COMBO_CLASSES}
    curves = [{"epoch": 1, "train_loss": 1.9, "val_loss": 1.1, "top1": 0.5, "top5": 0.9}]

    write_grafana_artifacts(
        tmp_path,
        outcomes_by_hand=outcomes,
        confusion=confusion,
        training_curves=curves,
    )

    outcomes_doc = json.loads((tmp_path / "grafana" / "benchmark_outcomes.json").read_text(encoding="utf-8"))
    assert len(outcomes_doc) == 10
    assert outcomes_doc[0]["hand_class"] in COMBO_CLASSES

    confusion_doc = json.loads((tmp_path / "grafana" / "confusion_matrix.json").read_text(encoding="utf-8"))
    assert len(confusion_doc) == 100

    curves_doc = json.loads((tmp_path / "grafana" / "training_curves.json").read_text(encoding="utf-8"))
    assert curves_doc[0]["train_loss"] == 1.9
    assert "time" in curves_doc[0]
    assert "$.rows" in infinity_row_selector() or "array" in infinity_row_selector()
