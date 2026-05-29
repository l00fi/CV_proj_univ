"""Tests for interactive HTML pipeline reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from poker_yolo.html_report import (
    load_report_json,
    render_html,
    report_from_dict,
    write_html_report,
    write_html_report_from_json,
)
from poker_yolo.reporting import RunReport
from poker_yolo.hands import COMBO_CLASSES


def _full_report() -> RunReport:
    report = RunReport(
        run_id="20260101_120000_train",
        phase="train",
        started_at=datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
        config_name="poker_cards",
    )
    report.finished_at = datetime(2026, 1, 1, 11, 0, 0, tzinfo=timezone.utc)
    report.status = "success"
    report.set_params({"epochs": 50, "batch": 16, "device": "0"})
    report.set_dataset_stats(
        {
            "train_images": 1000,
            "test_images": 200,
            "num_classes": 52,
            "benchmark_test_images": 150,
        }
    )
    report.set_metrics({"val_top1": 0.91, "train_duration_sec": 3600.0})
    report.set_resources({"train_cpu_avg_pct": 80.0, "train_ram_avg_mb": 4096.0})
    report.set_training_curves(
        [
            {"epoch": 1, "train_loss": 1.2, "val_loss": 1.1, "top1": 0.5, "top5": 0.8},
            {"epoch": 2, "train_loss": 0.9, "val_loss": 0.85, "top1": 0.7, "top5": 0.9},
        ]
    )
    report.set_production({"hands_benchmark_accuracy": 0.88, "hands_benchmark_images": 150.0})
    report.set_hands_benchmark_stats(
        {"hands_benchmark_accuracy": 0.88, "hands_benchmark_images": 150.0},
        {"pair": 10},
        outcomes_by_hand={COMBO_CLASSES[0]: {"correct": 8, "incorrect": 2}},
        confusion={COMBO_CLASSES[0]: {COMBO_CLASSES[0]: 8, COMBO_CLASSES[1]: 2}},
    )
    report.set_augmentations_summary(
        {"synthetic_to_real_ratio": 2.0, "yolo_probabilities": {"mosaic": 1.0}}
    )
    report.event("train.complete")
    return report


def test_render_html_contains_sections() -> None:
    report = _full_report()
    html = render_html(report, report.to_dict(), report_dir=Path("runs/reports"), embed_previews=False)
    assert "<!DOCTYPE html>" in html
    assert "Кривые обучения" in html
    assert "Hands benchmark" in html
    assert "lossChart" in html
    assert "chart.js" in html
    assert report.run_id in html


def test_write_html_report_creates_files(tmp_path: Path) -> None:
    report = _full_report()
    path = write_html_report(report, tmp_path, embed_previews=False)
    assert path.exists()
    assert (tmp_path / "latest.html").exists()
    content = path.read_text(encoding="utf-8")
    assert "val_top1" in content or "Val top-1" in content


def test_write_html_report_from_json_roundtrip(tmp_path: Path) -> None:
    report = _full_report()
    json_path = tmp_path / "report.json"
    json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False), encoding="utf-8")
    out = write_html_report_from_json(json_path, output=tmp_path / "custom.html", embed_previews=False)
    assert out == tmp_path / "custom.html"
    assert "20260101_120000_train" in out.read_text(encoding="utf-8")


def test_report_from_dict_matches_run_report() -> None:
    report = _full_report()
    restored = report_from_dict(report.to_dict())
    assert restored.run_id == report.run_id
    assert restored.metrics["val_top1"] == 0.91
    assert len(restored.training_curves) == 2


def test_load_report_json(tmp_path: Path) -> None:
    data = {"run_id": "x", "phase": "infer", "started_at": "2026-01-01T00:00:00+00:00"}
    path = tmp_path / "x.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_report_json(path)["phase"] == "infer"
