"""Tests for run reports and Prometheus export."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from poker_yolo.reporting import (
    ReportingConfig,
    RunReport,
    finalize_report,
    log_event,
    render_markdown,
    render_prometheus,
    start_report,
    write_report_files,
)


@pytest.fixture(autouse=True)
def reset_report_state():
    import poker_yolo.reporting as reporting

    reporting._current_report = None
    yield
    reporting._current_report = None


def test_run_report_tracks_events() -> None:
    report = RunReport(
        run_id="test_run",
        phase="validate",
        started_at=datetime.now(tz=timezone.utc),
        config_name="test",
    )
    report.event("validate.start", weights="best.pt")
    assert report.events[-1]["action"] == "validate.start"


def test_write_report_files_creates_json_md_and_prom(tmp_path: Path) -> None:
    report = RunReport(
        run_id="20260101_120000_validate",
        phase="validate",
        started_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        config_name="poker_cards",
    )
    report.finished_at = datetime(2026, 1, 1, 12, 5, 0, tzinfo=timezone.utc)
    report.status = "success"
    report.set_metrics({"val_map50": 0.88, "val_f1": 0.81})
    report.set_resources({"train_cpu_avg_pct": 72.5, "val_cpu_avg_pct": 35.0})
    report.set_augmentations_summary(
        {"synthetic_to_real_ratio": 2.1, "train_images_real": 109, "yolo_probabilities": {"mosaic": 1.0}}
    )
    report.set_predictions([{"index": 0, "preview_url": "http://localhost:8088/preview/sample_0.jpg", "detections_count": 5, "top_classes": []}])
    report.event("validate.complete", map50=0.88)

    paths = write_report_files(report, tmp_path)

    assert paths["json"].exists()
    assert paths["markdown"].exists()
    assert paths["prometheus"].exists()
    assert (tmp_path / "latest.json").exists()
    assert (tmp_path / "latest.md").exists()

    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["phase"] == "validate"
    assert payload["metrics"]["val_map50"] == 0.88

    prom = paths["prometheus"].read_text(encoding="utf-8")
    assert "poker_yolo_val_map50" in prom
    assert "poker_yolo_run_duration_seconds" in prom

    md = paths["markdown"].read_text(encoding="utf-8")
    assert "val_map50" in md
    assert "validate.complete" in md
    assert "Resource Usage" in md
    assert "Augmentation Statistics" in md
    assert "Sample Predictions" in md


def test_render_prometheus_sanitizes_metric_names() -> None:
    report = RunReport(
        run_id="r1",
        phase="infer",
        started_at=datetime.now(tz=timezone.utc),
        config_name="test",
    )
    report.finished_at = datetime.now(tz=timezone.utc)
    report.status = "success"
    report.set_metrics({"infer_latency_ms": 42.0})

    prom = render_prometheus(report)
    assert "poker_yolo_infer_latency_ms" in prom


def test_start_and_finalize_report(tmp_path: Path) -> None:
    reporting = ReportingConfig(
        log_dir=tmp_path / "logs",
        report_dir=tmp_path / "reports",
    )
    report = start_report("train", config_name="poker_cards")
    report.set_metrics({"val_map50": 0.75})
    report.set_artifact("weights", tmp_path / "best.pt")

    paths = finalize_report(reporting, status="success")

    assert paths["json"].exists()
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert payload["phase"] == "train"


def test_log_event_without_active_report(caplog) -> None:
    log_event("orphan.event", detail="x")
    assert "orphan.event" in caplog.text


def test_pushgateway_called_when_configured(tmp_path: Path, mocker) -> None:
    reporting = ReportingConfig(
        log_dir=tmp_path / "logs",
        report_dir=tmp_path / "reports",
        pushgateway_url="http://pushgateway:9091",
    )
    start_report("validate")
    mock_urlopen = mocker.patch("poker_yolo.reporting.urllib.request.urlopen")
    mock_urlopen.return_value.__enter__.return_value.status = 200

    finalize_report(reporting, status="success")

    mock_urlopen.assert_called_once()
    request = mock_urlopen.call_args[0][0]
    assert request.full_url.startswith("http://pushgateway:9091/metrics/job/poker_yolo/")


def test_finalize_report_raises_without_session() -> None:
    reporting = ReportingConfig(log_dir=Path("runs/logs"), report_dir=Path("runs/reports"))
    with pytest.raises(RuntimeError, match="No active report"):
        finalize_report(reporting)
