"""Tests for observability stack integration (Pushgateway, Prometheus, Grafana)."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import yaml

from poker_yolo.monitoring import compute_augmentation_summary, compute_dataset_stats, compute_production_metrics
from poker_yolo.observability import (
    GRAFANA_METRICS,
    OBSERVABILITY_SERVICES,
    collect_export_metrics,
    missing_grafana_metrics,
    prepare_export_metrics,
    push_to_pushgateway,
    pushgateway_url,
    validate_prometheus_exposition,
)
from poker_yolo.reporting import RunReport, render_prometheus


def _sample_train_report() -> RunReport:
    """Report shaped like a successful full pipeline run."""
    report = RunReport(
        run_id="20260101_120000_train",
        phase="train",
        started_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        config_name="poker_cards_smoke",
    )
    report.finished_at = datetime(2026, 1, 1, 12, 30, 0, tzinfo=timezone.utc)
    report.status = "success"

    resources = {
        "train_cpu_avg_pct": 55.0,
        "train_cpu_peak_pct": 80.0,
        "train_ram_avg_mb": 2048.0,
        "train_ram_peak_mb": 2500.0,
        "train_gpu_util_avg_pct": 0.0,
        "train_gpu_mem_peak_mb": 0.0,
        "val_cpu_avg_pct": 30.0,
        "val_ram_avg_mb": 1800.0,
        "val_gpu_util_avg_pct": 0.0,
    }
    report.set_resources(resources)
    report.set_metrics(
        {
            "train_duration_sec": 900.0,
            "val_duration_sec": 60.0,
            "infer_duration_sec": 45.0,
            "train_map50": 0.4,
            "val_map50": 0.35,
            "val_precision": 0.5,
            "val_recall": 0.45,
            "val_f1": 0.47,
            "infer_latency_ms": 120.0,
            "infer_fps": 8.0,
            "infer_images": 31.0,
        }
    )

    aug = compute_augmentation_summary(
        __import__("poker_yolo.augmentations", fromlist=["AugmentationConfig"]).AugmentationConfig(
            enabled=True,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.15,
            cutmix=0.05,
            fliplr=0.5,
            flipud=0.0,
            degrees=10.0,
            translate=0.1,
            scale=0.45,
            shear=3.0,
            perspective=0.0003,
            hsv_h=0.015,
            hsv_s=0.6,
            hsv_v=0.35,
            albumentations={"blur": 0.08},
        ),
        train_images=109,
    )
    dataset_stats = {"train_images": 109, "test_images": 31, "num_classes": 52}
    report.set_augmentations_summary(aug)
    report.set_dataset_stats(dataset_stats)
    report.set_production(
        compute_production_metrics(
            weights_path=None,
            train_duration_sec=900.0,
            val_duration_sec=60.0,
            resource_metrics=resources,
            val_metrics={"map50": 0.35, "precision": 0.5, "recall": 0.45, "f1": 0.47},
            dataset_stats=dataset_stats,
            aug_summary=aug,
            infer_duration_sec=45.0,
            infer_metrics={"infer_latency_ms": 120.0, "infer_fps": 8.0, "infer_images": 31.0},
        )
    )
    report.production["model_size_mb"] = 6.2
    report.metrics["model_size_mb"] = 6.2
    return report


def test_prepare_export_metrics_deduplicates_sanitized_names() -> None:
    report = _sample_train_report()
    report.set_metrics({"train.cpu_avg_pct": 10.0, "train_cpu_avg_pct": 99.0})
    prepared = prepare_export_metrics(report)
    assert prepared["train_cpu_avg_pct"] == 99.0
    assert "train.cpu_avg_pct" not in prepared


def test_validate_prometheus_exposition_rejects_duplicate_type_lines() -> None:
    prom = render_prometheus(_sample_train_report())
    broken = prom + "\n# TYPE poker_yolo_val_map50 gauge\n"
    errors = validate_prometheus_exposition(broken)
    assert any("duplicate TYPE" in err for err in errors)


def test_legacy_duplicate_export_format_is_invalid() -> None:
    """Reproduce pre-fix bug: separate loops over metrics/resources/production."""
    report = _sample_train_report()
    labels = 'phase="train",run_id="r1",status="success"'
    lines: list[str] = []
    for source in (report.metrics, report.resources, report.production):
        for name, value in source.items():
            metric = name.lower().replace("-", "_").replace(".", "_")
            lines.extend(
                [
                    f"# TYPE poker_yolo_{metric} gauge",
                    f"poker_yolo_{metric}{{{labels}}} {float(value)}",
                ]
            )
    legacy = "\n".join(lines) + "\n"
    errors = validate_prometheus_exposition(legacy)
    assert errors, "legacy duplicate export must be rejected before push"


def test_collect_export_metrics_deduplicates_resource_and_production() -> None:
    report = _sample_train_report()
    exported = collect_export_metrics(report)
    assert exported["train_cpu_avg_pct"] == 55.0
    assert exported["val_map50"] == 0.35
    assert exported["pipeline_duration_sec"] == 1005.0


def test_render_prometheus_has_no_duplicate_type_lines() -> None:
    prom = render_prometheus(_sample_train_report())
    type_lines = [line for line in prom.splitlines() if line.startswith("# TYPE poker_yolo_")]
    metric_names = [line.split()[2] for line in type_lines]
    assert len(metric_names) == len(set(metric_names))


def test_render_prometheus_includes_grafana_dashboard_metrics() -> None:
    prom = render_prometheus(_sample_train_report())
    missing = missing_grafana_metrics(prom)
    assert missing == [], f"Missing Grafana metrics: {missing}"


def test_pushgateway_url_uses_stable_instance_group() -> None:
    url = pushgateway_url("http://localhost:9091")
    assert url == "http://localhost:9091/metrics/job/poker_yolo/instance/main"


def test_push_to_pushgateway_posts_exposition_format() -> None:
    received: dict[str, bytes | str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            received["path"] = self.path
            received["body"] = self.rfile.read(length)
            self.send_response(200)
            self.end_headers()

        def log_message(self, format: str, *args) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    try:
        body = render_prometheus(_sample_train_report()).encode("utf-8")
        error = push_to_pushgateway(body, f"http://127.0.0.1:{port}", retries=1)
        assert error is None
        assert received["path"] == "/metrics/job/poker_yolo/instance/main"
        assert b"poker_yolo_val_map50" in received["body"]
        assert b"# TYPE poker_yolo_val_map50 gauge" in received["body"]
    finally:
        server.shutdown()


def test_prometheus_scrape_config_targets_pushgateway(project_root: Path) -> None:
    config = yaml.safe_load((project_root / "observability" / "prometheus.yml").read_text(encoding="utf-8"))
    jobs = {job["job_name"]: job for job in config["scrape_configs"]}
    assert "pushgateway" in jobs
    assert jobs["pushgateway"]["honor_labels"] is True
    assert jobs["pushgateway"]["static_configs"][0]["targets"] == ["pushgateway:9091"]


def test_grafana_dashboard_queries_use_last_over_time(project_root: Path) -> None:
    dashboard = json.loads(
        (project_root / "observability" / "grafana" / "provisioning" / "dashboards" / "poker-yolo.json").read_text(
            encoding="utf-8"
        )
    )
    prom_panels = [
        target["expr"]
        for panel in dashboard["panels"]
        if panel.get("datasource", {}).get("type") == "prometheus"
        for target in panel.get("targets", [])
    ]
    assert prom_panels
    assert all("last_over_time(" in expr for expr in prom_panels)


def test_docker_compose_observability_services(project_root: Path) -> None:
    compose = yaml.safe_load((project_root / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    for name, meta in OBSERVABILITY_SERVICES.items():
        assert name in services, f"Missing docker service: {name}"
        if "port" in meta:
            port_mapping = str(meta["port"])
            published = yaml.dump(services[name].get("ports", []))
            assert port_mapping in published, f"Service {name} should publish port {port_mapping}"


def test_report_server_mounts_reports_directory(project_root: Path) -> None:
    compose = yaml.safe_load((project_root / "docker-compose.yml").read_text(encoding="utf-8"))
    volumes = compose["services"]["report-server"]["volumes"]
    assert any("runs/reports" in v for v in volumes)


def test_grafana_metric_catalog_matches_dashboard_expressions(project_root: Path) -> None:
    dashboard = json.loads(
        (project_root / "observability" / "grafana" / "provisioning" / "dashboards" / "poker-yolo.json").read_text(
            encoding="utf-8"
        )
    )
    exprs = [
        target["expr"]
        for panel in dashboard["panels"]
        if panel.get("datasource", {}).get("type") == "prometheus"
        for target in panel.get("targets", [])
    ]
    for metric in GRAFANA_METRICS:
        if metric in {"poker_yolo_run_duration_seconds", "poker_yolo_run_info"}:
            continue
        assert any(metric in expr for expr in exprs), f"Dashboard missing query for {metric}"


@pytest.mark.integration
def test_live_pushgateway_accepts_exposition() -> None:
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen("http://localhost:9091", timeout=2)
    except (urllib.error.URLError, OSError):
        pytest.skip("Pushgateway is not running on localhost:9091")

    body = render_prometheus(_sample_train_report()).encode("utf-8")
    assert push_to_pushgateway(body, "http://localhost:9091", retries=1) is None
