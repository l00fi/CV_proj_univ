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
            "train_top1": 0.4,
            "val_top1": 0.35,
            "val_top5": 0.8,
            "val_loss": 0.45,
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
            val_metrics={"top1": 0.35, "top5": 0.8, "loss": 0.45},
            dataset_stats=dataset_stats,
            aug_summary=aug,
            infer_duration_sec=45.0,
            infer_metrics={"infer_latency_ms": 120.0, "infer_fps": 8.0, "infer_images": 31.0},
        )
    )
    report.production["model_size_mb"] = 6.2
    report.metrics["model_size_mb"] = 6.2
    report.set_hands_benchmark_stats(
        {
            "hands_benchmark_images": 140.0,
            "hands_benchmark_images_with_predictions": 120.0,
            "hands_benchmark_top1_conf_avg": 0.72,
            "hands_benchmark_unique_classes": 8.0,
            "hands_benchmark_prediction_rate": 0.86,
            "hands_benchmark_accuracy": 0.55,
            "hands_benchmark_evaluated_images": 100.0,
            "hands_benchmark_correct_total": 55.0,
            "hands_benchmark_incorrect_total": 45.0,
        },
        {"pair": 20, "flush": 15},
        outcomes_by_hand={"pair": {"correct": 12, "incorrect": 8}, "flush": {"correct": 10, "incorrect": 5}},
        confusion={"pair": {"pair": 12, "flush": 8}, "flush": {"flush": 10, "pair": 5}},
    )
    report.set_training_curves(
        [
            {"epoch": 1, "train_loss": 1.94, "val_loss": 1.22, "top1": 0.59, "top5": 0.92},
            {"epoch": 2, "train_loss": 1.41, "val_loss": 0.95, "top1": 0.66, "top5": 0.96},
        ]
    )
    report.set_predictions(
        [
            {
                "index": 0,
                "predictions_count": 1,
                "top_classes": [{"class_name": "AS", "confidence": 0.91}],
            },
            {
                "index": 1,
                "predictions_count": 1,
                "top_classes": [{"class_name": "KH", "confidence": 0.88}],
            },
        ]
    )
    return report


def test_prepare_export_metrics_deduplicates_sanitized_names() -> None:
    report = _sample_train_report()
    report.set_metrics({"train.cpu_avg_pct": 10.0, "train_cpu_avg_pct": 99.0})
    prepared = prepare_export_metrics(report)
    assert prepared["train_cpu_avg_pct"] == 99.0
    assert "train.cpu_avg_pct" not in prepared


def test_validate_prometheus_exposition_rejects_duplicate_type_lines() -> None:
    prom = render_prometheus(_sample_train_report())
    broken = prom + "\n# TYPE poker_yolo_val_top1 gauge\n"
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


def test_render_prometheus_includes_grafana_dashboard_metrics() -> None:
    prom = render_prometheus(_sample_train_report())
    missing = missing_grafana_metrics(prom)
    assert missing == [], f"Missing Grafana metrics: {missing}"


def test_render_prometheus_preview_metrics_have_single_type_block() -> None:
    prom = render_prometheus(_sample_train_report())
    assert prom.count("# TYPE poker_yolo_preview_sample_predictions gauge") == 1
    assert prom.count("# TYPE poker_yolo_hands_benchmark_outcome gauge") == 1
    assert prom.count("hand_index=") >= 10
    assert prom.count("poker_yolo_hands_confusion") >= 100
    assert "poker_yolo_train_epoch_metric" in prom
    assert 'sample_index="0"' in prom


def test_pushgateway_url_uses_stable_instance_group() -> None:
    url = pushgateway_url("http://localhost:9091")
    assert url == "http://localhost:9091/metrics/job/poker_yolo/instance/main"


def test_push_to_pushgateway_posts_exposition_format() -> None:
    received: dict[str, bytes | str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_DELETE(self) -> None:
            received["deleted"] = self.path
            self.send_response(200)
            self.end_headers()

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
        assert received.get("deleted") == "/metrics/job/poker_yolo/instance/main"
        assert received["path"] == "/metrics/job/poker_yolo/instance/main"
        assert b"poker_yolo_val_top1" in received["body"]
        assert b"# TYPE poker_yolo_val_top1 gauge" in received["body"]
    finally:
        server.shutdown()


def _grafana_dashboard_panels(project_root: Path) -> list[dict]:
    dash_dir = project_root / "observability" / "grafana" / "provisioning" / "dashboards"
    panels: list[dict] = []
    for path in sorted(dash_dir.glob("poker-yolo*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        panels.extend(doc["panels"])
    return panels


def test_grafana_prometheus_targets_use_instant_queries(project_root: Path) -> None:
    prom_targets = [
        target
        for panel in _grafana_dashboard_panels(project_root)
        for target in panel.get("targets", [])
        if "expr" in target
    ]
    assert prom_targets
    for target in prom_targets:
        assert target.get("instant") is True, f"Missing instant query on {target['expr'][:60]}"
        assert target.get("range") is False


def test_prometheus_scrape_config_targets_pushgateway(project_root: Path) -> None:
    config = yaml.safe_load((project_root / "observability" / "prometheus.yml").read_text(encoding="utf-8"))
    jobs = {job["job_name"]: job for job in config["scrape_configs"]}
    assert "pushgateway" in jobs
    assert jobs["pushgateway"]["honor_labels"] is True
    assert jobs["pushgateway"]["static_configs"][0]["targets"] == ["pushgateway:9091"]


def test_grafana_infinity_panels_use_reports_datasource_and_array_json(project_root: Path) -> None:
    infinity_panels = {17, 101, 102, 103}
    for panel in _grafana_dashboard_panels(project_root):
        if panel.get("id") not in infinity_panels:
            continue
        assert panel.get("datasource", {}).get("uid") == "reports", f"Panel {panel['id']} must use Reports DS"
        target = panel["targets"][0]
        url = target.get("url", "")
        assert "report-server:8088/grafana/" in url
        if panel["id"] in {17, 103}:
            assert "$" in target.get("root_selector", ""), "benchmark panels need JSONata row selector"
        if panel["id"] in {101, 102}:
            assert panel.get("options", {}).get("xField") == "epoch"
            assert target.get("format") == "timeseries"
            col_names = {c.get("selector") for c in target.get("columns", [])}
            assert {"epoch", "time"}.issubset(col_names)


def test_grafana_dashboard_queries_use_last_over_time(project_root: Path) -> None:
    prom_exprs = [
        target["expr"]
        for panel in _grafana_dashboard_panels(project_root)
        if panel.get("datasource", {}).get("type") == "prometheus"
        for target in panel.get("targets", [])
        if "expr" in target
    ]
    assert prom_exprs
    assert all("last_over_time(" in expr for expr in prom_exprs)


def test_docker_compose_observability_services(project_root: Path) -> None:
    compose = yaml.safe_load((project_root / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    for name, meta in OBSERVABILITY_SERVICES.items():
        assert name in services, f"Missing docker service: {name}"
        if "port_env" in meta:
            port_var = str(meta["port_env"])
            published = yaml.dump(services[name].get("ports", []))
            assert port_var in published, f"Service {name} should map host port via ${port_var}"


def test_report_server_mounts_reports_directory(project_root: Path) -> None:
    compose = yaml.safe_load((project_root / "docker-compose.yml").read_text(encoding="utf-8"))
    volumes = compose["services"]["report-server"]["volumes"]
    assert any("runs/reports" in v for v in volumes)


def test_docker_compose_poker_yolo_uses_env_file(project_root: Path) -> None:
    compose = yaml.safe_load((project_root / "docker-compose.yml").read_text(encoding="utf-8"))
    svc = compose["services"]["poker-yolo"]
    assert svc.get("gpus") == "all"
    assert svc.get("env_file") == [".env"]
    env = svc.get("environment", {})
    assert env.get("NVIDIA_VISIBLE_DEVICES") == "${NVIDIA_VISIBLE_DEVICES}"
    assert env.get("MLFLOW_TRACKING_URI") == "${MLFLOW_TRACKING_URI}"
    assert env.get("REQUIRE_CUDA") == "${REQUIRE_CUDA}"


def test_env_example_documents_required_keys(project_root: Path) -> None:
    example = (project_root / ".env.example").read_text(encoding="utf-8")
    for key in (
        "KAGGLE_USERNAME",
        "KAGGLE_KEY",
        "MLFLOW_TRACKING_URI",
        "NVIDIA_VISIBLE_DEVICES",
        "REQUIRE_CUDA",
    ):
        assert key in example


def test_grafana_metric_catalog_matches_dashboard_expressions(project_root: Path) -> None:
    exprs = [
        target["expr"]
        for panel in _grafana_dashboard_panels(project_root)
        if panel.get("datasource", {}).get("type") == "prometheus"
        for target in panel.get("targets", [])
        if "expr" in target
    ]
    for metric in GRAFANA_METRICS:
        if metric in {"poker_yolo_run_duration_seconds", "poker_yolo_run_info"}:
            continue
        assert any(metric in expr for expr in exprs), f"Dashboards missing query for {metric}"


def test_grafana_training_curves_and_inference_dashboard_files_exist(project_root: Path) -> None:
    dash_dir = project_root / "observability" / "grafana" / "provisioning" / "dashboards"
    training = json.loads((dash_dir / "poker-yolo-training.json").read_text(encoding="utf-8"))
    curves = json.loads((dash_dir / "poker-yolo-curves.json").read_text(encoding="utf-8"))
    inference = json.loads((dash_dir / "poker-yolo-inference.json").read_text(encoding="utf-8"))
    assert training["uid"] == "poker-yolo-training"
    assert curves["uid"] == "poker-yolo-curves"
    assert inference["uid"] == "poker-yolo-inference"
    assert {p["id"] for p in training["panels"]} == {100, 20, 21, 1, 2, 3}
    assert {p["id"] for p in curves["panels"]} == {100, 101, 102}
    assert {p["id"] for p in inference["panels"]} == {5, 10, 11, 12, 18, 19, 16, 17, 103}


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
