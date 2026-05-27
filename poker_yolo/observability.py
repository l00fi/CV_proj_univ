"""Prometheus / Pushgateway / Grafana integration helpers."""

from __future__ import annotations

import logging
import math
import re
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from poker_yolo.reporting import RunReport

logger = logging.getLogger(__name__)

PUSHGATEWAY_JOB = "poker_yolo"
PUSHGATEWAY_INSTANCE = "main"
_METRIC_NAME_RE = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")

# Metric names expected by observability/grafana/provisioning/dashboards/poker-yolo.json
GRAFANA_METRICS: tuple[str, ...] = (
    "poker_yolo_train_cpu_avg_pct",
    "poker_yolo_val_cpu_avg_pct",
    "poker_yolo_train_gpu_util_avg_pct",
    "poker_yolo_val_gpu_util_avg_pct",
    "poker_yolo_train_ram_avg_mb",
    "poker_yolo_val_ram_avg_mb",
    "poker_yolo_train_gpu_mem_peak_mb",
    "poker_yolo_val_map50",
    "poker_yolo_train_duration_sec",
    "poker_yolo_val_duration_sec",
    "poker_yolo_pipeline_duration_sec",
    "poker_yolo_synthetic_to_real_ratio",
    "poker_yolo_estimated_augmented_views_per_epoch",
    "poker_yolo_model_size_mb",
    "poker_yolo_val_precision",
    "poker_yolo_val_recall",
    "poker_yolo_val_f1",
    "poker_yolo_train_map50",
    "poker_yolo_run_duration_seconds",
    "poker_yolo_run_info",
)

OBSERVABILITY_SERVICES: dict[str, dict[str, str | int]] = {
    "mlflow": {"port": 5000},
    "pushgateway": {"port": 9091, "profile": "observability"},
    "prometheus": {"port": 9090, "profile": "observability"},
    "grafana": {"port": 3001, "profile": "observability"},
    "report-server": {"port": 8088, "profile": "observability"},
}


def collect_export_metrics(report: RunReport) -> dict[str, float]:
    """Merge report fields without duplicates (resources are also copied into production/metrics)."""
    merged: dict[str, float] = {}
    merged.update(report.resources)
    merged.update(report.production)
    merged.update(report.metrics)
    return merged


def sanitize_prometheus_name(name: str) -> str:
    """Normalize a report key into a valid Prometheus metric suffix."""
    cleaned = name.lower()
    for ch in "-./() ":
        cleaned = cleaned.replace(ch, "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("_")
    if cleaned and cleaned[0].isdigit():
        cleaned = f"m_{cleaned}"
    return cleaned


def escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def prepare_export_metrics(report: RunReport) -> dict[str, float]:
    """Merge, sanitize, deduplicate and drop non-finite values for Prometheus export."""
    prepared: dict[str, float] = {}
    for source in (report.resources, report.production, report.metrics):
        for name, value in sorted(source.items()):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric):
                continue
            metric = sanitize_prometheus_name(name)
            if not metric or not _METRIC_NAME_RE.match(metric):
                continue
            prepared[metric] = numeric
    return prepared


def validate_prometheus_exposition(text: str) -> list[str]:
    """Return validation errors (empty list means Pushgateway-safe exposition)."""
    errors: list[str] = []
    seen_types: set[str] = set()
    for line in text.splitlines():
        if not line.startswith("# TYPE "):
            continue
        parts = line.split()
        if len(parts) < 4:
            errors.append(f"malformed TYPE line: {line}")
            continue
        metric = parts[2]
        if metric in seen_types:
            errors.append(f"duplicate TYPE line for {metric}")
        seen_types.add(metric)
    return errors


def pushgateway_url(base_url: str) -> str:
    """Build Pushgateway URL that replaces the latest pipeline snapshot."""
    return (
        f"{base_url.rstrip('/')}/metrics/job/{PUSHGATEWAY_JOB}/instance/{PUSHGATEWAY_INSTANCE}"
    )


def push_to_pushgateway(
    body: bytes,
    base_url: str,
    *,
    retries: int = 3,
    retry_delay_sec: float = 1.0,
    timeout_sec: float = 10.0,
) -> str | None:
    """Push exposition text. Returns None on success, error message on failure."""
    text = body.decode("utf-8")
    validation_errors = validate_prometheus_exposition(text)
    if validation_errors:
        message = "; ".join(validation_errors)
        logger.error("Invalid Prometheus exposition, skipping push: %s", message)
        return message

    url = pushgateway_url(base_url)
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "text/plain; version=0.0.4")

    last_error: str | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                if response.status >= 400:
                    raise urllib.error.URLError(f"HTTP {response.status}")
            logger.info("Pushed metrics to Pushgateway: %s", url)
            return None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace").strip()
            last_error = f"HTTP Error {exc.code}: {detail or exc.reason}"
            logger.warning(
                "Pushgateway push failed (attempt %d/%d): %s",
                attempt,
                retries,
                last_error,
            )
        except urllib.error.URLError as exc:
            last_error = str(exc.reason or exc)
            logger.warning(
                "Pushgateway push failed (attempt %d/%d): %s",
                attempt,
                retries,
                last_error,
            )

        if attempt < retries:
            time.sleep(retry_delay_sec)

    logger.error("Failed to push metrics to Pushgateway after %d attempts: %s", retries, last_error)
    return last_error or "unknown Pushgateway error"


def missing_grafana_metrics(prometheus_text: str) -> list[str]:
    return [name for name in GRAFANA_METRICS if name not in prometheus_text]
