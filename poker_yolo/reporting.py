from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poker_yolo.logging_config import log_action
from poker_yolo.observability import (
    escape_label_value,
    prepare_export_metrics,
    push_to_pushgateway,
)

logger = logging.getLogger(__name__)

_current_report: RunReport | None = None


@dataclass
class ReportingConfig:
    log_dir: Path
    report_dir: Path
    level: str = "INFO"
    json_logs: bool = True
    console_json: bool = False
    pushgateway_url: str | None = None
    reports_base_url: str = "http://localhost:8088"
    preview_samples: int = 3

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None, project_root: Path) -> ReportingConfig:
        raw = raw or {}
        log_dir = project_root / raw.get("log_dir", "runs/logs")
        report_dir = project_root / raw.get("report_dir", "runs/reports")
        pushgateway = os.environ.get("PROMETHEUS_PUSHGATEWAY_URL", raw.get("pushgateway_url"))
        base_url = os.environ.get("REPORTS_BASE_URL", raw.get("reports_base_url", "http://localhost:8088"))
        return cls(
            log_dir=log_dir.resolve(),
            report_dir=report_dir.resolve(),
            level=str(raw.get("level", "INFO")),
            json_logs=bool(raw.get("json_logs", True)),
            console_json=bool(raw.get("console_json", False)),
            pushgateway_url=pushgateway or None,
            reports_base_url=str(base_url),
            preview_samples=int(raw.get("preview_samples", 3)),
        )


@dataclass
class RunReport:
    run_id: str
    phase: str
    started_at: datetime
    config_name: str
    events: list[dict[str, Any]] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    resources: dict[str, float] = field(default_factory=dict)
    augmentations_summary: dict[str, Any] = field(default_factory=dict)
    dataset_stats: dict[str, Any] = field(default_factory=dict)
    predictions: list[dict[str, Any]] = field(default_factory=list)
    production: dict[str, float] = field(default_factory=dict)
    status: str = "running"
    error: str | None = None
    finished_at: datetime | None = None

    def event(self, action: str, **details: Any) -> None:
        entry = {
            "ts": _utcnow().isoformat(),
            "action": action,
            **details,
        }
        self.events.append(entry)
        log_action(logger, logging.INFO, action, f"{self.phase}: {action}", **details)

    def set_params(self, params: dict[str, Any]) -> None:
        self.params.update(params)

    def set_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics.update({k: float(v) for k, v in metrics.items() if v is not None})

    def set_artifact(self, key: str, path: Path | str) -> None:
        self.artifacts[key] = str(path)

    def set_resources(self, resources: dict[str, float]) -> None:
        self.resources.update({k: float(v) for k, v in resources.items()})

    def set_augmentations_summary(self, summary: dict[str, Any]) -> None:
        self.augmentations_summary.update(summary)

    def set_dataset_stats(self, stats: dict[str, Any]) -> None:
        self.dataset_stats.update(stats)

    def set_predictions(self, samples: list[dict[str, Any]]) -> None:
        self.predictions = samples

    def set_production(self, metrics: dict[str, float]) -> None:
        self.production.update({k: float(v) for k, v in metrics.items() if v is not None})
        self.metrics.update(self.production)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "phase": self.phase,
            "config_name": self.config_name,
            "status": self.status,
            "error": self.error,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_sec": _duration_sec(self) if self.finished_at else None,
            "params": self.params,
            "metrics": self.metrics,
            "resources": self.resources,
            "augmentations_summary": self.augmentations_summary,
            "dataset_stats": self.dataset_stats,
            "predictions": self.predictions,
            "production": self.production,
            "artifacts": self.artifacts,
            "events": self.events,
        }


def start_report(phase: str, config_name: str = "poker_cards") -> RunReport:
    global _current_report
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    _current_report = RunReport(
        run_id=f"{ts}_{phase}",
        phase=phase,
        started_at=_utcnow(),
        config_name=config_name,
    )
    _current_report.event("pipeline.start", phase=phase, run_id=_current_report.run_id)
    return _current_report


def get_report() -> RunReport | None:
    return _current_report


def log_event(action: str, **details: Any) -> None:
    report = get_report()
    if report is not None:
        report.event(action, **details)
    else:
        log_action(logger, logging.INFO, action, action, **details)


def finalize_report(
    reporting: ReportingConfig,
    status: str = "success",
    error: str | None = None,
) -> dict[str, Path]:
    global _current_report
    report = _current_report
    if report is None:
        raise RuntimeError("No active report session. Call start_report() first.")

    report.status = status
    report.error = error
    report.finished_at = _utcnow()
    report.event("pipeline.finish", status=status, error=error)

    reporting.report_dir.mkdir(parents=True, exist_ok=True)
    paths = write_report_files(report, reporting.report_dir)

    if reporting.pushgateway_url:
        _push_prometheus_metrics(report, reporting.pushgateway_url)

    _current_report = None
    log_action(
        logger,
        logging.INFO,
        "report.saved",
        f"Report saved: {paths['json']}",
        json=str(paths["json"]),
        markdown=str(paths["markdown"]),
    )
    return paths


def write_report_files(report: RunReport, report_dir: Path) -> dict[str, Path]:
    payload = report.to_dict()
    json_path = report_dir / f"{report.run_id}.json"
    md_path = report_dir / f"{report.run_id}.md"
    prom_path = report_dir / f"{report.run_id}.prom"
    latest_json = report_dir / "latest.json"
    latest_md = report_dir / "latest.md"
    latest_prom = report_dir / "latest.prom"

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    prom_path.write_text(render_prometheus(report), encoding="utf-8")

    for src, dst in (
        (json_path, latest_json),
        (md_path, latest_md),
        (prom_path, latest_prom),
    ):
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    return {"json": json_path, "markdown": md_path, "prometheus": prom_path}


def render_markdown(report: RunReport) -> str:
    duration = _duration_sec(report)
    lines = [
        f"# Poker YOLO Report — {report.phase}",
        "",
        f"- **Run ID:** `{report.run_id}`",
        f"- **Status:** {report.status}",
        f"- **Started:** {report.started_at.isoformat()}",
        f"- **Finished:** {report.finished_at.isoformat() if report.finished_at else '—'}",
        f"- **Duration:** {duration:.1f}s",
        "",
    ]

    if report.error:
        lines.extend(["## Error", "", f"```\n{report.error}\n```", ""])

    if report.metrics:
        lines.extend(["## Metrics", "", "| Metric | Value |", "|--------|-------|"])
        for key, value in sorted(report.metrics.items()):
            if isinstance(value, float):
                lines.append(f"| `{key}` | {value:.4f} |")
            else:
                lines.append(f"| `{key}` | {value} |")
        lines.append("")

    if report.resources:
        lines.extend(["## Resource Usage (CPU / RAM / GPU)", "", "| Resource | Value |", "|----------|-------|"])
        for key, value in sorted(report.resources.items()):
            lines.append(f"| `{key}` | {value:.2f} |")
        lines.append("")

    if report.augmentations_summary:
        lines.extend(["## Augmentation Statistics", ""])
        ratio = report.augmentations_summary.get("synthetic_to_real_ratio")
        real = report.augmentations_summary.get("train_images_real")
        est = report.augmentations_summary.get("estimated_augmented_views_per_epoch")
        if ratio is not None:
            lines.append(f"- **Synthetic-to-real ratio:** {ratio}")
        if real is not None:
            lines.append(f"- **Real train images:** {real}")
        if est is not None:
            lines.append(f"- **Estimated augmented views/epoch:** {est}")
        lines.append("")
        yolo_probs = report.augmentations_summary.get("yolo_probabilities", {})
        if yolo_probs:
            lines.extend(["### YOLO augmentations (probability)", "", "| Transform | p |", "|-----------|---|"])
            for name, prob in sorted(yolo_probs.items()):
                lines.append(f"| `{name}` | {prob} |")
            lines.append("")
        alb_probs = report.augmentations_summary.get("albumentations_probabilities", {})
        if alb_probs:
            lines.extend(["### Albumentations (probability per sample)", "", "| Transform | p |", "|-----------|---|"])
            for name, prob in sorted(alb_probs.items()):
                lines.append(f"| `{name}` | {prob} |")
            lines.append("")

    if report.production:
        lines.extend(["## Production KPIs", "", "| KPI | Value |", "|-----|-------|"])
        for key, value in sorted(report.production.items()):
            lines.append(f"| `{key}` | {value:.4f} |")
        lines.append("")

    if report.predictions:
        lines.extend(["## Sample Predictions", ""])
        for sample in report.predictions:
            url = sample.get("preview_url", "")
            src = sample.get("source_image", "")
            top = sample.get("top_classes", [])
            classes_str = ", ".join(f"{t['class_name']} ({t['confidence']:.2f})" for t in top) or "—"
            lines.append(f"### Sample {sample.get('index', 0)}")
            lines.append(f"- **Source:** `{src}`")
            lines.append(f"- **Detections:** {sample.get('detections_count', 0)}")
            lines.append(f"- **Top classes:** {classes_str}")
            if url:
                lines.append(f"- **Preview:** [{url}]({url})")
                lines.append(f"\n![sample_{sample.get('index', 0)}]({url})\n")
        lines.append("")

    if report.dataset_stats:
        lines.extend(["## Dataset", ""])
        for key in ("train_images", "test_images", "num_classes"):
            if key in report.dataset_stats:
                lines.append(f"- **{key}:** {report.dataset_stats[key]}")
        lines.append("")

    if report.artifacts:
        lines.extend(["## Artifacts", ""])
        for key, path in sorted(report.artifacts.items()):
            lines.append(f"- **{key}:** `{path}`")
        lines.append("")

    if report.params:
        lines.extend(["## Parameters", "", "| Parameter | Value |", "|-----------|-------|"])
        for key, value in sorted(report.params.items()):
            lines.append(f"| `{key}` | {value} |")
        lines.append("")

    lines.extend(["## Event Timeline", ""])
    for event in report.events:
        action = event.get("action", "unknown")
        ts = event.get("ts", "")
        extra = {k: v for k, v in event.items() if k not in {"action", "ts"}}
        detail = f" — `{extra}`" if extra else ""
        lines.append(f"- `{ts}` **{action}**{detail}")

    lines.extend(
        [
            "",
            "---",
            "*Prometheus text: `runs/reports/latest.prom` — importable in Grafana via Prometheus datasource.*",
        ]
    )
    return "\n".join(lines)


def render_prometheus(report: RunReport) -> str:
    """Prometheus exposition format for Grafana dashboards."""
    labels = (
        f'phase="{escape_label_value(report.phase)}"'
        f',run_id="{escape_label_value(report.run_id)}"'
        f',status="{escape_label_value(report.status)}"'
    )
    config_label = escape_label_value(report.config_name)
    lines = [
        "# HELP poker_yolo_run_duration_seconds Pipeline run duration in seconds.",
        "# TYPE poker_yolo_run_duration_seconds gauge",
        f"poker_yolo_run_duration_seconds{{{labels}}} {_duration_sec(report)}",
        "# HELP poker_yolo_run_info Static info metric (always 1 for latest run).",
        "# TYPE poker_yolo_run_info gauge",
        f'poker_yolo_run_info{{{labels},config="{config_label}"}} 1',
    ]

    for metric, value in sorted(prepare_export_metrics(report).items()):
        lines.extend(
            [
                f"# HELP poker_yolo_{metric} Reported {metric} metric.",
                f"# TYPE poker_yolo_{metric} gauge",
                f"poker_yolo_{metric}{{{labels}}} {value}",
            ]
        )
    return "\n".join(lines) + "\n"


def _push_prometheus_metrics(report: RunReport, pushgateway_url_base: str) -> None:
    body = render_prometheus(report).encode("utf-8")
    error = push_to_pushgateway(body, pushgateway_url_base)
    if error:
        logger.warning("Failed to push metrics to Pushgateway: %s", error)


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _duration_sec(report: RunReport) -> float:
    if report.finished_at is None:
        return 0.0
    return (report.finished_at - report.started_at).total_seconds()


