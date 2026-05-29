#!/usr/bin/env python3
"""Regenerate runs/reports/grafana/*.json and latest.prom from existing weights."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from poker_yolo.config import Config
from poker_yolo.grafana_artifacts import write_grafana_artifacts
from poker_yolo.observability import push_to_pushgateway
from poker_yolo.predictions import analyze_hands_benchmark
from poker_yolo.reporting import RunReport, render_prometheus
from poker_yolo.training_curves import parse_training_results_csv, results_csv_path


def _pushgateway_url() -> str:
    return os.environ.get(
        "PROMETHEUS_PUSHGATEWAY_URL",
        os.environ.get("PUSHGATEWAY_URL", "http://localhost:9091"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/classify/runs/train/poker_cards_smoke/weights/best.pt"),
    )
    parser.add_argument("--report-dir", type=Path, default=Path("runs/reports"))
    parser.add_argument(
        "--from-latest-json",
        action="store_true",
        help="Merge metrics/resources from runs/reports/latest.json (full pipeline snapshot).",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Only refresh training curves / Prometheus (no second YOLO predict).",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push latest.prom to Pushgateway (PROMETHEUS_PUSHGATEWAY_URL or localhost:9091).",
    )
    args = parser.parse_args()

    if not args.weights.is_file():
        raise SystemExit(f"Weights not found: {args.weights}")

    config = Config.from_yaml(args.config)
    curves = parse_training_results_csv(results_csv_path(args.weights))
    report = RunReport(
        run_id="backfill_grafana",
        phase="train",
        started_at=datetime.now(timezone.utc),
        config_name=config.name,
    )
    report.status = "success"
    report.set_training_curves(curves)
    report.set_artifact("weights", args.weights)
    report.set_artifact("training_results_csv", results_csv_path(args.weights))

    if not args.skip_benchmark:
        analysis = analyze_hands_benchmark(
            config,
            args.weights,
            args.report_dir,
            n_samples=config.reporting.preview_samples,
            reports_base_url=config.reporting.reports_base_url,
        )
        report.set_hands_benchmark_stats(
            analysis.aggregate,
            analysis.predicted_class_counts,
            outcomes_by_hand=analysis.outcomes_by_hand,
            confusion=analysis.confusion,
        )
        report.set_predictions(analysis.samples)

    if args.from_latest_json:
        latest_path = args.report_dir / "latest.json"
        if latest_path.is_file():
            snapshot = json.loads(latest_path.read_text(encoding="utf-8"))
            report.metrics.update(
                {k: float(v) for k, v in (snapshot.get("metrics") or {}).items() if v is not None}
            )
            report.resources.update(
                {k: float(v) for k, v in (snapshot.get("resources") or {}).items() if v is not None}
            )
            report.production.update(
                {k: float(v) for k, v in (snapshot.get("production") or {}).items() if v is not None}
            )
            report.benchmark_outcomes_by_hand = snapshot.get("benchmark_outcomes_by_hand") or {}
            report.benchmark_confusion = snapshot.get("benchmark_confusion") or {}
            report.predictions = snapshot.get("predictions") or []
            if snapshot.get("benchmark_class_counts"):
                report.benchmark_class_counts = snapshot["benchmark_class_counts"]

    write_grafana_artifacts(
        args.report_dir,
        outcomes_by_hand=report.benchmark_outcomes_by_hand,
        confusion=report.benchmark_confusion,
        training_curves=curves,
    )
    prom = render_prometheus(report)
    (args.report_dir / "latest.prom").write_text(prom, encoding="utf-8")
    print(f"Wrote Grafana JSON under {args.report_dir / 'grafana'}")
    print(f"Updated {args.report_dir / 'latest.prom'}")
    if args.push:
        pg_url = _pushgateway_url()
        err = push_to_pushgateway(prom.encode("utf-8"), pg_url)
        if err:
            raise SystemExit(f"Pushgateway push failed: {err}")
        print(f"Pushed metrics to {pg_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
