"""Shared pipeline helpers (infer source checks, report enrichment)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from poker_yolo.config import Config, ReportingConfig
from poker_yolo.infer_source import resolve_infer_source
from poker_yolo.monitoring import (
    compute_augmentation_summary,
    compute_dataset_stats,
    compute_production_metrics,
)
from poker_yolo.predictions import analyze_hands_benchmark_from_infer_results
from poker_yolo.reporting import finalize_report, get_report
from poker_yolo.training_curves import parse_training_results_csv, results_csv_path

logger = logging.getLogger(__name__)


def validate_infer_source(source: Path, reporting: ReportingConfig) -> int | None:
    """Return exit code 1 after failed report when source is missing or empty."""
    if not source.exists():
        logger.error("Inference source not found: %s", source)
        finalize_report(reporting, status="failed", error=f"Inference source not found: {source}")
        return 1
    resolved = resolve_infer_source(source)
    if isinstance(resolved, list) and not resolved:
        logger.error("No images found under inference source: %s", source)
        finalize_report(
            reporting,
            status="failed",
            error=f"No images found under inference source: {source}",
        )
        return 1
    return None


def enrich_report_after_train(
    config: Config,
    weights: Path,
    train_duration: float,
    val_metrics: dict[str, float],
    val_duration: float,
    *,
    pred_dir: Path | None = None,
    infer_duration: float = 0.0,
    infer_results: list[Any] | None = None,
) -> None:
    report = get_report()
    if report is None:
        return

    dataset_stats = compute_dataset_stats(config.dataset_root, config.data_yaml)
    benchmark_stats = compute_dataset_stats(config.val_dataset_root, config.val_data_yaml)
    dataset_stats["benchmark_test_images"] = benchmark_stats.get("test_images", 0)
    dataset_stats["benchmark_source"] = "dataset/test/images"
    aug_summary = compute_augmentation_summary(
        config.augmentations,
        int(dataset_stats.get("train_images", 0)),
        task=config.task,
    )
    report.set_dataset_stats(dataset_stats)
    report.set_augmentations_summary(aug_summary)

    curves_csv = results_csv_path(weights)
    curves = parse_training_results_csv(curves_csv)
    if curves:
        report.set_training_curves(curves)
        report.set_artifact("training_results_csv", curves_csv)

    infer_metrics = {k: v for k, v in report.metrics.items() if k.startswith("infer_")}
    report.set_production(
        compute_production_metrics(
            weights_path=weights,
            train_duration_sec=train_duration,
            val_duration_sec=val_duration,
            resource_metrics=dict(report.resources),
            val_metrics=val_metrics,
            dataset_stats=dataset_stats,
            aug_summary=aug_summary,
            infer_duration_sec=infer_duration,
            infer_metrics=infer_metrics or None,
        )
    )

    if pred_dir is not None:
        report.set_artifact("inference_output", pred_dir)

    if infer_results:
        analysis = analyze_hands_benchmark_from_infer_results(
            config,
            config.reporting.report_dir,
            infer_results,
            n_samples=config.reporting.preview_samples,
            reports_base_url=config.reporting.reports_base_url,
        )
        if analysis is None:
            logger.warning(
                "Hands benchmark skipped: infer results do not cover %s",
                config.infer_source,
            )
        else:
            report.set_hands_benchmark_stats(
                analysis.aggregate,
                analysis.predicted_class_counts,
                outcomes_by_hand=analysis.outcomes_by_hand,
                confusion=analysis.confusion,
            )
            report.set_predictions(analysis.samples)
