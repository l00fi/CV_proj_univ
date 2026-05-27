from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from poker_yolo.config import Config
from poker_yolo.kaggle_dataset import ensure_dataset_if_needed
from poker_yolo.infer import run_inference
from poker_yolo.logging_config import setup_logging
from poker_yolo.monitoring import (
    compute_augmentation_summary,
    compute_dataset_stats,
    compute_production_metrics,
)
from poker_yolo.predictions import save_sample_predictions
from poker_yolo.reporting import finalize_report, get_report, start_report
from poker_yolo.train import run_training
from poker_yolo.validate import run_validation

logger = logging.getLogger("poker_yolo")


def _default_config() -> Path:
    candidates = [
        Path("configs/default.yaml"),
        Path("/app/configs/default.yaml"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _default_weights(config: Config) -> Path:
    candidates = [
        Path(config.project) / config.name / "weights" / "best.pt",
        Path("runs/detect") / config.project / config.name / "weights" / "best.pt",
        Path("/app/runs/train/poker_cards/weights/best.pt"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="poker-yolo",
        description="YOLOv8 pipeline for poker card detection",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config (default: configs/default.yaml)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser(
        "train",
        help="Full pipeline: train → validate → infer → report",
    )
    train_parser.add_argument(
        "--skip-infer",
        action="store_true",
        help="Skip inference step (train + validate only)",
    )
    train_parser.add_argument(
        "--infer-source",
        type=Path,
        default=None,
        help="Images for post-train inference (default: infer.source in config)",
    )
    train_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving annotated inference images",
    )

    val_parser = sub.add_parser("validate", help="Validate trained model only")
    val_parser.add_argument("--weights", type=Path, default=None, help="Path to .pt weights")

    infer_parser = sub.add_parser("infer", help="Run inference only")
    infer_parser.add_argument("--weights", type=Path, default=None, help="Path to .pt weights")
    infer_parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Image file, directory, or glob pattern",
    )
    infer_parser.add_argument("--no-save", action="store_true", help="Skip saving annotated images")

    return parser


def _populate_report_params(report, config: Config) -> None:
    report.set_params(
        {
            "model_weights": config.model_weights,
            "imgsz": config.imgsz,
            "epochs": config.epochs,
            "batch": config.batch,
            "device": config.device,
            "data_yaml": str(config.data_yaml),
            "aug_enabled": config.augmentations.enabled,
            "mosaic": config.augmentations.mosaic,
            "mixup": config.augmentations.mixup,
            "infer_source": str(config.infer_source),
        }
    )


def _enrich_report_after_train(
    config: Config,
    weights: Path,
    train_duration: float,
    val_metrics: dict[str, float],
    val_duration: float,
    pred_dir: Path | None = None,
    infer_duration: float = 0.0,
) -> None:
    report = get_report()
    if report is None:
        return

    dataset_stats = compute_dataset_stats(config.dataset_root, config.data_yaml)
    aug_summary = compute_augmentation_summary(
        config.augmentations,
        int(dataset_stats.get("train_images", 0)),
    )
    report.set_dataset_stats(dataset_stats)
    report.set_augmentations_summary(aug_summary)

    samples = save_sample_predictions(
        config,
        weights,
        config.reporting.report_dir,
        n_samples=config.reporting.preview_samples,
        reports_base_url=config.reporting.reports_base_url,
    )
    report.set_predictions(samples)
    if samples:
        report.set_artifact("predictions_manifest", config.reporting.report_dir / "preview" / "manifest.json")

    infer_metrics = {k: v for k, v in report.metrics.items() if k.startswith("infer_")}

    production = compute_production_metrics(
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
    report.set_production(production)

    if pred_dir is not None:
        report.set_artifact("inference_output", pred_dir)


def _print_results_summary(
    config: Config,
    paths: dict[str, Path],
    weights: Path,
    pred_dir: Path | None,
) -> None:
    report = get_report()
    base = config.reporting.reports_base_url.rstrip("/")
    lines = [
        "",
        "=" * 60,
        "Pipeline complete — where to view results",
        "=" * 60,
        f"  Report (Markdown):  {paths['markdown']}",
        f"  Report (JSON):      {paths['json']}",
        f"  Weights:            {weights}",
    ]
    if pred_dir is not None:
        lines.append(f"  Inference output:   {pred_dir}")
    lines.extend([
        f"  Preview images:     {base}/preview/",
        f"  MLflow UI:          {config.mlflow_tracking_uri}",
        "  Grafana dashboard:  http://localhost:3001/d/poker-yolo-main/poker-yolo-training-and-inference",
        "=" * 60,
    ])
    if report and report.metrics:
        key_metrics = {k: v for k, v in report.metrics.items() if k.startswith(("val_", "infer_"))}
        if key_metrics:
            lines.insert(-1, "  Key metrics:")
            for name, value in sorted(key_metrics.items()):
                lines.insert(
                    -1,
                    f"    {name}: {value:.4f}" if isinstance(value, float) else f"    {name}: {value}",
                )
    logger.info("\n".join(lines))


def _run_full_pipeline(
    config: Config,
    *,
    infer_source: Path | None = None,
    skip_infer: bool = False,
    save_infer: bool = True,
) -> int:
    """Train → validate → infer → enrich report → finalize."""
    weights, train_duration = run_training(config)
    logger.info("Training finished. Weights: %s (%.1fs)", weights, train_duration)

    val_metrics, val_duration = run_validation(config, weights)
    logger.info("Validation finished: %s (%.1fs)", val_metrics, val_duration)

    pred_dir: Path | None = None
    infer_duration = 0.0

    if not skip_infer:
        source = infer_source or config.infer_source
        if not source.exists():
            logger.error("Inference source not found: %s", source)
            finalize_report(config.reporting, status="failed", error=f"Inference source not found: {source}")
            return 1

        t0 = time.perf_counter()
        pred_dir = run_inference(config, weights, source, save=save_infer)
        infer_duration = time.perf_counter() - t0
        logger.info("Inference finished: %s (%.1fs)", pred_dir, infer_duration)

    _enrich_report_after_train(
        config,
        weights,
        train_duration,
        val_metrics,
        val_duration,
        pred_dir=pred_dir,
        infer_duration=infer_duration,
    )
    paths = finalize_report(config.reporting, status="success")
    _print_results_summary(config, paths, weights, pred_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_path = args.config or _default_config()
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        return 1

    config = Config.from_yaml(config_path)
    if args.command in {"train", "validate", "infer"}:
        ensure_dataset_if_needed(config.dataset_root)
    setup_logging(
        config.reporting.log_dir,
        level=config.reporting.level,
        json_file=config.reporting.json_logs,
        console_json=config.reporting.console_json,
    )

    report = start_report(args.command, config_name=config.name)
    _populate_report_params(report, config)

    try:
        if args.command == "train":
            return _run_full_pipeline(
                config,
                infer_source=args.infer_source,
                skip_infer=args.skip_infer,
                save_infer=not args.no_save,
            )

        if args.command == "validate":
            weights = args.weights or _default_weights(config)
            if not weights.exists():
                logger.error("Weights not found: %s", weights)
                finalize_report(config.reporting, status="failed", error=f"Weights not found: {weights}")
                return 1
            val_metrics, val_duration = run_validation(config, weights)
            _enrich_report_after_train(config, weights, 0.0, val_metrics, val_duration)
            paths = finalize_report(config.reporting, status="success")
            logger.info("Final report: %s", paths["markdown"])
            return 0

        if args.command == "infer":
            weights = args.weights or _default_weights(config)
            if not weights.exists():
                logger.error("Weights not found: %s", weights)
                finalize_report(config.reporting, status="failed", error=f"Weights not found: {weights}")
                return 1
            if not args.source.exists():
                logger.error("Source not found: %s", args.source)
                finalize_report(config.reporting, status="failed", error=f"Source not found: {args.source}")
                return 1
            pred_dir = run_inference(config, weights, args.source, save=not args.no_save)
            paths = finalize_report(config.reporting, status="success")
            logger.info("Final report: %s", paths["markdown"])
            logger.info("Inference output: %s", pred_dir)
            return 0

        parser.print_help()
        finalize_report(config.reporting, status="failed", error="Unknown command")
        return 1

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        finalize_report(config.reporting, status="failed", error=str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
