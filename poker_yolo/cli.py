from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from poker_yolo.config import Config
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

    sub.add_parser("train", help="Train YOLOv8 model")

    val_parser = sub.add_parser("validate", help="Validate trained model")
    val_parser.add_argument("--weights", type=Path, default=None, help="Path to .pt weights")

    infer_parser = sub.add_parser("infer", help="Run inference on images")
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
        }
    )


def _enrich_report_after_train(
    config: Config,
    weights: Path,
    train_duration: float,
    val_metrics: dict[str, float],
    val_duration: float,
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

    production = compute_production_metrics(
        weights_path=weights,
        train_duration_sec=train_duration,
        val_duration_sec=val_duration,
        resource_metrics=dict(report.resources),
        val_metrics=val_metrics,
        dataset_stats=dataset_stats,
        aug_summary=aug_summary,
    )
    report.set_production(production)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_path = args.config or _default_config()
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        return 1

    config = Config.from_yaml(config_path)
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
            weights, train_duration = run_training(config)
            logger.info("Training finished. Weights: %s (%.1fs)", weights, train_duration)
            val_metrics, val_duration = run_validation(config, weights)
            logger.info("Post-train validation: %s (%.1fs)", val_metrics, val_duration)
            _enrich_report_after_train(config, weights, train_duration, val_metrics, val_duration)
            paths = finalize_report(config.reporting, status="success")
            logger.info("Final report: %s", paths["markdown"])
            return 0

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
            run_inference(config, weights, args.source, save=not args.no_save)
            paths = finalize_report(config.reporting, status="success")
            logger.info("Final report: %s", paths["markdown"])
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
