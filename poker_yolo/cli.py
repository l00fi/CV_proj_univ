from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from poker_yolo.config import Config, mlflow_ui_url
from poker_yolo.device import release_cuda_memory
from poker_yolo.infer import run_inference
from poker_yolo.kaggle_dataset import ensure_dataset_if_needed, ensure_hands_data_yaml
from poker_yolo.logging_config import setup_logging
from poker_yolo.paths import default_pipeline_config, default_weights
from poker_yolo.pipeline import enrich_report_after_train, validate_infer_source
from poker_yolo.preflight import run_preflight_cpu_smoke
from poker_yolo.reporting import finalize_report, get_report, start_report
from poker_yolo.train import run_training
from poker_yolo.validate import run_validation

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="poker-yolo",
        description="YOLOv8 pipeline for poker combination classification",
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
        help="Full pipeline: CPU preflight → train → validate → infer → report",
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
    train_parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip CPU preflight (train.preflight_cpu_epochs) before GPU training",
    )

    val_parser = sub.add_parser("validate", help="Validate trained model only")
    val_parser.add_argument("--weights", type=Path, default=None, help="Path to .pt weights")

    infer_parser = sub.add_parser("infer", help="Run inference only")
    infer_parser.add_argument("--weights", type=Path, default=None, help="Path to .pt weights")
    infer_parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Image file, directory, or glob (default: infer.source in config)",
    )
    infer_parser.add_argument("--no-save", action="store_true", help="Skip saving annotated images")

    return parser


def _populate_report_params(report, config: Config) -> None:
    report.set_params(
        {
            "model_weights": config.model_weights,
            "task": config.task,
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
        f"  MLflow UI:          {mlflow_ui_url(config.mlflow_tracking_uri)}",
        f"  Grafana (training): http://localhost:{os.environ.get('GRAFANA_PORT', '3001')}/d/poker-yolo-training/poker-yolo-training",
        f"  Grafana (curves):     http://localhost:{os.environ.get('GRAFANA_PORT', '3001')}/d/poker-yolo-curves/poker-yolo-training-curves",
        f"  Grafana (benchmark): http://localhost:{os.environ.get('GRAFANA_PORT', '3001')}/d/poker-yolo-inference/poker-yolo-benchmark-inference",
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


def _prepare_datasets(config: Config, command: str) -> None:
    """Download Kaggle data only when training/validation need it."""
    if command in {"train", "validate"} and config.dataset_root.name == "kaggle":
        ensure_dataset_if_needed(config.dataset_root, split_seed=config.seed)
    if config.val_data_yaml != config.data_yaml:
        ensure_hands_data_yaml(config.val_data_yaml)


def _run_full_pipeline(
    config: Config,
    *,
    infer_source: Path | None = None,
    skip_infer: bool = False,
    skip_preflight: bool = False,
    save_infer: bool = True,
) -> int:
    if not skip_preflight and config.preflight_cpu_epochs > 0:
        run_preflight_cpu_smoke(config)

    weights, train_duration = run_training(config)
    logger.info("Training finished. Weights: %s (%.1fs)", weights, train_duration)

    val_metrics, val_duration = run_validation(config, weights)
    logger.info("Validation finished: %s (%.1fs)", val_metrics, val_duration)

    pred_dir = None
    infer_duration = 0.0
    infer_results = None

    if not skip_infer:
        source = infer_source or config.infer_source
        if (code := validate_infer_source(source, config.reporting)) is not None:
            return code

        t0 = time.perf_counter()
        infer_run = run_inference(config, weights, source, save=save_infer)
        infer_duration = time.perf_counter() - t0
        pred_dir = infer_run.output_dir
        infer_results = infer_run.results
        logger.info("Inference finished: %s (%.1fs)", pred_dir, infer_duration)
        release_cuda_memory()

    enrich_report_after_train(
        config,
        weights,
        train_duration,
        val_metrics,
        val_duration,
        pred_dir=pred_dir,
        infer_duration=infer_duration,
        infer_results=infer_results,
    )
    paths = finalize_report(config.reporting, status="success")
    _print_results_summary(config, paths, weights, pred_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_path = args.config or default_pipeline_config()
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        return 1

    config = Config.from_yaml(config_path)
    if args.command in {"train", "validate", "infer"}:
        _prepare_datasets(config, args.command)

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
                skip_preflight=args.skip_preflight,
                save_infer=not args.no_save,
            )

        if args.command == "validate":
            weights = args.weights or default_weights(config)
            if not weights.exists():
                logger.error("Weights not found: %s", weights)
                finalize_report(config.reporting, status="failed", error=f"Weights not found: {weights}")
                return 1
            val_metrics, val_duration = run_validation(config, weights)
            enrich_report_after_train(config, weights, 0.0, val_metrics, val_duration)
            paths = finalize_report(config.reporting, status="success")
            logger.info("Final report: %s", paths["markdown"])
            return 0

        if args.command == "infer":
            weights = args.weights or default_weights(config)
            if not weights.exists():
                logger.error("Weights not found: %s", weights)
                finalize_report(config.reporting, status="failed", error=f"Weights not found: {weights}")
                return 1
            source = args.source or config.infer_source
            if (code := validate_infer_source(source, config.reporting)) is not None:
                return code
            infer_run = run_inference(config, weights, source, save=not args.no_save)
            paths = finalize_report(config.reporting, status="success")
            logger.info("Final report: %s", paths["markdown"])
            logger.info("Inference output: %s", infer_run.output_dir)
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
