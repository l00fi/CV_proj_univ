"""Tests for CLI argument parsing and command dispatch."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from poker_yolo.cli import _default_weights, build_parser, main


def test_build_parser_subcommands() -> None:
    parser = build_parser()
    args = parser.parse_args(["train"])
    assert args.command == "train"

    args = parser.parse_args(["train", "--skip-infer"])
    assert args.command == "train"
    assert args.skip_infer is True

    args = parser.parse_args(["validate", "--weights", "model.pt"])
    assert args.command == "validate"
    assert args.weights == Path("model.pt")

    args = parser.parse_args(["infer", "--source", "images/", "--no-save"])
    assert args.command == "infer"
    assert args.source == Path("images/")
    assert args.no_save is True


def test_main_train_dispatches_training_and_validation(
    minimal_config_path, mocker,
) -> None:
    mock_train = mocker.patch("poker_yolo.cli.run_training", return_value=(Path("best.pt"), 10.0))
    mock_val = mocker.patch("poker_yolo.cli.run_validation", return_value=({"map50": 0.5}, 2.0))
    mock_infer = mocker.patch("poker_yolo.cli.run_inference", return_value=Path("runs/infer/pred_1"))
    mocker.patch("poker_yolo.cli._enrich_report_after_train")
    mocker.patch("poker_yolo.cli._print_results_summary")
    mocker.patch("poker_yolo.cli.setup_logging")

    code = main(["--config", str(minimal_config_path), "train"])

    assert code == 0
    mock_train.assert_called_once()
    mock_val.assert_called_once()
    mock_infer.assert_called_once()


def test_main_validate_missing_weights_returns_error(minimal_config_path, mocker) -> None:
    mocker.patch("poker_yolo.cli.run_validation")
    mocker.patch("poker_yolo.cli.setup_logging")
    code = main(["--config", str(minimal_config_path), "validate"])
    assert code == 1


def test_main_validate_with_weights(minimal_config_path, tmp_path, mocker) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake")
    mock_val = mocker.patch("poker_yolo.cli.run_validation", return_value=({"map50": 0.5}, 1.0))
    mocker.patch("poker_yolo.cli._enrich_report_after_train")
    mocker.patch("poker_yolo.cli.setup_logging")

    code = main(["--config", str(minimal_config_path), "validate", "--weights", str(weights)])

    assert code == 0
    mock_val.assert_called_once()


def test_main_infer_missing_source_returns_error(minimal_config_path, tmp_path, mocker) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake")
    mocker.patch("poker_yolo.cli.setup_logging")
    code = main([
        "--config", str(minimal_config_path),
        "infer", "--weights", str(weights), "--source", str(tmp_path / "missing"),
    ])
    assert code == 1


def test_main_infer_success(minimal_config_path, tmp_path, project_root, mocker) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake")
    source = project_root / "dataset" / "test" / "images"
    if not source.exists():
        pytest.skip("Test images not available")

    mock_infer = mocker.patch("poker_yolo.cli.run_inference", return_value=tmp_path / "pred")
    mocker.patch("poker_yolo.cli.setup_logging")
    code = main([
        "--config", str(minimal_config_path),
        "infer", "--weights", str(weights), "--source", str(source),
    ])

    assert code == 0
    mock_infer.assert_called_once()


def test_main_train_skips_infer_when_requested(minimal_config_path, mocker) -> None:
    mocker.patch("poker_yolo.cli.run_training", return_value=(Path("best.pt"), 10.0))
    mocker.patch("poker_yolo.cli.run_validation", return_value=({"map50": 0.5}, 2.0))
    mock_infer = mocker.patch("poker_yolo.cli.run_inference")
    mocker.patch("poker_yolo.cli._enrich_report_after_train")
    mocker.patch("poker_yolo.cli._print_results_summary")
    mocker.patch("poker_yolo.cli.setup_logging")

    code = main(["--config", str(minimal_config_path), "train", "--skip-infer"])

    assert code == 0
    mock_infer.assert_not_called()


def test_main_creates_final_report(minimal_config_path, tmp_path, mocker) -> None:
    mocker.patch("poker_yolo.cli.setup_logging")
    mocker.patch("poker_yolo.cli.run_training", return_value=(Path("best.pt"), 10.0))
    mocker.patch("poker_yolo.cli.run_validation", return_value=({"map50": 0.5}, 2.0))
    mocker.patch("poker_yolo.cli.run_inference", return_value=tmp_path / "pred")
    mocker.patch("poker_yolo.cli._enrich_report_after_train")
    mocker.patch("poker_yolo.cli._print_results_summary")

    main(["--config", str(minimal_config_path), "train"])

    reports_dir = tmp_path / "runs" / "reports"
    assert (reports_dir / "latest.json").exists()
    assert (reports_dir / "latest.md").exists()


def test_main_config_not_found() -> None:
    code = main(["--config", "nonexistent.yaml", "train"])
    assert code == 1


def test_default_weights_fallback(minimal_config_path, project_root, tmp_path) -> None:
    config = __import__("poker_yolo.config", fromlist=["Config"]).Config.from_yaml(
        minimal_config_path, project_root=project_root,
    )
    weights_dir = tmp_path / "runs" / "train" / config.name / "weights"
    weights_dir.mkdir(parents=True)
    best = weights_dir / "best.pt"
    best.write_bytes(b"x")

    config.project = str(tmp_path / "runs" / "train")
    assert _default_weights(config) == best
