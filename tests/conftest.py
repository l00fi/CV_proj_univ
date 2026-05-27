"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolate_cli_and_reporting_env(request, mocker, monkeypatch) -> None:
    """Prevent CLI tests from hanging on Pushgateway / prediction export."""
    monkeypatch.delenv("PROMETHEUS_PUSHGATEWAY_URL", raising=False)
    if request.node.fspath.basename == "test_cli.py":
        mocker.patch("poker_yolo.cli._enrich_report_after_train")


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def default_config_path(project_root: Path) -> Path:
    path = project_root / "configs" / "default.yaml"
    assert path.exists(), f"Missing config: {path}"
    return path


@pytest.fixture
def minimal_config_path(tmp_path: Path, project_root: Path) -> Path:
    data_yaml = project_root / "dataset" / "data.yaml"
    config = {
        "data": {
            "yaml_path": str(data_yaml),
            "dataset_root": str(project_root / "dataset"),
        },
        "model": {"weights": "yolov8n.pt", "imgsz": 640},
        "train": {
            "epochs": 1,
            "batch": 2,
            "patience": 1,
            "device": "cpu",
            "workers": 0,
            "project": str(tmp_path / "runs" / "train"),
            "name": "test_run",
            "seed": 0,
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 1,
        },
        "augmentations": {
            "enabled": True,
            "mosaic": 1.0,
            "mixup": 0.1,
            "copy_paste": 0.2,
            "cutmix": 0.05,
            "fliplr": 0.5,
            "flipud": 0.0,
            "degrees": 10.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 3.0,
            "perspective": 0.0003,
            "hsv_h": 0.015,
            "hsv_s": 0.6,
            "hsv_v": 0.35,
            "albumentations": {
                "blur": 0.1,
                "brightness_contrast": 0.4,
                "gauss_noise": 0.1,
            },
        },
        "validate": {"conf": 0.25, "iou": 0.45, "split": "test"},
        "infer": {"conf": 0.35, "iou": 0.45, "save_dir": str(tmp_path / "runs" / "infer")},
        "mlflow": {
            "tracking_uri": "http://localhost:5000",
            "experiment_name": "poker-yolo-test",
            "run_name": "test",
        },
        "reporting": {
            "log_dir": str(tmp_path / "runs" / "logs"),
            "report_dir": str(tmp_path / "runs" / "reports"),
            "level": "INFO",
            "json_logs": True,
            "console_json": False,
            "reports_base_url": "http://localhost:8088",
            "preview_samples": 3,
        },
    }
    path = tmp_path / "test_config.yaml"
    path.write_text(yaml.dump(config), encoding="utf-8")
    return path


@pytest.fixture
def mock_mlflow(mocker):
    """Patch mlflow calls so tests do not require a running server."""
    mocker.patch("poker_yolo.mlflow_utils._wait_for_mlflow")
    mocker.patch("poker_yolo.mlflow_utils.mlflow.set_tracking_uri")
    mocker.patch("poker_yolo.mlflow_utils.mlflow.set_experiment", return_value=mocker.Mock(experiment_id="exp-1"))
    active_run = mocker.Mock()
    mocker.patch("poker_yolo.mlflow_utils.mlflow.start_run", return_value=active_run)
    mocker.patch("poker_yolo.mlflow_utils.mlflow.active_run", return_value=active_run)
    mocker.patch("poker_yolo.mlflow_utils.mlflow.end_run")
    mocker.patch("poker_yolo.mlflow_utils.mlflow.log_params")
    mocker.patch("poker_yolo.mlflow_utils.mlflow.log_param")
    mocker.patch("poker_yolo.mlflow_utils.mlflow.log_metrics")
    mocker.patch("poker_yolo.mlflow_utils.mlflow.log_artifact")
    mocker.patch("poker_yolo.mlflow_utils.mlflow.log_artifacts")
    mocker.patch("poker_yolo.mlflow_utils.mlflow.set_tag")
    mocker.patch("mlflow.log_metrics")
    mocker.patch("mlflow.active_run", return_value=active_run)
    mocker.patch("mlflow.end_run")
    mocker.patch("mlflow.log_param")
    return active_run
