"""Tests for YAML config loading and train argument building."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from poker_yolo.config import Config


def test_device_auto_resolves_to_cpu_without_cuda(default_config_path: Path, project_root: Path, mocker) -> None:
    mocker.patch("torch.cuda.is_available", return_value=False)
    config = Config.from_yaml(default_config_path, project_root=project_root)
    assert config.device == "cpu"
    assert config.train_args()["device"] == "cpu"


def test_device_auto_resolves_to_gpu_when_available(default_config_path: Path, project_root: Path, mocker) -> None:
    mocker.patch("torch.cuda.is_available", return_value=True)
    mocker.patch("torch.cuda.device_count", return_value=1)
    config = Config.from_yaml(default_config_path, project_root=project_root)
    assert config.device == "0"


def test_poker_yolo_device_env_overrides_cuda_probe(
    default_config_path: Path, project_root: Path, monkeypatch, mocker,
) -> None:
    monkeypatch.setenv("POKER_YOLO_DEVICE", "0")
    mocker.patch("torch.cuda.is_available", return_value=False)
    config = Config.from_yaml(default_config_path, project_root=project_root)
    assert config.device == "0"


def test_config_from_default_yaml(default_config_path: Path, project_root: Path) -> None:
    config = Config.from_yaml(default_config_path, project_root=project_root)

    assert config.model_weights == "yolov8n.pt"
    assert config.imgsz == 640
    assert config.epochs == 50
    assert config.augmentations.enabled is True
    assert config.augmentations.mosaic == 1.0
    assert config.augmentations.mixup == 0.2
    assert config.data_yaml == (project_root / "dataset" / "kaggle" / "data.yaml").resolve()
    assert config.dataset_root == (project_root / "dataset" / "kaggle").resolve()
    assert config.val_split == "val"
    assert config.val_metric_conf == 0.001
    assert config.infer_source.name == "images"


def test_config_from_minimal_yaml(minimal_config_path: Path, project_root: Path) -> None:
    config = Config.from_yaml(minimal_config_path, project_root=project_root)

    assert config.epochs == 1
    assert config.batch == 2
    assert config.device == "cpu"
    assert config.augmentations.copy_paste == 0.2
    assert "blur" in config.augmentations.albumentations
    assert config.reporting.report_dir.name == "reports"


def test_mlflow_tracking_uri_env_override(minimal_config_path: Path, project_root: Path, monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://custom:5000")
    config = Config.from_yaml(minimal_config_path, project_root=project_root)
    assert config.mlflow_tracking_uri == "http://custom:5000"


def test_train_args_include_ultralytics_augmentations(minimal_config_path: Path, project_root: Path) -> None:
    config = Config.from_yaml(minimal_config_path, project_root=project_root)
    args = config.train_args()

    assert args["mosaic"] == 1.0
    assert args["mixup"] == 0.1
    assert args["copy_paste"] == 0.2
    assert args["cutmix"] == 0.05
    assert args["data"] == str(config.data_yaml)
    assert args["pretrained"] is True


def test_train_args_include_albumentations_when_enabled(minimal_config_path: Path, project_root: Path) -> None:
    config = Config.from_yaml(minimal_config_path, project_root=project_root)
    args = config.train_args()

    assert "augmentations" in args
    assert len(args["augmentations"]) == 3


def test_train_args_skip_albumentations_when_disabled(minimal_config_path: Path, project_root: Path) -> None:
    import yaml

    raw = yaml.safe_load(minimal_config_path.read_text(encoding="utf-8"))
    raw["augmentations"]["enabled"] = False
    disabled_path = minimal_config_path.parent / "disabled.yaml"
    disabled_path.write_text(yaml.dump(raw), encoding="utf-8")

    config = Config.from_yaml(disabled_path, project_root=project_root)
    args = config.train_args()

    assert "augmentations" not in args


def test_augmentation_fallback_from_train_section(tmp_path: Path, project_root: Path) -> None:
    import yaml

    config_dict = {
        "data": {
            "yaml_path": str(project_root / "dataset" / "data.yaml"),
            "dataset_root": str(project_root / "dataset"),
        },
        "model": {"weights": "yolov8n.pt", "imgsz": 640},
        "train": {
            "epochs": 1,
            "batch": 2,
            "patience": 1,
            "device": "cpu",
            "workers": 0,
            "project": "runs/train",
            "name": "test",
            "seed": 0,
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 1,
            "mosaic": 0.8,
            "mixup": 0.05,
            "degrees": 7.0,
            "translate": 0.08,
            "scale": 0.4,
            "hsv_h": 0.01,
            "hsv_s": 0.5,
            "hsv_v": 0.3,
        },
        "validate": {"metric_conf": 0.001, "iou": 0.45, "split": "test"},
        "infer": {"conf": 0.35, "iou": 0.45, "save_dir": "runs/infer"},
        "mlflow": {
            "tracking_uri": "http://localhost:5000",
            "experiment_name": "test",
            "run_name": None,
        },
        "reporting": {
            "log_dir": "runs/logs",
            "report_dir": "runs/reports",
        },
    }
    path = tmp_path / "fallback.yaml"
    path.write_text(yaml.dump(config_dict), encoding="utf-8")

    config = Config.from_yaml(path, project_root=project_root)
    assert config.augmentations.mosaic == 0.8
    assert config.augmentations.mixup == 0.05
    assert config.augmentations.degrees == 7.0
