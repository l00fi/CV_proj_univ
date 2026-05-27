"""Tests for training, validation and inference orchestration (mocked YOLO)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from poker_yolo.config import Config
from poker_yolo.infer import run_inference
from poker_yolo.train import run_training
from poker_yolo.validate import run_validation


def _mock_yolo_class(mocker):
    mock_model = MagicMock()
    mock_yolo_cls = mocker.patch("poker_yolo.train.YOLO", return_value=mock_model)
    return mock_yolo_cls, mock_model


def test_run_training_logs_to_mlflow_and_returns_best_weights(
    minimal_config_path, project_root, tmp_path, mock_mlflow, mocker,
) -> None:
    save_dir = tmp_path / "runs" / "train" / "test_run"
    (save_dir / "weights").mkdir(parents=True)
    best = save_dir / "weights" / "best.pt"
    best.write_bytes(b"weights")
    (save_dir / "results.csv").write_text("epoch,map50\n1,0.5\n", encoding="utf-8")

    mock_yolo_cls, mock_model = _mock_yolo_class(mocker)
    mocker.patch("poker_yolo.validate.YOLO", mock_yolo_cls)
    mock_model.train.return_value = SimpleNamespace(
        save_dir=str(save_dir),
        box=SimpleNamespace(map50=0.75, map=0.55, mp=0.8, mr=0.7),
    )

    config = Config.from_yaml(minimal_config_path, project_root=project_root)
    weights, duration = run_training(config)

    assert weights == best
    assert duration >= 0.0
    mock_model.train.assert_called_once()
    train_kwargs = mock_model.train.call_args.kwargs
    assert "augmentations" in train_kwargs
    assert train_kwargs["mosaic"] == 1.0
    assert mock_model.add_callback.call_count == 4


def test_run_validation_calls_yolo_val(
    minimal_config_path, project_root, tmp_path, mock_mlflow, mocker,
) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"w")

    mock_model = MagicMock()
    mock_model.val.return_value = SimpleNamespace(
        box=SimpleNamespace(map50=0.9, map=0.7, mp=0.88, mr=0.82),
    )
    mocker.patch("poker_yolo.validate.YOLO", return_value=mock_model)

    config = Config.from_yaml(minimal_config_path, project_root=project_root)
    metrics, val_duration = run_validation(config, weights)

    assert metrics["map50"] == 0.9
    assert val_duration >= 0.0
    assert metrics["f1"] == pytest.approx(2 * 0.88 * 0.82 / (0.88 + 0.82))
    mock_model.val.assert_called_once_with(
        data=str(config.data_yaml),
        split="test",
        imgsz=config.imgsz,
        conf=config.val_metric_conf,
        iou=config.val_iou,
        device=config.device,
        verbose=True,
    )


def test_run_inference_returns_output_dir(
    minimal_config_path, project_root, tmp_path, mock_mlflow, mocker,
) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"w")
    source = tmp_path / "img.jpg"
    source.write_bytes(b"jpg")

    mock_result = MagicMock()
    mock_result.save_dir = str(tmp_path / "pred")
    mock_model = MagicMock()
    mock_model.predict.return_value = [mock_result]
    mocker.patch("poker_yolo.infer.YOLO", return_value=mock_model)
    mocker.patch("poker_yolo.infer.time.time", return_value=12345)

    config = Config.from_yaml(minimal_config_path, project_root=project_root)
    output = run_inference(config, weights, source, save=True)

    assert output.name == "pred_12345"
    mock_model.predict.assert_called_once()
    predict_kwargs = mock_model.predict.call_args.kwargs
    assert predict_kwargs["conf"] == config.infer_conf
    assert predict_kwargs["save"] is True
