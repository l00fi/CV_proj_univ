"""Tests for MLflow utility helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from poker_yolo.config import Config
from poker_yolo.mlflow_utils import (
    log_config,
    log_json_artifact,
    log_metrics,
    log_ultralytics_results,
    setup_mlflow,
)


def test_setup_mlflow_starts_run(minimal_config_path, project_root, mock_mlflow, mocker) -> None:
    import poker_yolo.mlflow_utils as mlflow_utils

    config = Config.from_yaml(minimal_config_path, project_root=project_root)
    setup_mlflow(config, run_name="unit-test")

    mlflow_utils.mlflow.set_tracking_uri.assert_called_once_with(config.mlflow_tracking_uri)
    mlflow_utils.mlflow.start_run.assert_called_once()
    mlflow_utils.mlflow.set_tag.assert_called_once_with("project", "poker-yolo")


def test_log_config_includes_augmentation_params(minimal_config_path, project_root, mock_mlflow, mocker) -> None:
    import poker_yolo.mlflow_utils as mlflow_utils

    config = Config.from_yaml(minimal_config_path, project_root=project_root)
    log_config(config)

    params = mlflow_utils.mlflow.log_params.call_args[0][0]
    assert params["epochs"] == 1
    assert params["aug_enabled"] is True
    assert params["mosaic"] == 1.0
    assert params["aug_alb_blur"] == 0.1


def test_log_metrics_filters_none(mock_mlflow, mocker) -> None:
    import poker_yolo.mlflow_utils as mlflow_utils

    log_metrics({"map50": 0.9, "missing": None})
    mlflow_utils.mlflow.log_metrics.assert_called_once_with({"map50": 0.9}, step=None)


def test_log_ultralytics_results_with_prefix(mock_mlflow, mocker) -> None:
    import poker_yolo.mlflow_utils as mlflow_utils

    results = SimpleNamespace(
        box=SimpleNamespace(map50=0.8, map=0.6, mp=0.85, mr=0.75, maps=[0.7, 0.9]),
    )
    log_ultralytics_results(results, prefix="val_")

    calls = mlflow_utils.mlflow.log_metrics.call_args_list
    assert len(calls) == 2
    assert calls[0][0][0]["val_map50"] == 0.8
    assert calls[1][0][0]["class_map_0"] == 0.7


def test_log_json_artifact_writes_and_cleans_up(mock_mlflow, mocker, tmp_path, monkeypatch) -> None:
    import poker_yolo.mlflow_utils as mlflow_utils

    filename = "summary.json"
    log_json_artifact({"map50": 0.88}, filename)

    mlflow_utils.mlflow.log_artifact.assert_called_once()
    assert not Path(filename).exists()
