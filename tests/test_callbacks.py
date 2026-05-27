"""Tests for MLflow epoch callbacks."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import mlflow

from poker_yolo.callbacks import mlflow_epoch_callback, register_mlflow_callbacks, _on_train_start


def test_mlflow_epoch_callback_logs_numeric_metrics(mocker) -> None:
    mocker.patch("poker_yolo.callbacks.mlflow.active_run", return_value=MagicMock())
    log_metrics = mocker.patch("poker_yolo.callbacks.mlflow.log_metrics")
    mocker.patch("poker_yolo.callbacks.log_event")

    trainer = SimpleNamespace(
        epoch=3,
        metrics={"loss": 1.5, "map50": 0.8, "precision(B)": 0.7, "tag": "ignored"},
    )
    mlflow_epoch_callback(trainer)

    log_metrics.assert_called_once_with(
        {"epoch_loss": 1.5, "epoch_map50": 0.8, "epoch_precision_B": 0.7},
        step=3,
    )


def test_mlflow_epoch_callback_skips_mlflow_without_active_run(mocker) -> None:
    mocker.patch("poker_yolo.callbacks.mlflow.active_run", return_value=None)
    log_metrics = mocker.patch("poker_yolo.callbacks.mlflow.log_metrics")
    mocker.patch("poker_yolo.callbacks.log_event")

    mlflow_epoch_callback(SimpleNamespace(epoch=1, metrics={"loss": 1.0}))
    log_metrics.assert_not_called()


def test_register_mlflow_callbacks(mocker) -> None:
    model = MagicMock()
    register_mlflow_callbacks(model)

    assert model.add_callback.call_count == 4
    model.add_callback.assert_any_call("on_train_start", _on_train_start)
