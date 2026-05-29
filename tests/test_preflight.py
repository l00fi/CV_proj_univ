"""Tests for CPU preflight before GPU training."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

from poker_yolo.config import Config
from poker_yolo.preflight import run_preflight_cpu_smoke


def test_run_preflight_cpu_smoke_skipped_when_zero(minimal_config_path, project_root) -> None:
    config = Config.from_yaml(minimal_config_path, project_root=project_root)
    assert config.preflight_cpu_epochs == 0
    assert run_preflight_cpu_smoke(config) is None


def test_run_preflight_cpu_smoke_runs_cpu_training(
    minimal_config_path, project_root, mocker,
) -> None:
    config = replace(
        Config.from_yaml(minimal_config_path, project_root=project_root),
        preflight_cpu_epochs=1,
    )
    mock_train = mocker.patch(
        "poker_yolo.preflight.run_training",
        return_value=(Path("runs/preflight/best.pt"), 12.0),
    )

    weights, duration = run_preflight_cpu_smoke(config)

    assert duration == 12.0
    assert weights == Path("runs/preflight/best.pt")
    mock_train.assert_called_once()
    preflight_config = mock_train.call_args[0][0]
    assert preflight_config.device == "cpu"
    assert preflight_config.epochs == 1
    assert preflight_config.workers == 0
    assert preflight_config.name == "test_run_preflight"
    assert preflight_config.gpu_resource_fraction is None
