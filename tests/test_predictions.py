"""Tests for sample prediction export."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from poker_yolo.config import Config
from poker_yolo.predictions import save_sample_predictions


def test_save_sample_predictions_creates_previews(minimal_config_path, project_root, tmp_path, mocker) -> None:
    config = Config.from_yaml(minimal_config_path, project_root=project_root)
    config.reporting.report_dir = tmp_path / "reports"

    test_dir = project_root / "dataset" / "test" / "images"
    if not test_dir.exists():
        import pytest
        pytest.skip("Test images unavailable")

    images = sorted(test_dir.glob("*"))[:3]
    if not images:
        import pytest
        pytest.skip("No test images")

    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake")

    box_item = MagicMock()
    box_item.cls = MagicMock(item=MagicMock(return_value=0))
    box_item.conf = MagicMock(item=MagicMock(return_value=0.95))
    box_item.xyxy = MagicMock()
    box_item.xyxy.tolist = MagicMock(return_value=[10.0, 10.0, 50.0, 50.0])

    mock_box = MagicMock()
    mock_box.__len__ = MagicMock(return_value=1)
    mock_box.__iter__ = MagicMock(return_value=iter([box_item]))

    mock_result = MagicMock()
    mock_result.boxes = mock_box
    mock_result.save = MagicMock()

    mock_model = MagicMock()
    mock_model.predict.return_value = [mock_result]
    mocker.patch("poker_yolo.predictions.YOLO", return_value=mock_model)

    samples = save_sample_predictions(
        config,
        weights,
        config.reporting.report_dir,
        n_samples=3,
        reports_base_url="http://localhost:8088",
    )

    assert len(samples) >= 1
    assert samples[0]["preview_url"].startswith("http://localhost:8088/preview/")
    assert "detections" in samples[0]
    assert (config.reporting.report_dir / "preview" / "manifest.json").exists()
