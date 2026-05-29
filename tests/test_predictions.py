"""Tests for hands benchmark and classification result parsing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from poker_yolo.classify_results import extract_top_classes
from poker_yolo.config import Config
from poker_yolo.predictions import analyze_hands_benchmark


def test_extract_top_classes_handles_tensor_probs() -> None:
    import torch

    mock_result = MagicMock()
    probs = MagicMock()
    probs.top5 = torch.tensor([2, 0, 1])
    probs.top5conf = torch.tensor([0.9, 0.5, 0.3])
    mock_result.probs = probs
    mock_result.names = {0: "pair", 1: "flush", 2: "straight"}

    ranked = extract_top_classes(mock_result, limit=3)
    assert ranked[0]["class_id"] == 2
    assert ranked[0]["class_name"] == "straight"
    assert ranked[0]["confidence"] == pytest.approx(0.9)


def test_analyze_hands_benchmark_creates_previews(minimal_config_path, project_root, tmp_path, mocker) -> None:
    config = Config.from_yaml(minimal_config_path, project_root=project_root)
    config.reporting.report_dir = tmp_path / "reports"

    test_dir = project_root / "dataset" / "test" / "images"
    if not test_dir.exists():
        pytest.skip("Test images unavailable")

    images = sorted(test_dir.glob("*"))[:3]
    if not images:
        pytest.skip("No test images")

    mocker.patch("poker_yolo.predictions.hands_test_images", return_value=images)

    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake")

    mock_result = MagicMock()
    probs = MagicMock()
    probs.top5 = [0, 1, 2, 3, 4]
    probs.top5conf = [0.95, 0.7, 0.4, 0.2, 0.1]
    mock_result.probs = probs
    mock_result.names = {0: "royal_flush", 1: "straight_flush", 2: "four_of_a_kind"}
    mock_result.save = MagicMock()

    mock_model = MagicMock()
    mock_model.predict.return_value = [mock_result for _ in images]
    mocker.patch("poker_yolo.predictions.YOLO", return_value=mock_model)

    analysis = analyze_hands_benchmark(
        config,
        weights,
        config.reporting.report_dir,
        n_samples=3,
        reports_base_url="http://localhost:8088",
    )

    assert len(analysis.samples) >= 1
    assert analysis.samples[0]["preview_url"].startswith("http://localhost:8088/preview/")
    assert "top_classes" in analysis.samples[0]
    assert "predicted_combo" in analysis.samples[0]
    assert (config.reporting.report_dir / "preview" / "manifest.json").exists()
