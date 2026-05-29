"""Tests for hands-benchmark classification aggregation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from poker_yolo.config import Config
from poker_yolo.predictions import (
    HANDS_BENCHMARK_SOURCE,
    analyze_hands_benchmark,
    analyze_hands_benchmark_from_infer_results,
    hands_test_images,
    infer_results_cover_benchmark,
)


def _mock_result(top_ids: list[int], top_confs: list[float], names: dict[int, str]) -> MagicMock:
    result = MagicMock()
    probs = MagicMock()
    probs.top5 = top_ids
    probs.top5conf = top_confs
    result.probs = probs
    result.names = names
    result.save = MagicMock()
    return result


def test_hands_test_images_use_dataset_test_dir(minimal_config_path: Path, project_root: Path) -> None:
    config = Config.from_yaml(minimal_config_path, project_root=project_root)
    images = hands_test_images(config)
    if not images:
        import pytest

        pytest.skip("Hands benchmark images not available")
    assert all("test" in str(p).replace("\\", "/") for p in images)
    assert images[0].parent.name == "images"


def test_analyze_hands_benchmark_aggregate_stats(
    minimal_config_path: Path, project_root: Path, tmp_path: Path, mocker,
) -> None:
    config = Config.from_yaml(minimal_config_path, project_root=project_root)
    images = hands_test_images(config)
    if len(images) < 2:
        import pytest

        pytest.skip("Need at least two hands benchmark images")

    subset = images[:3]
    mocker.patch("poker_yolo.predictions.hands_test_images", return_value=subset)

    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake")

    names = {0: "pair", 1: "straight", 2: "royal_flush"}
    mock_model = MagicMock()
    mock_model.predict.return_value = [
        _mock_result([0, 1], [0.9, 0.1], names),
        _mock_result([1, 0], [0.8, 0.2], names),
        _mock_result([2, 0], [0.7, 0.3], names),
    ]
    mocker.patch("poker_yolo.predictions.YOLO", return_value=mock_model)

    analysis = analyze_hands_benchmark(
        config,
        weights,
        tmp_path / "reports",
        n_samples=1,
        seed=0,
        reports_base_url="http://localhost:8088",
    )

    assert analysis.aggregate["hands_benchmark_images"] == float(len(subset))
    assert analysis.aggregate["hands_benchmark_images_with_predictions"] == float(len(subset))
    assert analysis.aggregate["hands_benchmark_prediction_rate"] > 0
    assert analysis.predicted_class_counts
    assert len(analysis.samples) == 1
    assert analysis.samples[0]["source_dataset"] == HANDS_BENCHMARK_SOURCE
    assert (tmp_path / "reports" / "preview" / "manifest.json").exists()


def test_analyze_hands_benchmark_reuses_infer_results(
    minimal_config_path: Path, project_root: Path, tmp_path: Path, mocker,
) -> None:
    config = Config.from_yaml(minimal_config_path, project_root=project_root)
    images = hands_test_images(config)
    if len(images) < 2:
        import pytest

        pytest.skip("Need at least two hands benchmark images")

    subset = images[:2]
    mocker.patch("poker_yolo.predictions.hands_test_images", return_value=subset)

    names = {0: "pair", 1: "straight"}
    infer_results = []
    for image_path in subset:
        result = _mock_result([0, 1], [0.9, 0.1], names)
        result.path = str(image_path)
        infer_results.append(result)

    assert infer_results_cover_benchmark(config, infer_results)

    yolo = mocker.patch("poker_yolo.predictions.YOLO")
    analysis = analyze_hands_benchmark_from_infer_results(
        config,
        tmp_path / "reports",
        infer_results,
        n_samples=1,
        seed=0,
    )

    assert analysis is not None
    yolo.assert_not_called()
    assert analysis.aggregate["hands_benchmark_images"] == 2.0
