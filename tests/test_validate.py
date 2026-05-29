"""Tests for validation metric extraction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from poker_yolo.ultralytics_metrics import extract_metrics


def test_extract_metrics_full() -> None:
    results = SimpleNamespace(
        box=SimpleNamespace(map50=0.85, map=0.62, mp=0.90, mr=0.80),
    )
    metrics = extract_metrics(results)

    assert metrics["map50"] == 0.85
    assert metrics["map50_95"] == 0.62
    assert metrics["precision"] == 0.90
    assert metrics["recall"] == 0.80
    assert metrics["f1"] == pytest.approx(2 * 0.90 * 0.80 / (0.90 + 0.80))


def test_extract_metrics_without_box() -> None:
    assert extract_metrics(SimpleNamespace()) == {}


def test_extract_metrics_zero_precision_recall_no_f1() -> None:
    results = SimpleNamespace(box=SimpleNamespace(map50=0.0, map=0.0, mp=0.0, mr=0.0))
    metrics = extract_metrics(results)

    assert "f1" not in metrics
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0

