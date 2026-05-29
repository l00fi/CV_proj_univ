"""Tests for full 10-class benchmark grids."""

from __future__ import annotations

from poker_yolo.hands import COMBO_CLASSES
from poker_yolo.benchmark_grids import (
    empty_confusion_matrix,
    empty_outcomes_by_hand,
    normalize_confusion_matrix,
    normalize_outcomes_by_hand,
)


def test_empty_grids_cover_all_combo_classes() -> None:
    assert len(COMBO_CLASSES) == 10
    outcomes = empty_outcomes_by_hand()
    assert set(outcomes) == set(COMBO_CLASSES)
    confusion = empty_confusion_matrix()
    assert set(confusion) == set(COMBO_CLASSES)
    for row in confusion.values():
        assert set(row) == set(COMBO_CLASSES)


def test_normalize_fills_missing_classes() -> None:
    outcomes = normalize_outcomes_by_hand({"pair": {"correct": 3, "incorrect": 1}})
    assert outcomes["royal_flush"] == {"correct": 0, "incorrect": 0}
    assert outcomes["pair"]["correct"] == 3

    confusion = normalize_confusion_matrix({"pair": {"flush": 2, "pair": 5}})
    assert confusion["pair"]["flush"] == 2
    assert confusion["flush"]["pair"] == 0
