"""Full 10-class outcome and confusion grids for benchmark reporting."""

from __future__ import annotations

from poker_yolo.hands import COMBO_CLASSES


def empty_outcomes_by_hand() -> dict[str, dict[str, int]]:
    return {name: {"correct": 0, "incorrect": 0} for name in COMBO_CLASSES}


def empty_confusion_matrix() -> dict[str, dict[str, int]]:
    return {true: {pred: 0 for pred in COMBO_CLASSES} for true in COMBO_CLASSES}


def normalize_outcomes_by_hand(raw: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    merged = empty_outcomes_by_hand()
    for hand, counts in raw.items():
        if hand not in merged:
            continue
        merged[hand]["correct"] = int(counts.get("correct", 0))
        merged[hand]["incorrect"] = int(counts.get("incorrect", 0))
    return merged


def normalize_confusion_matrix(raw: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    merged = empty_confusion_matrix()
    for true_hand, preds in raw.items():
        if true_hand not in merged:
            continue
        for pred_hand, count in preds.items():
            if pred_hand in merged[true_hand]:
                merged[true_hand][pred_hand] = int(count)
    return merged
