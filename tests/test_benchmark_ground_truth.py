"""Tests for benchmark ground-truth derivation from YOLO card labels."""

from __future__ import annotations

from pathlib import Path

from poker_yolo.predictions import _ground_truth_hand


def test_ground_truth_hand_from_labels(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    label_file = labels_dir / "hand_a.txt"
    # Full house: three 7s + pair of 4s (7C, 7D, 7H, 4C, 4D)
    label_file.write_text(
        "\n".join(
            [
                "24 0.1 0.1 0.1 0.1",
                "25 0.2 0.2 0.1 0.1",
                "26 0.3 0.3 0.1 0.1",
                "12 0.4 0.4 0.1 0.1",
                "13 0.5 0.5 0.1 0.1",
            ]
        ),
        encoding="utf-8",
    )
    names = [
        "10C", "10D", "10H", "10S", "2C", "2D", "2H", "2S",
        "3C", "3D", "3H", "3S", "4C", "4D", "4H", "4S",
        "5C", "5D", "5H", "5S", "6C", "6D", "6H", "6S",
        "7C", "7D", "7H", "7S", "8C", "8D", "8H", "8S",
        "9C", "9D", "9H", "9S", "AC", "AD", "AH", "AS",
        "JC", "JD", "JH", "JS", "KC", "KD", "KH", "KS",
        "QC", "QD", "QH", "QS",
    ]
    image = tmp_path / "images" / "hand_a.jpg"
    image.parent.mkdir()
    image.touch()

    hand = _ground_truth_hand(image, labels_dir, names)
    assert hand == "full_house"
