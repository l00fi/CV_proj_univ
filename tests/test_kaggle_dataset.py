"""Tests for Kaggle dataset download/conversion helpers."""

from __future__ import annotations

from pathlib import Path

import yaml

from poker_yolo.kaggle_dataset import (
    CLASS_NAME_TO_ID,
    _convert_classification_tree_to_yolo,
    ensure_kaggle_yolo_dataset,
    kaggle_folder_to_yolo_name,
)


def test_kaggle_folder_to_yolo_name_maps_standard_cards() -> None:
    assert kaggle_folder_to_yolo_name("ace of clubs") == "AC"
    assert kaggle_folder_to_yolo_name("ten of spades") == "10S"
    assert kaggle_folder_to_yolo_name("king of hearts") == "KH"


def test_kaggle_folder_to_yolo_name_skips_joker() -> None:
    assert kaggle_folder_to_yolo_name("joker") is None
    assert kaggle_folder_to_yolo_name("Joker") is None


def test_convert_classification_tree_to_yolo(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for split in ("train", "valid", "test"):
        class_dir = raw / split / "ace of clubs"
        class_dir.mkdir(parents=True)
        (class_dir / "card.jpg").write_bytes(b"fake")

    target = tmp_path / "yolo"
    _convert_classification_tree_to_yolo(raw, target)

    train_images = list((target / "train" / "images").glob("*.jpg"))
    train_labels = list((target / "train" / "labels").glob("*.txt"))
    assert len(train_images) == 1
    assert len(train_labels) == 1
    class_id = CLASS_NAME_TO_ID["AC"]
    assert train_labels[0].read_text(encoding="utf-8").strip() == f"{class_id} 0.5 0.5 1.0 1.0"


def test_ensure_kaggle_yolo_dataset_uses_marker(tmp_path: Path, mocker) -> None:
    target = tmp_path / "kaggle"
    mocker.patch(
        "poker_yolo.kaggle_dataset._download_kaggle_dataset",
        return_value=target / "_download",
    )
    mocker.patch("poker_yolo.kaggle_dataset._convert_classification_tree_to_yolo")

    first = ensure_kaggle_yolo_dataset(target)
    second = ensure_kaggle_yolo_dataset(target)
    assert first == second
    assert (target / ".kaggle_ready").exists()
    assert yaml.safe_load(first.read_text(encoding="utf-8"))["nc"] == 52
    assert yaml.safe_load(first.read_text(encoding="utf-8"))["val"] == "valid/images"
