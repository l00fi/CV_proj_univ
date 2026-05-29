"""Tests for Kaggle dataset download/conversion helpers."""

from __future__ import annotations

from pathlib import Path

import yaml

from poker_yolo.kaggle_dataset import (
    _convert_mrph_to_classification,
    _normalize_hand_type,
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


def test_normalize_hand_type() -> None:
    assert _normalize_hand_type("RoyalFlush") == "royal_flush"
    assert _normalize_hand_type("ThreeOfAKind") == "three_of_a_kind"
    assert _normalize_hand_type("HighCard") == "high_card"
    assert _normalize_hand_type("TwoPairs") == "two_pair"


def test_convert_mrph_to_classification(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir(parents=True)
    for idx in range(6):
        (images / f"img_{idx}.jpg").write_bytes(b"fake")
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text(
        "filename,hand_type\n"
        "img_0.jpg,RoyalFlush\n"
        "img_1.jpg,RoyalFlush\n"
        "img_2.jpg,Pair\n"
        "img_3.jpg,Pair\n"
        "img_4.jpg,Pair\n"
        "img_5.jpg,Straight\n",
        encoding="utf-8",
    )
    target = tmp_path / "kaggle"
    classes = _convert_mrph_to_classification(
        images_root=images,
        labels_csv=labels_csv,
        target_root=target,
        split_seed=7,
    )

    assert classes == ["pair", "royal_flush", "straight"]
    assert any((target / "train" / "pair").glob("*.jpg"))
    assert any((target / "val" / "pair").glob("*.jpg")) or any((target / "test" / "pair").glob("*.jpg"))


def test_ensure_kaggle_yolo_dataset_uses_marker(tmp_path: Path, mocker) -> None:
    target = tmp_path / "kaggle"
    mocker.patch(
        "poker_yolo.kaggle_dataset._download_kaggle_dataset",
        return_value=target / "_download",
    )
    mocker.patch(
        "poker_yolo.kaggle_dataset._prepare_classification_dataset",
        return_value=["pair", "straight", "royal_flush"],
    )

    first = ensure_kaggle_yolo_dataset(target)
    second = ensure_kaggle_yolo_dataset(target)
    assert first == second
    assert (target / ".kaggle_ready").exists()
    assert yaml.safe_load(first.read_text(encoding="utf-8"))["nc"] == 3
    assert yaml.safe_load(first.read_text(encoding="utf-8"))["val"] == "val"


def test_ensure_kaggle_yolo_dataset_rebuilds_when_slug_changes(tmp_path: Path, mocker) -> None:
    target = tmp_path / "kaggle"
    download = mocker.patch(
        "poker_yolo.kaggle_dataset._download_kaggle_dataset",
        return_value=target / "_download",
    )
    convert = mocker.patch(
        "poker_yolo.kaggle_dataset._prepare_classification_dataset",
        return_value=["pair", "straight"],
    )

    ensure_kaggle_yolo_dataset(target, dataset_slug="owner/dataset-a")
    ensure_kaggle_yolo_dataset(target, dataset_slug="owner/dataset-b")

    assert download.call_count == 2
    assert convert.call_count == 2
    marker = (target / ".kaggle_ready").read_text(encoding="utf-8")
    assert "dataset=owner/dataset-b" in marker
    assert "task=classify" in marker
    assert "format=2" in marker
