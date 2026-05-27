"""Tests for dataset layout and data.yaml integrity."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_data_yaml_exists_and_has_52_classes(project_root: Path) -> None:
    data_yaml = project_root / "dataset" / "data.yaml"
    assert data_yaml.exists()

    raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    assert raw["nc"] == 52
    assert len(raw["names"]) == 52
    assert "AC" in raw["names"]
    assert "10S" in raw["names"]


def test_data_yaml_paths_exist(project_root: Path) -> None:
    raw = yaml.safe_load((project_root / "dataset" / "data.yaml").read_text(encoding="utf-8"))
    dataset_root = project_root / "dataset"

    train_images = dataset_root / raw["train"]
    test_images = dataset_root / raw["test"]
    train_labels = dataset_root / "train" / "labels"
    test_labels = dataset_root / "test" / "labels"

    assert train_images.exists()
    assert test_images.exists()
    assert train_labels.exists()
    assert test_labels.exists()
    assert len(list(train_images.glob("*"))) > 0
    assert len(list(test_images.glob("*"))) > 0


def test_train_labels_match_images(project_root: Path) -> None:
    train_images = project_root / "dataset" / "train" / "images"
    train_labels = project_root / "dataset" / "train" / "labels"

    image_stems = {p.stem for p in train_images.iterdir() if p.is_file()}
    label_stems = {p.stem for p in train_labels.glob("*.txt")}

    assert image_stems == label_stems
