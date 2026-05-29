#!/usr/bin/env python3
"""Merge Roboflow train+test into a single hands evaluation split under dataset/test."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def merge_hands_dataset(dataset_root: Path) -> None:
    dataset_root = dataset_root.resolve()
    test_images = dataset_root / "test" / "images"
    test_labels = dataset_root / "test" / "labels"
    train_images = dataset_root / "train" / "images"
    train_labels = dataset_root / "train" / "labels"
    data_yaml = dataset_root / "data.yaml"

    test_images.mkdir(parents=True, exist_ok=True)
    test_labels.mkdir(parents=True, exist_ok=True)

    if train_images.exists():
        for image in train_images.iterdir():
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            dest = test_images / image.name
            if not dest.exists():
                shutil.copy2(image, dest)
            label = train_labels / f"{image.stem}.txt"
            if label.exists():
                dest_label = test_labels / label.name
                if not dest_label.exists():
                    shutil.copy2(label, dest_label)

    if train_images.exists():
        shutil.rmtree(dataset_root / "train", ignore_errors=True)

    raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) if data_yaml.exists() else {}
    names = raw.get("names", [])
    payload = {
        "path": str(dataset_root.resolve()),
        "train": "test/images",
        "val": "test/images",
        "test": "test/images",
        "nc": raw.get("nc", len(names)),
        "names": names,
        "purpose": "hands_evaluation",
    }
    data_yaml.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Hands dataset ready: {len(list(test_images.glob('*')))} images in {test_images}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="Root folder with Roboflow export",
    )
    args = parser.parse_args()
    merge_hands_dataset(args.dataset_root)


if __name__ == "__main__":
    main()
