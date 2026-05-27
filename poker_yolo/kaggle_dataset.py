"""Download Kaggle playing-cards dataset and convert to YOLO layout for training."""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

KAGGLE_DATASET = "gpiosenka/cards-image-datasetclassification"
MARKER_FILENAME = ".kaggle_ready"
YOLO_CLASS_NAMES: list[str] = [
    "10C",
    "10D",
    "10H",
    "10S",
    "2C",
    "2D",
    "2H",
    "2S",
    "3C",
    "3D",
    "3H",
    "3S",
    "4C",
    "4D",
    "4H",
    "4S",
    "5C",
    "5D",
    "5H",
    "5S",
    "6C",
    "6D",
    "6H",
    "6S",
    "7C",
    "7D",
    "7H",
    "7S",
    "8C",
    "8D",
    "8H",
    "8S",
    "9C",
    "9D",
    "9H",
    "9S",
    "AC",
    "AD",
    "AH",
    "AS",
    "JC",
    "JD",
    "JH",
    "JS",
    "KC",
    "KD",
    "KH",
    "KS",
    "QC",
    "QD",
    "QH",
    "QS",
]
CLASS_NAME_TO_ID = {name: idx for idx, name in enumerate(YOLO_CLASS_NAMES)}

RANK_TOKENS = {
    "ace": "A",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "jack": "J",
    "queen": "Q",
    "king": "K",
}
SUIT_TOKENS = {
    "clubs": "C",
    "diamonds": "D",
    "hearts": "H",
    "spades": "S",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def kaggle_folder_to_yolo_name(folder_name: str) -> str | None:
    """Map Kaggle class folder (e.g. 'ace of clubs') to YOLO label ('AC')."""
    text = folder_name.strip().lower().replace("_", " ")
    if "joker" in text:
        return None
    if " of " not in text:
        return None
    rank_part, suit_part = text.split(" of ", 1)
    rank = RANK_TOKENS.get(rank_part.strip())
    suit = SUIT_TOKENS.get(suit_part.strip())
    if rank is None or suit is None:
        return None
    return f"{rank}{suit}"


def ensure_kaggle_yolo_dataset(
    target_root: Path,
    *,
    dataset_slug: str = KAGGLE_DATASET,
    force: bool = False,
) -> Path:
    """Download (once) and build YOLO dataset under target_root. Returns path to data.yaml."""
    target_root = target_root.resolve()
    data_yaml = target_root / "data.yaml"
    marker = target_root / MARKER_FILENAME

    if marker.exists() and data_yaml.exists() and not force:
        _ensure_data_yaml_split_aliases(data_yaml)
        logger.info("Kaggle YOLO dataset already prepared: %s", target_root)
        return data_yaml

    if force and target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    raw_root = _download_kaggle_dataset(target_root, dataset_slug)
    _convert_classification_tree_to_yolo(raw_root, target_root)
    _write_data_yaml(data_yaml)
    _ensure_data_yaml_split_aliases(data_yaml)
    marker.write_text(f"dataset={dataset_slug}\n", encoding="utf-8")
    logger.info("Kaggle dataset ready at %s (%s classes)", target_root, len(YOLO_CLASS_NAMES))
    return data_yaml


def _download_kaggle_dataset(target_root: Path, dataset_slug: str) -> Path:
    download_dir = target_root / "_download"
    download_dir.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError("Install the kaggle package to download the training dataset") from exc

    api = KaggleApi()
    api.authenticate()
    logger.info("Downloading Kaggle dataset %s ...", dataset_slug)
    api.dataset_download_files(dataset_slug, path=str(download_dir), unzip=True, quiet=False)

    extracted_roots = [p for p in download_dir.iterdir() if p.is_dir()]
    if not extracted_roots:
        zips = list(download_dir.glob("*.zip"))
        if zips:
            with zipfile.ZipFile(zips[0], "r") as zf:
                zf.extractall(download_dir)
            extracted_roots = [p for p in download_dir.iterdir() if p.is_dir()]

    for candidate in [download_dir, *extracted_roots]:
        if _find_split_dir(candidate, "train") is not None:
            return candidate

    raise FileNotFoundError(
        f"Could not find train/valid folders under {download_dir}. "
        "Check Kaggle dataset layout or credentials."
    )


def _find_split_dir(root: Path, split: str) -> Path | None:
    direct = root / split
    if direct.is_dir():
        return direct
    for path in root.rglob(split):
        if path.is_dir() and path.name == split:
            parent_classes = [c for c in path.iterdir() if c.is_dir()]
            if parent_classes:
                return path
    return None


def _convert_classification_tree_to_yolo(raw_root: Path, target_root: Path) -> None:
    mapping = {
        "train": "train",
        "valid": "valid",
        "val": "valid",
        "validation": "valid",
        "test": "test",
    }
    converted = 0
    skipped_joker = 0

    for split_name, yolo_split in mapping.items():
        split_dir = _find_split_dir(raw_root, split_name)
        if split_dir is None:
            continue

        images_dir = target_root / yolo_split / "images"
        labels_dir = target_root / yolo_split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            yolo_name = kaggle_folder_to_yolo_name(class_dir.name)
            if yolo_name is None:
                skipped_joker += 1
                continue
            class_id = CLASS_NAME_TO_ID[yolo_name]

            for image_path in class_dir.iterdir():
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                stem = f"{yolo_split}_{class_id:02d}_{converted:06d}"
                dest_image = images_dir / f"{stem}{image_path.suffix.lower()}"
                shutil.copy2(image_path, dest_image)
                label_path = labels_dir / f"{stem}.txt"
                label_path.write_text(f"{class_id} 0.5 0.5 1.0 1.0\n", encoding="utf-8")
                converted += 1

    if converted == 0:
        raise RuntimeError(f"No images converted from Kaggle tree at {raw_root}")

    if skipped_joker:
        logger.info("Skipped %s non-standard class folders (e.g. joker)", skipped_joker)

    download_cache = target_root / "_download"
    if download_cache.exists():
        shutil.rmtree(download_cache)


def _write_data_yaml(path: Path) -> None:
    # Ultralytics ``model.val(split="valid")`` resolves ``data["valid"]``; keep ``val`` for YOLO v8 defaults.
    dataset_root = path.parent
    payload = {
        "path": str(dataset_root),
        "train": "train/images",
        "val": "valid/images",
        "valid": "valid/images",
        "test": "test/images",
        "nc": len(YOLO_CLASS_NAMES),
        "names": YOLO_CLASS_NAMES,
        "source": KAGGLE_DATASET,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _ensure_data_yaml_split_aliases(data_yaml: Path) -> None:
    """Older caches only had ``val``; configs use ``validate.split: valid`` → Ultralytics needs ``valid`` key."""
    if not data_yaml.exists():
        return
    raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return
    changed = False
    if "path" not in raw:
        raw["path"] = str(data_yaml.parent)
        changed = True
    if "valid" not in raw and raw.get("val"):
        raw["valid"] = raw["val"]
        changed = True
    if "val" not in raw and raw.get("valid"):
        raw["val"] = raw["valid"]
        changed = True
    if changed:
        data_yaml.write_text(
            yaml.safe_dump(raw, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )


def default_kaggle_root(project_root: Path | None = None) -> Path:
    root = project_root or Path.cwd()
    return (root / "dataset" / "kaggle").resolve()


def ensure_dataset_if_needed(dataset_root: Path) -> None:
    """Download and convert Kaggle dataset when ``dataset_root`` is the kaggle cache."""
    if dataset_root.name == "kaggle":
        ensure_kaggle_yolo_dataset(dataset_root)
