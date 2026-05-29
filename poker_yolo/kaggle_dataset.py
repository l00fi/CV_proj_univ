"""Download Kaggle playing-cards dataset and convert to YOLO layout for training."""

from __future__ import annotations

import logging
import random
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

KAGGLE_DATASET = "arnaudlewandowski/mandines-real-poker-hands-mrph-dataset" #"gpiosenka/cards-image-datasetclassification"
MARKER_FILENAME = ".kaggle_ready"
KAGGLE_TASK = "classify"
KAGGLE_FORMAT_VERSION = "2"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train, val, test

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

HAND_TYPE_ALIASES = {
    "highcard": "high_card",
    "onepair": "pair",
    "pair": "pair",
    "twopair": "two_pair",
    "twopairs": "two_pair",
    "threeofakind": "three_of_a_kind",
    "straight": "straight",
    "flush": "flush",
    "fullhouse": "full_house",
    "fourofakind": "four_of_a_kind",
    "straightflush": "straight_flush",
    "royalflush": "royal_flush",
}


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
    split_seed: int = 42,
    force: bool = False,
) -> Path:
    """Download (once) and build YOLO dataset under target_root. Returns path to data.yaml."""
    target_root = target_root.resolve()
    data_yaml = target_root / "data.yaml"
    marker = target_root / MARKER_FILENAME
    marker_meta = _read_marker_meta(marker)
    marker_dataset = marker_meta.get("dataset")
    marker_task = marker_meta.get("task")
    marker_format = marker_meta.get("format")

    if (
        marker.exists()
        and data_yaml.exists()
        and not force
        and marker_dataset == dataset_slug
        and marker_task == KAGGLE_TASK
        and marker_format == KAGGLE_FORMAT_VERSION
    ):
        _ensure_data_yaml_split_aliases(data_yaml)
        logger.info("Kaggle YOLO dataset already prepared: %s", target_root)
        return data_yaml
    if marker.exists() and data_yaml.exists() and not force and (
        marker_dataset != dataset_slug or marker_task != KAGGLE_TASK or marker_format != KAGGLE_FORMAT_VERSION
    ):
        logger.info(
            "Kaggle cache metadata changed (%s/%s/%s -> %s/%s/%s), rebuilding %s",
            marker_dataset or "unknown",
            marker_task or "unknown",
            marker_format or "unknown",
            dataset_slug,
            KAGGLE_TASK,
            KAGGLE_FORMAT_VERSION,
            target_root,
        )
        force = True

    if force and target_root.exists():
        for child in target_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)
    target_root.mkdir(parents=True, exist_ok=True)

    raw_root = _download_kaggle_dataset(target_root, dataset_slug)
    class_names = _prepare_classification_dataset(raw_root, target_root, split_seed=split_seed)
    _write_data_yaml(data_yaml, dataset_slug, class_names)
    _ensure_data_yaml_split_aliases(data_yaml)
    marker.write_text(
        f"dataset={dataset_slug}\n"
        f"task={KAGGLE_TASK}\n"
        f"format={KAGGLE_FORMAT_VERSION}\n",
        encoding="utf-8",
    )
    logger.info("Kaggle classification dataset ready at %s (%s classes)", target_root, len(class_names))
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
        images_root, labels_csv = _detect_mrph_layout(candidate)
        if images_root is not None and labels_csv is not None:
            return candidate
        if _find_split_dir(candidate, "train") is not None:
            return candidate

    raise FileNotFoundError(
        f"Could not find train/valid folders or MRPH layout (images+labels.csv) under {download_dir}. "
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


def _prepare_classification_dataset(raw_root: Path, target_root: Path, *, split_seed: int) -> list[str]:
    """Convert Kaggle source to Ultralytics classification layout train/val/test/<class>."""
    images_root, labels_csv = _detect_mrph_layout(raw_root)
    if images_root is not None and labels_csv is not None:
        class_names = _convert_mrph_to_classification(
            images_root=images_root,
            labels_csv=labels_csv,
            target_root=target_root,
            split_seed=split_seed,
        )
    else:
        class_names = _convert_split_tree_to_classification(raw_root, target_root)

    download_cache = target_root / "_download"
    if download_cache.exists():
        shutil.rmtree(download_cache)
    return class_names


def _detect_mrph_layout(raw_root: Path) -> tuple[Path | None, Path | None]:
    for candidate in [raw_root, *[p for p in raw_root.iterdir() if p.is_dir()]]:
        images_root = candidate / "images"
        labels_csv = candidate / "labels.csv"
        if images_root.is_dir() and labels_csv.is_file():
            return images_root, labels_csv
    return None, None


def _normalize_hand_type(text: str) -> str:
    compact = re.sub(r"[^a-z0-9]+", "", text.strip().lower())
    return HAND_TYPE_ALIASES.get(compact, text.strip().lower().replace(" ", "_"))


def _read_mrph_rows(labels_csv: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in labels_csv.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        filename, hand_type = [part.strip() for part in line.split(",", 1)]
        rows.append((filename, _normalize_hand_type(hand_type)))
    return rows


def _split_rows(rows: list[tuple[str, str]], seed: int) -> dict[str, list[tuple[str, str]]]:
    rng = random.Random(seed)
    by_class: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        by_class[row[1]].append(row)

    result: dict[str, list[tuple[str, str]]] = {"train": [], "val": [], "test": []}
    for class_name, items in sorted(by_class.items()):
        shuffled = items[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(n * DEFAULT_SPLIT_RATIOS[0])) if n >= 3 else max(1, n - 1)
        n_val = max(1, int(n * DEFAULT_SPLIT_RATIOS[1])) if n >= 10 else (1 if n - n_train > 1 else 0)
        n_test = max(0, n - n_train - n_val)
        if n_test == 0 and n > 2:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            elif n_val > 0:
                n_val -= 1

        result["train"].extend(shuffled[:n_train])
        result["val"].extend(shuffled[n_train : n_train + n_val])
        result["test"].extend(shuffled[n_train + n_val : n_train + n_val + n_test])
    return result


def _convert_mrph_to_classification(
    *,
    images_root: Path,
    labels_csv: Path,
    target_root: Path,
    split_seed: int,
) -> list[str]:
    rows = _read_mrph_rows(labels_csv)
    if not rows:
        raise RuntimeError(f"No rows found in {labels_csv}")
    split_rows = _split_rows(rows, split_seed)

    class_names = sorted({class_name for _, class_name in rows})
    copied = 0
    for split, items in split_rows.items():
        for filename, class_name in items:
            source = images_root / filename
            if not source.is_file():
                continue
            out_dir = target_root / split / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            dest = out_dir / source.name
            shutil.copy2(source, dest)
            copied += 1
    if copied == 0:
        raise RuntimeError(f"No images copied from MRPH source {images_root}")
    return class_names


def _convert_split_tree_to_classification(raw_root: Path, target_root: Path) -> list[str]:
    mapping = {"train": "train", "valid": "val", "val": "val", "validation": "val", "test": "test"}
    class_names: set[str] = set()
    copied = 0
    for split_name, out_split in mapping.items():
        split_dir = _find_split_dir(raw_root, split_name)
        if split_dir is None:
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = _normalize_hand_type(class_dir.name)
            class_names.add(class_name)
            out_dir = target_root / out_split / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for image_path in class_dir.iterdir():
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                shutil.copy2(image_path, out_dir / image_path.name)
                copied += 1
    if copied == 0:
        raise RuntimeError(f"No images converted from Kaggle tree at {raw_root}")
    return sorted(class_names)


def _write_data_yaml(path: Path, dataset_slug: str, class_names: list[str]) -> None:
    dataset_root = path.parent
    payload = {
        "path": str(dataset_root),
        "train": "train",
        "val": "val",
        "valid": "val",
        "test": "test",
        "nc": len(class_names),
        "names": class_names,
        "source": dataset_slug,
        "task": KAGGLE_TASK,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _read_marker_meta(marker: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    if not marker.exists():
        return meta
    for line in marker.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            meta[key] = value
    return meta


def normalize_data_yaml(data_yaml: Path, *, resolve_path: bool = True) -> None:
    """Ensure ``path`` and ``val``/``valid`` aliases for Ultralytics."""
    if not data_yaml.exists():
        return
    raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return
    changed = False
    if resolve_path:
        expected_path = str(data_yaml.parent.resolve())
        if raw.get("path") != expected_path:
            raw["path"] = expected_path
            changed = True
    elif "path" not in raw:
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


def ensure_hands_data_yaml(data_yaml: Path) -> None:
    """Set ``path`` to the yaml parent so splits resolve under ``dataset/``."""
    normalize_data_yaml(data_yaml, resolve_path=True)


def _ensure_data_yaml_split_aliases(data_yaml: Path) -> None:
    normalize_data_yaml(data_yaml, resolve_path=False)


def ensure_dataset_if_needed(dataset_root: Path, *, split_seed: int = 42) -> None:
    """Download and convert Kaggle dataset when ``dataset_root`` is the kaggle cache."""
    if dataset_root.name == "kaggle":
        ensure_kaggle_yolo_dataset(dataset_root, split_seed=split_seed)
