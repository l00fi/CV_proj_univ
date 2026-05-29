"""Resolve inference image sources for Ultralytics predict."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def _dir_has_direct_images(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    return any(_is_image(entry) for entry in directory.iterdir())


def _skip_kaggle_train(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return "kaggle" in parts and "train" in parts


def collect_images_under(root: Path, *, skip_kaggle_train: bool = True) -> list[str]:
    """Collect image paths under ``root``, optionally skipping huge Kaggle train split."""
    images: list[str] = []
    for path in sorted(root.rglob("*")):
        if not _is_image(path):
            continue
        if skip_kaggle_train and _skip_kaggle_train(path):
            continue
        images.append(str(path))
    return images


def resolve_infer_source(source: Path) -> Path | list[str]:
    """
    Ultralytics accepts a directory only when it contains images directly.

    ``infer.source: dataset`` points at a folder with subdirs (``test/images``,
    ``kaggle/...``). Expand it to a file list for predict.
    """
    if not source.exists():
        return source
    if source.is_file():
        return source
    if _dir_has_direct_images(source):
        return source

    images = collect_images_under(source)
    if images:
        logger.info(
            "Expanded inference source %s -> %d images (excluding kaggle/train)",
            source,
            len(images),
        )
        return images

    return source
