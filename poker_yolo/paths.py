"""Shared path resolution for configs and trained weights."""

from __future__ import annotations

from pathlib import Path

from poker_yolo.config import Config, classify_weights_path


def resolve_config_path(*candidates: str | Path) -> Path:
    """Return the first existing config path, or the first candidate."""
    paths = [Path(c) for c in candidates]
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def default_pipeline_config() -> Path:
    return resolve_config_path("configs/default.yaml", "/app/configs/default.yaml")


def default_hands_config() -> Path:
    return resolve_config_path("configs/hands.yaml", "/app/configs/hands.yaml")


def default_weights(config: Config) -> Path:
    """Locate ``best.pt`` for the configured run (Ultralytics classify layout)."""
    candidates = [
        classify_weights_path(config.project, config.name),
        Path("/app") / classify_weights_path(config.project, config.name),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]
