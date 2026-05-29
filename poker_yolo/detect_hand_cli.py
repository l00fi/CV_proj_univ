"""CLI entry point for poker hand classification on an image."""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

from poker_yolo.classify_results import predict_top_classes
from poker_yolo.config import Config
from poker_yolo.hands import HAND_NAMES_RU
from poker_yolo.paths import default_hands_config, default_weights
from poker_yolo.predictions import hands_test_images

logger = logging.getLogger(__name__)


def _pick_test_image(config: Config, seed: int | None) -> Path:
    images = hands_test_images(config)
    if not images:
        raise FileNotFoundError("No test images in dataset/test/images")
    return random.Random(seed).choice(images)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Определить покерную комбинацию на изображении с помощью обученной модели",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML-конфиг (по умолчанию configs/hands.yaml)",
    )
    parser.add_argument("--weights", type=Path, default=None, help="Путь к best.pt")
    parser.add_argument("--image", type=Path, default=None, help="Изображение (default: random test)")
    parser.add_argument("--seed", type=int, default=None, help="Seed для случайного test-изображения")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config_path = args.config or default_hands_config()
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        return 1

    config = Config.from_yaml(config_path)
    weights = args.weights or default_weights(config)
    if not weights.exists():
        logger.error(
            "Weights not found: %s — train first: uv run poker-yolo --config configs/local.yaml train",
            weights,
        )
        return 1

    try:
        image_path = args.image or _pick_test_image(config, args.seed)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if not image_path.exists():
        logger.error("Image not found: %s", image_path)
        return 1

    predictions = predict_top_classes(config, weights, image_path)

    print(f"Изображение: {image_path}")
    print(f"Модель:      {weights}")
    if not predictions:
        print("Комбинация:  не определена")
        return 1
    best_label, best_conf = predictions[0]
    print(f"Комбинация:  {HAND_NAMES_RU.get(best_label, best_label)} ({best_conf:.1%})")
    print("Top-5:")
    for label, conf in predictions:
        print(f"  - {HAND_NAMES_RU.get(label, label)}: {conf:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
