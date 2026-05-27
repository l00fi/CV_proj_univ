"""CLI entry point for poker hand detection on an image."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

from poker_yolo.cli import _default_config, _default_weights
from poker_yolo.config import Config
from poker_yolo.hands import (
    HandEvaluation,
    HandValidationError,
    dedupe_detections,
    evaluate_hand,
    format_cards_ru,
)


def _load_class_names(data_yaml: Path) -> list[str]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    return list(data.get("names", []))


def _pick_test_image(config: Config, seed: int | None) -> Path:
    data = yaml.safe_load(config.data_yaml.read_text(encoding="utf-8"))
    test_dir = config.dataset_root / data["test"]
    images = sorted(
        p for p in test_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not images:
        raise FileNotFoundError(f"No test images in {test_dir}")
    rng = random.Random(seed)
    return rng.choice(images)


def _run_detection(
    config: Config,
    weights: Path,
    image_path: Path,
    class_names: list[str],
) -> list[tuple[str, float]]:
    model = YOLO(str(weights))
    results = model.predict(
        source=str(image_path),
        imgsz=config.imgsz,
        conf=config.infer_conf,
        iou=config.infer_iou,
        device=config.device,
        save=False,
        verbose=False,
    )
    result = results[0]
    detections: list[tuple[str, float]] = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return detections

    for box in boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        detections.append((name, conf))
    return detections


def analyze_image(
    config: Config,
    weights: Path,
    image_path: Path,
) -> tuple[list[tuple[str, float]], HandEvaluation | HandValidationError]:
    class_names = _load_class_names(config.data_yaml)
    raw = _run_detection(config, weights, image_path, class_names)
    deduped = dedupe_detections(raw)
    labels = [label for label, _ in deduped]
    return deduped, evaluate_hand(labels)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Определить покерную комбинацию на изображении с помощью обученной модели",
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML-конфиг")
    parser.add_argument("--weights", type=Path, default=None, help="Путь к best.pt")
    parser.add_argument("--image", type=Path, default=None, help="Изображение (default: random test)")
    parser.add_argument("--seed", type=int, default=None, help="Seed для случайного test-изображения")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config_path = args.config or _default_config()
    if not config_path.exists():
        print(f"Конфиг не найден: {config_path}", file=sys.stderr)
        return 1

    config = Config.from_yaml(config_path)
    weights = args.weights or _default_weights(config)
    if not weights.exists():
        print(
            f"Веса модели не найдены: {weights}\n"
            "Сначала обучите модель: uv run poker-yolo --config configs/local.yaml train",
            file=sys.stderr,
        )
        return 1

    try:
        image_path = args.image or _pick_test_image(config, args.seed)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not image_path.exists():
        print(f"Изображение не найдено: {image_path}", file=sys.stderr)
        return 1

    detections, result = analyze_image(config, weights, image_path)

    print(f"Изображение: {image_path}")
    print(f"Модель:      {weights}")
    if detections:
        cards_str = ", ".join(f"{label} ({conf:.0%})" for label, conf in detections)
        print(f"Карты:       {cards_str}")
    else:
        print("Карты:       не обнаружены")

    print()
    if isinstance(result, HandValidationError):
        print("Комбинация:  такой комбинации нет")
        print(f"Причина:     {result.message_ru}")
        return 1

    print(f"Комбинация:  {result.name_ru}")
    print(f"Описание:    {result.description_ru}")
    print(f"Состав:      {format_cards_ru(result.cards)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
