from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from poker_yolo.classify_results import extract_top_classes
from poker_yolo.config import Config
from poker_yolo.benchmark_grids import normalize_confusion_matrix, normalize_outcomes_by_hand
from poker_yolo.hands import HandValidationError, evaluate_hand
from poker_yolo.infer_source import IMAGE_SUFFIXES

logger = logging.getLogger(__name__)

HANDS_BENCHMARK_SOURCE = "dataset/test/images"


@dataclass
class HandsBenchmarkAnalysis:
    samples: list[dict[str, Any]]
    aggregate: dict[str, float]
    predicted_class_counts: dict[str, int]
    outcomes_by_hand: dict[str, dict[str, int]]
    confusion: dict[str, dict[str, int]]


def _load_benchmark_yaml(config: Config) -> dict:
    import yaml

    return yaml.safe_load(config.val_data_yaml.read_text(encoding="utf-8"))


def hands_test_images(config: Config) -> list[Path]:
    """All benchmark images under ``dataset/test/images`` (via val/benchmark yaml)."""
    data = _load_benchmark_yaml(config)
    test_dir = config.val_dataset_root / data["test"]
    return sorted(p for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _benchmark_labels_dir(config: Config) -> Path:
    data = _load_benchmark_yaml(config)
    test_path = Path(str(data["test"]))
    labels_path = test_path.with_name("labels") if test_path.name == "images" else test_path.parent / "labels"
    return config.val_dataset_root / labels_path


def _benchmark_class_names(config: Config) -> list[str]:
    names = _load_benchmark_yaml(config).get("names", [])
    return list(names) if isinstance(names, list) else []


def _ground_truth_hand(image_path: Path, labels_dir: Path, class_names: list[str]) -> str | None:
    label_file = labels_dir / f"{image_path.stem}.txt"
    if not label_file.is_file():
        return None
    cards: list[str] = []
    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            class_id = int(parts[0])
        except ValueError:
            continue
        if 0 <= class_id < len(class_names):
            cards.append(str(class_names[class_id]))
    evaluation = evaluate_hand(cards)
    if isinstance(evaluation, HandValidationError):
        return None
    return evaluation.hand_type


def _result_image_path(result: Any) -> Path | None:
    path = getattr(result, "path", None)
    if not path:
        return None
    return Path(str(path)).resolve()


def infer_results_cover_benchmark(config: Config, infer_results: list[Any]) -> bool:
    """True when every hands-benchmark image has a matching pipeline infer result."""
    if not infer_results:
        return False
    by_path = {
        resolved
        for result in infer_results
        if (resolved := _result_image_path(result)) is not None
    }
    images = hands_test_images(config)
    return bool(images) and all(img.resolve() in by_path for img in images)


def _order_infer_results_for_benchmark(
    config: Config,
    infer_results: list[Any],
) -> list[Any] | None:
    by_path = {
        _result_image_path(result): result
        for result in infer_results
        if _result_image_path(result) is not None
    }
    images = hands_test_images(config)
    ordered: list[Any] = []
    for image_path in images:
        result = by_path.get(image_path.resolve())
        if result is None:
            return None
        ordered.append(result)
    return ordered


def _build_hands_benchmark_analysis(
    config: Config,
    output_dir: Path,
    images: list[Path],
    results: list[Any],
    *,
    n_samples: int,
    seed: int,
    reports_base_url: str,
    predict_conf: float,
) -> HandsBenchmarkAnalysis:
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    labels_dir = _benchmark_labels_dir(config)
    card_names = _benchmark_class_names(config)

    per_image: list[tuple[Path, str | None, list[dict[str, Any]]]] = []
    class_counter: Counter[str] = Counter()
    confidences: list[float] = []
    outcomes_by_hand: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "incorrect": 0})
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    evaluated = 0
    correct_total = 0

    for image_path, result in zip(images, results, strict=True):
        true_hand = _ground_truth_hand(image_path, labels_dir, card_names)
        top_classes = extract_top_classes(result)
        per_image.append((image_path, true_hand, top_classes))
        if top_classes:
            pred_hand = top_classes[0]["class_name"]
            class_counter[pred_hand] += 1
            confidences.append(float(top_classes[0]["confidence"]))
            if true_hand is not None:
                evaluated += 1
                is_correct = pred_hand == true_hand
                if is_correct:
                    correct_total += 1
                    outcomes_by_hand[true_hand]["correct"] += 1
                else:
                    outcomes_by_hand[true_hand]["incorrect"] += 1
                confusion[true_hand][pred_hand] += 1

    images_with_pred = sum(1 for _, _, preds in per_image if preds)
    n_images = len(images)

    aggregate: dict[str, float] = {
        "hands_benchmark_images": float(n_images),
        "hands_benchmark_images_with_predictions": float(images_with_pred),
        "hands_benchmark_top1_conf_avg": sum(confidences) / max(len(confidences), 1),
        "hands_benchmark_unique_classes": float(len(class_counter)),
        "hands_benchmark_prediction_rate": images_with_pred / max(n_images, 1),
        "hands_benchmark_infer_conf": float(predict_conf),
        "hands_benchmark_evaluated_images": float(evaluated),
        "hands_benchmark_correct_total": float(correct_total),
        "hands_benchmark_incorrect_total": float(max(evaluated - correct_total, 0)),
        "hands_benchmark_accuracy": correct_total / max(evaluated, 1),
    }

    rng = random.Random(seed)
    preview_indices = sorted(rng.sample(range(n_images), min(n_samples, n_images)))

    samples: list[dict[str, Any]] = []
    for preview_idx, image_idx in enumerate(preview_indices):
        image_path, true_hand, top_classes = per_image[image_idx]
        out_name = f"sample_{preview_idx}.jpg"
        out_path = preview_dir / out_name
        results[image_idx].save(str(out_path))
        pred_hand = top_classes[0]["class_name"] if top_classes else None

        meta = {
            "index": preview_idx,
            "source_image": str(image_path),
            "source_dataset": HANDS_BENCHMARK_SOURCE,
            "preview_image": str(out_path),
            "preview_url": f"{reports_base_url.rstrip('/')}/preview/{out_name}",
            "predictions_count": len(top_classes),
            "top_classes": top_classes,
            "true_combo": true_hand,
            "predicted_combo": pred_hand,
            "prediction_correct": pred_hand == true_hand if true_hand and pred_hand else None,
            "predicted_confidence": top_classes[0]["confidence"] if top_classes else 0.0,
        }
        (preview_dir / f"sample_{preview_idx}.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        samples.append(meta)
        logger.info(
            "Saved hands benchmark preview %s (pred=%s true=%s, source=%s)",
            out_path,
            pred_hand,
            true_hand,
            image_path.name,
        )

    confusion_plain = normalize_confusion_matrix(
        {true: dict(preds) for true, preds in confusion.items()}
    )
    outcomes_plain = normalize_outcomes_by_hand(
        {hand: dict(counts) for hand, counts in outcomes_by_hand.items()}
    )

    manifest = preview_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_dataset": HANDS_BENCHMARK_SOURCE,
                "images_analyzed": n_images,
                "infer_conf": predict_conf,
                "aggregate": aggregate,
                "class_counts": dict(class_counter),
                "outcomes_by_hand": outcomes_plain,
                "confusion": confusion_plain,
                "samples": samples,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info(
        "Hands benchmark: %d/%d correct (%.1f%%) -> %s",
        correct_total,
        evaluated,
        100.0 * aggregate["hands_benchmark_accuracy"],
        manifest,
    )
    return HandsBenchmarkAnalysis(
        samples=samples,
        aggregate=aggregate,
        predicted_class_counts=dict(class_counter),
        outcomes_by_hand=outcomes_plain,
        confusion=confusion_plain,
    )


def analyze_hands_benchmark_from_infer_results(
    config: Config,
    output_dir: Path,
    infer_results: list[Any],
    *,
    n_samples: int = 3,
    seed: int = 42,
    reports_base_url: str = "http://localhost:8088",
    conf: float | None = None,
) -> HandsBenchmarkAnalysis | None:
    """Score hands benchmark from existing infer results (no second YOLO predict)."""
    ordered = _order_infer_results_for_benchmark(config, infer_results)
    if ordered is None:
        return None
    predict_conf = config.infer_conf if conf is None else conf
    images = hands_test_images(config)
    logger.info(
        "Hands benchmark: reusing %d pipeline infer results (skip second predict)",
        len(images),
    )
    return _build_hands_benchmark_analysis(
        config,
        output_dir,
        images,
        ordered,
        n_samples=n_samples,
        seed=seed,
        reports_base_url=reports_base_url,
        predict_conf=predict_conf,
    )


def analyze_hands_benchmark(
    config: Config,
    weights: Path,
    output_dir: Path,
    *,
    n_samples: int = 3,
    seed: int = 42,
    reports_base_url: str = "http://localhost:8088",
    conf: float | None = None,
    predict_device: str | None = None,
) -> HandsBenchmarkAnalysis:
    """Run inference on hands benchmark; compare to combo derived from YOLO card labels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    images = hands_test_images(config)
    if not images:
        logger.warning("No hands benchmark images in %s", HANDS_BENCHMARK_SOURCE)
        return HandsBenchmarkAnalysis([], {}, {}, {}, {})

    predict_conf = config.infer_conf if conf is None else conf
    device = predict_device if predict_device is not None else config.device
    logger.info(
        "Hands benchmark: predicting %d images on %s (may take several minutes on CPU)",
        len(images),
        device,
    )

    model = YOLO(str(weights), task=config.task)
    results = model.predict(
        task=config.task,
        source=[str(p) for p in images],
        imgsz=config.imgsz,
        conf=predict_conf,
        iou=config.infer_iou,
        device=device,
        save=False,
        verbose=False,
    )
    del model
    logger.info("Hands benchmark: predict finished, scoring %d images", len(images))

    return _build_hands_benchmark_analysis(
        config,
        output_dir,
        images,
        results,
        n_samples=n_samples,
        seed=seed,
        reports_base_url=reports_base_url,
        predict_conf=predict_conf,
    )
