from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

from poker_yolo.config import Config

logger = logging.getLogger(__name__)


def save_sample_predictions(
    config: Config,
    weights: Path,
    output_dir: Path,
    n_samples: int = 3,
    seed: int = 42,
    reports_base_url: str = "http://localhost:8088",
) -> list[dict[str, Any]]:
    """Run inference on N test images and save annotated previews for reports/Grafana."""
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    data = yaml.safe_load(config.data_yaml.read_text(encoding="utf-8"))
    class_names: list[str] = data.get("names", [])
    test_dir = config.dataset_root / data["test"]
    images = sorted(
        [p for p in test_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    )
    if not images:
        logger.warning("No test images found in %s", test_dir)
        return []

    rng = random.Random(seed)
    selected = rng.sample(images, min(n_samples, len(images)))

    model = YOLO(str(weights))
    samples: list[dict[str, Any]] = []

    for idx, image_path in enumerate(selected):
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
        out_name = f"sample_{idx}.jpg"
        out_path = preview_dir / out_name
        result.save(str(out_path))

        detections = []
        boxes = result.boxes
        if boxes is not None and len(boxes):
            for box in boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                detections.append(
                    {"class_id": cls_id, "class_name": name, "confidence": conf, "bbox_xyxy": xyxy}
                )

        meta = {
            "index": idx,
            "source_image": str(image_path),
            "preview_image": str(out_path),
            "preview_url": f"{reports_base_url.rstrip('/')}/preview/{out_name}",
            "detections_count": len(detections),
            "detections": detections,
            "top_classes": _top_classes(detections),
        }
        meta_path = preview_dir / f"sample_{idx}.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        samples.append(meta)
        logger.info("Saved prediction preview %s (%d detections)", out_path, len(detections))

    manifest = preview_dir / "manifest.json"
    manifest.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    return samples


def _top_classes(detections: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    ranked = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    return [
        {"class_name": d["class_name"], "confidence": d["confidence"]}
        for d in ranked[:limit]
    ]
