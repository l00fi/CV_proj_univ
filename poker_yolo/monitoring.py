from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger(__name__)


def get_gpu_stats() -> tuple[float, float]:
    """Return (gpu_util_percent, gpu_memory_mb). Zeroes when GPU unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0, 0.0
        util = 0.0
        try:
            import pynvml  # noqa: PLC0415

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            util = float(rates.gpu)
        except Exception:
            util = 0.0
        mem_mb = float(torch.cuda.max_memory_allocated() / (1024**2))
        return util, mem_mb
    except Exception:
        return 0.0, 0.0


@dataclass
class ResourceMonitor:
    """Sample CPU/RAM/GPU usage during a pipeline phase."""

    phase: str
    interval_sec: float = 2.0
    samples: list[dict[str, float]] = field(default_factory=list)
    _thread: threading.Thread | None = field(default=None, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)

    def sample_once(self) -> dict[str, float]:
        gpu_util, gpu_mem = get_gpu_stats()
        sample = {
            "cpu_pct": float(psutil.cpu_percent(interval=None)),
            "ram_mb": float(psutil.virtual_memory().used / (1024**2)),
            "gpu_util_pct": gpu_util,
            "gpu_mem_mb": gpu_mem,
        }
        self.samples.append(sample)
        return sample

    def start_background(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop_background(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval_sec)

    def summary(self) -> dict[str, float]:
        if not self.samples:
            self.sample_once()
        cpu = [s["cpu_pct"] for s in self.samples]
        ram = [s["ram_mb"] for s in self.samples]
        gpu_u = [s["gpu_util_pct"] for s in self.samples]
        gpu_m = [s["gpu_mem_mb"] for s in self.samples]

        def _avg(vals: list[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        prefix = self.phase
        return {
            f"{prefix}_cpu_avg_pct": _avg(cpu),
            f"{prefix}_cpu_peak_pct": max(cpu) if cpu else 0.0,
            f"{prefix}_ram_avg_mb": _avg(ram),
            f"{prefix}_ram_peak_mb": max(ram) if ram else 0.0,
            f"{prefix}_gpu_util_avg_pct": _avg(gpu_u),
            f"{prefix}_gpu_util_peak_pct": max(gpu_u) if gpu_u else 0.0,
            f"{prefix}_gpu_mem_peak_mb": max(gpu_m) if gpu_m else 0.0,
            f"{prefix}_resource_samples": float(len(self.samples)),
        }


def compute_augmentation_summary(
    aug_config: Any,
    train_images: int,
) -> dict[str, Any]:
    """Estimate augmentation exposure vs real dataset size."""
    yolo = aug_config.to_ultralytics_args()
    alb = dict(aug_config.albumentations)

    mosaic_p = float(yolo.get("mosaic", 0.0))
    mixup_p = float(yolo.get("mixup", 0.0))
    copy_paste_p = float(yolo.get("copy_paste", 0.0))
    cutmix_p = float(yolo.get("cutmix", 0.0))
    fliplr_p = float(yolo.get("fliplr", 0.0))

    # Expected extra synthetic exposure per real sample (heuristic for YOLO pipeline)
    yolo_synthetic = mosaic_p * 3.0 + mixup_p + copy_paste_p + cutmix_p + fliplr_p * 0.5
    alb_expected = sum(alb.values()) / max(len(alb), 1) if alb else 0.0
    synthetic_to_real_ratio = yolo_synthetic + alb_expected
    effective_views = train_images * (1.0 + synthetic_to_real_ratio)

    return {
        "train_images_real": train_images,
        "augmentations_enabled": aug_config.enabled,
        "synthetic_to_real_ratio": round(synthetic_to_real_ratio, 4),
        "estimated_augmented_views_per_epoch": round(effective_views, 1),
        "yolo_probabilities": {
            "mosaic": mosaic_p,
            "mixup": mixup_p,
            "copy_paste": copy_paste_p,
            "cutmix": cutmix_p,
            "fliplr": fliplr_p,
            "degrees": float(yolo.get("degrees", 0.0)),
            "scale": float(yolo.get("scale", 0.0)),
        },
        "albumentations_probabilities": alb,
        "albumentations_transforms_count": len(alb),
    }


def compute_dataset_stats(dataset_root: Path, data_yaml: Path) -> dict[str, Any]:
    import yaml

    raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = dataset_root
    train_img = root / raw.get("train", "train/images")
    val_key = "val" if "val" in raw else "valid"
    val_img = root / raw.get(val_key, raw.get("valid", "valid/images"))
    test_img = root / raw.get("test", "test/images")
    train_count = len(list(train_img.glob("*"))) if train_img.exists() else 0
    val_count = len(list(val_img.glob("*"))) if val_img.exists() else 0
    test_count = len(list(test_img.glob("*"))) if test_img.exists() else 0
    names = raw.get("names", [])
    return {
        "train_images": train_count,
        "val_images": val_count,
        "test_images": test_count,
        "num_classes": len(names),
        "class_names": names,
    }


def compute_production_metrics(
    weights_path: Path | None,
    train_duration_sec: float,
    val_duration_sec: float,
    resource_metrics: dict[str, float],
    val_metrics: dict[str, float],
    dataset_stats: dict[str, Any],
    aug_summary: dict[str, Any],
    infer_duration_sec: float = 0.0,
    infer_metrics: dict[str, float] | None = None,
) -> dict[str, float]:
    """Aggregate production-oriented KPIs for dashboards and reports."""
    metrics: dict[str, float] = {
        **resource_metrics,
        "train_duration_sec": train_duration_sec,
        "val_duration_sec": val_duration_sec,
        "infer_duration_sec": infer_duration_sec,
        "pipeline_duration_sec": train_duration_sec + val_duration_sec + infer_duration_sec,
        "train_images": float(dataset_stats.get("train_images", 0)),
        "test_images": float(dataset_stats.get("test_images", 0)),
        "num_classes": float(dataset_stats.get("num_classes", 0)),
        "synthetic_to_real_ratio": float(aug_summary.get("synthetic_to_real_ratio", 0.0)),
        "estimated_augmented_views_per_epoch": float(
            aug_summary.get("estimated_augmented_views_per_epoch", 0.0)
        ),
    }
    for key, value in val_metrics.items():
        metrics[f"val_{key}"] = float(value)
    if infer_metrics:
        metrics.update({k: float(v) for k, v in infer_metrics.items() if v is not None})
    if weights_path and weights_path.exists():
        metrics["model_size_mb"] = weights_path.stat().st_size / (1024**2)
    return metrics
