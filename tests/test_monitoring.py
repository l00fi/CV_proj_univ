"""Tests for resource monitoring and production metrics."""

from __future__ import annotations

from poker_yolo.augmentations import AugmentationConfig
from poker_yolo.monitoring import (
    ResourceMonitor,
    compute_augmentation_summary,
    compute_dataset_stats,
    compute_production_metrics,
)


def test_resource_monitor_summary() -> None:
    monitor = ResourceMonitor("train")
    monitor.samples = [
        {"cpu_pct": 40.0, "ram_mb": 1000.0, "gpu_util_pct": 0.0, "gpu_mem_mb": 0.0},
        {"cpu_pct": 60.0, "ram_mb": 1200.0, "gpu_util_pct": 10.0, "gpu_mem_mb": 256.0},
    ]
    summary = monitor.summary()
    assert summary["train_cpu_avg_pct"] == 50.0
    assert summary["train_cpu_peak_pct"] == 60.0
    assert summary["train_ram_peak_mb"] == 1200.0
    assert summary["train_gpu_util_peak_pct"] == 10.0


def test_compute_augmentation_summary() -> None:
    config = AugmentationConfig(
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.3,
        albumentations={"blur": 0.1, "brightness_contrast": 0.5},
    )
    summary = compute_augmentation_summary(config, train_images=109, task="detect")
    assert summary["train_images_real"] == 109
    assert summary["synthetic_to_real_ratio"] > 0
    assert summary["estimated_augmented_views_per_epoch"] > 109
    assert "mosaic" in summary["yolo_probabilities"]
    assert summary["albumentations_transforms_count"] == 2


def test_compute_augmentation_summary_classify_skips_detect_block() -> None:
    config = AugmentationConfig(mosaic=1.0, mixup=0.2)
    summary = compute_augmentation_summary(config, train_images=50, task="classify")
    assert summary["train_images_real"] == 50
    assert summary["augmentations_enabled"] is False
    assert "synthetic_to_real_ratio" not in summary


def test_compute_dataset_stats(project_root) -> None:
    stats = compute_dataset_stats(project_root / "dataset", project_root / "dataset" / "data.yaml")
    assert stats["train_images"] > 0
    assert stats["test_images"] > 0
    assert stats["num_classes"] == 52


def test_compute_production_metrics(tmp_path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x" * 1024)
    metrics = compute_production_metrics(
        weights_path=weights,
        train_duration_sec=100.0,
        val_duration_sec=20.0,
        resource_metrics={"train_cpu_avg_pct": 55.0, "val_cpu_avg_pct": 30.0},
        val_metrics={"map50": 0.5, "f1": 0.4},
        dataset_stats={"train_images": 109, "test_images": 31, "num_classes": 52},
        aug_summary={"synthetic_to_real_ratio": 2.5, "estimated_augmented_views_per_epoch": 300.0},
    )
    assert metrics["pipeline_duration_sec"] == 120.0
    assert metrics["train_cpu_avg_pct"] == 55.0
    assert metrics["val_map50"] == 0.5
    assert metrics["model_size_mb"] > 0
