from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import albumentations as A


@dataclass
class AugmentationConfig:
    """On-the-fly augmentation settings for training."""

    enabled: bool = True
    mosaic: float = 1.0
    mixup: float = 0.15
    copy_paste: float = 0.3
    cutmix: float = 0.1
    fliplr: float = 0.5
    flipud: float = 0.0
    degrees: float = 15.0
    translate: float = 0.15
    scale: float = 0.5
    shear: float = 5.0
    perspective: float = 0.0005
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    albumentations: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None, train_fallback: dict[str, Any] | None = None) -> AugmentationConfig:
        train_fallback = train_fallback or {}
        if not raw:
            return cls(
                mosaic=float(train_fallback.get("mosaic", 1.0)),
                mixup=float(train_fallback.get("mixup", 0.15)),
                degrees=float(train_fallback.get("degrees", 15.0)),
                translate=float(train_fallback.get("translate", 0.15)),
                scale=float(train_fallback.get("scale", 0.5)),
                hsv_h=float(train_fallback.get("hsv_h", 0.015)),
                hsv_s=float(train_fallback.get("hsv_s", 0.7)),
                hsv_v=float(train_fallback.get("hsv_v", 0.4)),
            )

        alb = raw.get("albumentations", {})
        return cls(
            enabled=bool(raw.get("enabled", True)),
            mosaic=float(raw.get("mosaic", train_fallback.get("mosaic", 1.0))),
            mixup=float(raw.get("mixup", train_fallback.get("mixup", 0.15))),
            copy_paste=float(raw.get("copy_paste", 0.3)),
            cutmix=float(raw.get("cutmix", 0.1)),
            fliplr=float(raw.get("fliplr", 0.5)),
            flipud=float(raw.get("flipud", 0.0)),
            degrees=float(raw.get("degrees", train_fallback.get("degrees", 15.0))),
            translate=float(raw.get("translate", train_fallback.get("translate", 0.15))),
            scale=float(raw.get("scale", train_fallback.get("scale", 0.5))),
            shear=float(raw.get("shear", 5.0)),
            perspective=float(raw.get("perspective", 0.0005)),
            hsv_h=float(raw.get("hsv_h", train_fallback.get("hsv_h", 0.015))),
            hsv_s=float(raw.get("hsv_s", train_fallback.get("hsv_s", 0.7))),
            hsv_v=float(raw.get("hsv_v", train_fallback.get("hsv_v", 0.4))),
            albumentations=alb,
        )

    def to_ultralytics_args(self) -> dict[str, float]:
        return {
            "mosaic": self.mosaic,
            "mixup": self.mixup,
            "copy_paste": self.copy_paste,
            "cutmix": self.cutmix,
            "fliplr": self.fliplr,
            "flipud": self.flipud,
            "degrees": self.degrees,
            "translate": self.translate,
            "scale": self.scale,
            "shear": self.shear,
            "perspective": self.perspective,
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
        }

    def to_mlflow_params(self) -> dict[str, float | bool]:
        params = {"aug_enabled": self.enabled, **self.to_ultralytics_args()}
        for name, prob in self.albumentations.items():
            params[f"aug_alb_{name}"] = float(prob)
        return params


def build_albumentations(config: AugmentationConfig) -> list[A.BasicTransform]:
    """Build Albumentations transforms applied on-the-fly inside the YOLO dataloader."""
    if not config.enabled:
        return []

    alb = config.albumentations
    transforms: list[A.BasicTransform] = []

    if (p := alb.get("blur", 0.0)) > 0:
        transforms.append(
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                ],
                p=p,
            )
        )

    if (p := alb.get("motion_blur", 0.0)) > 0:
        transforms.append(A.MotionBlur(blur_limit=7, p=p))

    if (p := alb.get("gauss_noise", 0.0)) > 0:
        transforms.append(A.GaussNoise(std_range=(0.02, 0.08), p=p))

    if (p := alb.get("brightness_contrast", 0.0)) > 0:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=0.25,
                contrast_limit=0.25,
                p=p,
            )
        )

    if (p := alb.get("hue_saturation", 0.0)) > 0:
        transforms.append(
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=25,
                val_shift_limit=20,
                p=p,
            )
        )

    if (p := alb.get("clahe", 0.0)) > 0:
        transforms.append(A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=p))

    if (p := alb.get("image_compression", 0.0)) > 0:
        transforms.append(A.ImageCompression(quality_range=(60, 95), p=p))

    if (p := alb.get("coarse_dropout", 0.0)) > 0:
        transforms.append(
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                fill=0,
                p=p,
            )
        )

    if (p := alb.get("shift_scale_rotate", 0.0)) > 0:
        transforms.append(
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.88, 1.12),
                rotate=(-8, 8),
                p=p,
            )
        )

    if (p := alb.get("optical_distortion", 0.0)) > 0:
        transforms.append(A.OpticalDistortion(distort_limit=0.12, p=p))

    if (p := alb.get("to_gray", 0.0)) > 0:
        transforms.append(A.ToGray(p=p))

    return transforms
