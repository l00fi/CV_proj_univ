"""Tests for on-the-fly Albumentations pipeline."""

from __future__ import annotations

import albumentations as A
import numpy as np
import pytest

from poker_yolo.augmentations import AugmentationConfig, build_albumentations


def test_augmentation_config_from_dict_full() -> None:
    raw = {
        "enabled": True,
        "mosaic": 0.9,
        "mixup": 0.2,
        "copy_paste": 0.4,
        "albumentations": {"blur": 0.2, "to_gray": 0.05},
    }
    config = AugmentationConfig.from_dict(raw)

    assert config.enabled is True
    assert config.mosaic == 0.9
    assert config.copy_paste == 0.4
    assert config.albumentations["blur"] == 0.2


def test_augmentation_config_from_dict_none_uses_train_fallback() -> None:
    train = {"mosaic": 0.7, "mixup": 0.12, "degrees": 5.0}
    config = AugmentationConfig.from_dict(None, train_fallback=train)

    assert config.mosaic == 0.7
    assert config.mixup == 0.12
    assert config.degrees == 5.0


def test_to_ultralytics_args_keys() -> None:
    config = AugmentationConfig(mosaic=1.0, mixup=0.15, copy_paste=0.3)
    args = config.to_ultralytics_args()

    expected_keys = {
        "mosaic", "mixup", "copy_paste", "cutmix", "fliplr", "flipud",
        "degrees", "translate", "scale", "shear", "perspective",
        "hsv_h", "hsv_s", "hsv_v",
    }
    assert expected_keys == set(args.keys())


def test_to_mlflow_params_prefixes_albumentations() -> None:
    config = AugmentationConfig(albumentations={"blur": 0.1, "clahe": 0.2})
    params = config.to_mlflow_params()

    assert params["aug_enabled"] is True
    assert params["aug_alb_blur"] == 0.1
    assert params["aug_alb_clahe"] == 0.2


def test_build_albumentations_disabled_returns_empty() -> None:
    config = AugmentationConfig(enabled=False, albumentations={"blur": 1.0})
    assert build_albumentations(config) == []


def test_build_albumentations_skips_zero_probability() -> None:
    config = AugmentationConfig(
        albumentations={"blur": 0.0, "brightness_contrast": 0.5},
    )
    transforms = build_albumentations(config)

    assert len(transforms) == 1
    assert isinstance(transforms[0], A.RandomBrightnessContrast)


def test_build_albumentations_default_yaml_count(default_config_path) -> None:
    from poker_yolo.config import Config

    config = Config.from_yaml(default_config_path)
    transforms = build_albumentations(config.augmentations)
    assert len(transforms) == 11


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_albumentations_apply_on_the_fly_with_bboxes(seed: int) -> None:
    """Transforms must run per-sample without mutating disk data."""
    config = AugmentationConfig(
        albumentations={
            "blur": 0.5,
            "brightness_contrast": 0.8,
            "gauss_noise": 0.5,
            "coarse_dropout": 0.5,
            "shift_scale_rotate": 0.5,
        }
    )
    transforms = build_albumentations(config)
    pipeline = A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    bboxes = [[0.5, 0.5, 0.2, 0.3], [0.3, 0.7, 0.15, 0.25]]
    class_labels = [0, 1]

    result = pipeline(image=image, bboxes=bboxes, class_labels=class_labels)

    assert result["image"].shape == image.shape
    assert len(result["bboxes"]) == len(bboxes)
    assert len(result["class_labels"]) == len(class_labels)


def test_ultralytics_albumentations_wrapper_accepts_pipeline() -> None:
    from ultralytics.data.augment import Albumentations

    config = AugmentationConfig(albumentations={"blur": 0.1, "brightness_contrast": 0.5})
    transforms = build_albumentations(config)
    wrapper = Albumentations(p=1.0, transforms=transforms)

    assert wrapper.transform is not None

    labels = {
        "img": np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
        "cls": np.array([0]),
        "instances": _make_instances([[0.5, 0.5, 0.2, 0.3]]),
    }
    augmented = wrapper(labels)
    assert augmented["img"].shape == labels["img"].shape


def _make_instances(bboxes: list[list[float]]):
    import torch
    from ultralytics.utils.instance import Instances

    n = len(bboxes)
    boxes = torch.tensor(bboxes, dtype=torch.float32).reshape(n, 4)
    return Instances(bboxes=boxes, bbox_format="xywh", normalized=True)
