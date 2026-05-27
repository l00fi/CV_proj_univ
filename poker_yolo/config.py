from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from poker_yolo.augmentations import AugmentationConfig, build_albumentations
from poker_yolo.reporting import ReportingConfig


def _resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
    except ImportError:
        pass
    return "cpu"


@dataclass
class Config:
    data_yaml: Path
    dataset_root: Path
    model_weights: str
    imgsz: int
    epochs: int
    batch: int
    patience: int
    device: str
    workers: int
    project: str
    name: str
    seed: int
    lr0: float
    lrf: float
    momentum: float
    weight_decay: float
    warmup_epochs: int
    augmentations: AugmentationConfig
    reporting: ReportingConfig
    val_conf: float
    val_iou: float
    val_split: str
    infer_conf: float
    infer_iou: float
    infer_save_dir: Path
    mlflow_tracking_uri: str
    mlflow_experiment: str
    mlflow_run_name: str | None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_yaml(cls, path: Path, project_root: Path | None = None) -> Config:
        root = project_root or Path.cwd()
        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        data = raw["data"]
        model = raw["model"]
        train = raw["train"]
        validate = raw["validate"]
        infer = raw["infer"]
        mlflow_cfg = raw["mlflow"]

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", mlflow_cfg["tracking_uri"])
        aug = AugmentationConfig.from_dict(raw.get("augmentations"), train_fallback=train)
        reporting = ReportingConfig.from_dict(raw.get("reporting"), project_root=root)

        return cls(
            data_yaml=_resolve_path(root, data["yaml_path"]),
            dataset_root=_resolve_path(root, data["dataset_root"]),
            model_weights=model["weights"],
            imgsz=model["imgsz"],
            epochs=train["epochs"],
            batch=train["batch"],
            patience=train["patience"],
            device=_resolve_device(train["device"]),
            workers=train["workers"],
            project=train["project"],
            name=train["name"],
            seed=train["seed"],
            lr0=train["lr0"],
            lrf=train["lrf"],
            momentum=train["momentum"],
            weight_decay=train["weight_decay"],
            warmup_epochs=train["warmup_epochs"],
            augmentations=aug,
            reporting=reporting,
            val_conf=validate["conf"],
            val_iou=validate["iou"],
            val_split=validate["split"],
            infer_conf=infer["conf"],
            infer_iou=infer["iou"],
            infer_save_dir=_resolve_path(root, infer["save_dir"]),
            mlflow_tracking_uri=tracking_uri,
            mlflow_experiment=mlflow_cfg["experiment_name"],
            mlflow_run_name=mlflow_cfg.get("run_name"),
            raw=raw,
        )

    def train_args(self) -> dict[str, Any]:
        args: dict[str, Any] = {
            "data": str(self.data_yaml),
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "patience": self.patience,
            "device": self.device,
            "workers": self.workers,
            "project": self.project,
            "name": self.name,
            "seed": self.seed,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "exist_ok": True,
            "pretrained": True,
            "verbose": True,
            **self.augmentations.to_ultralytics_args(),
        }

        albumentations = build_albumentations(self.augmentations)
        if albumentations:
            args["augmentations"] = albumentations

        return args
