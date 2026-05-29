from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from poker_yolo.augmentations import AugmentationConfig, build_albumentations
from poker_yolo.device import resolve_device_setting


def _resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def classify_weights_path(project: str, name: str, weights: str = "best.pt") -> Path:
    """Ultralytics classify saves under ``runs/classify/{project}/{name}/weights/``."""
    return Path("runs/classify") / project / name / "weights" / weights


def resolve_mlflow_uri(yaml_uri: str) -> str:
    """Use env override; map Docker ``mlflow`` hostname to localhost when not in a container."""
    env = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if env:
        return env
    if "mlflow" in yaml_uri and not Path("/.dockerenv").is_file():
        port = os.environ.get("MLFLOW_PORT", "5000")
        return f"http://localhost:{port}"
    return yaml_uri


def mlflow_ui_url(tracking_uri: str) -> str:
    """URL for the browser on the host (Docker internal DNS is not reachable there)."""
    if "mlflow" in tracking_uri:
        port = os.environ.get("MLFLOW_PORT", "5000")
        return f"http://localhost:{port}"
    return tracking_uri


def _parse_gpu_resource_fraction(train: dict[str, Any]) -> float | None:
    env = os.environ.get("POKER_YOLO_GPU_RESOURCE_FRACTION", "").strip()
    if env:
        return float(env)
    raw = train.get("gpu_resource_fraction")
    if raw is None:
        return None
    return float(raw)


@dataclass
class ReportingConfig:
    log_dir: Path
    report_dir: Path
    level: str = "INFO"
    json_logs: bool = True
    console_json: bool = False
    pushgateway_url: str | None = None
    reports_base_url: str = "http://localhost:8088"
    preview_samples: int = 3

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None, project_root: Path) -> ReportingConfig:
        raw = raw or {}
        log_dir = project_root / raw.get("log_dir", "runs/logs")
        report_dir = project_root / raw.get("report_dir", "runs/reports")
        pushgateway = os.environ.get("PROMETHEUS_PUSHGATEWAY_URL", raw.get("pushgateway_url"))
        base_url = os.environ.get("REPORTS_BASE_URL", raw.get("reports_base_url", "http://localhost:8088"))
        return cls(
            log_dir=log_dir.resolve(),
            report_dir=report_dir.resolve(),
            level=str(raw.get("level", "INFO")),
            json_logs=bool(raw.get("json_logs", True)),
            console_json=bool(raw.get("console_json", False)),
            pushgateway_url=pushgateway or None,
            reports_base_url=str(base_url),
            preview_samples=int(raw.get("preview_samples", 3)),
        )


@dataclass
class Config:
    task: str
    data_yaml: Path
    dataset_root: Path
    val_data_yaml: Path
    val_dataset_root: Path
    model_weights: str
    imgsz: int
    epochs: int
    batch: int
    patience: int
    device: str
    gpu_resource_fraction: float | None
    preflight_cpu_epochs: int
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
    val_metric_conf: float
    val_iou: float
    val_split: str
    infer_conf: float
    infer_iou: float
    infer_save_dir: Path
    infer_source: Path
    mlflow_tracking_uri: str
    mlflow_experiment: str
    mlflow_run_name: str | None

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

        tracking_uri = resolve_mlflow_uri(mlflow_cfg["tracking_uri"])
        aug = AugmentationConfig.from_dict(raw.get("augmentations"), train_fallback=train)
        reporting = ReportingConfig.from_dict(raw.get("reporting"), project_root=root)

        benchmark = raw.get("benchmark") or {}
        if benchmark:
            val_data_yaml = _resolve_path(root, benchmark["yaml_path"])
            val_dataset_root = _resolve_path(root, benchmark["dataset_root"])
        else:
            val_data_yaml = _resolve_path(root, data["yaml_path"])
            val_dataset_root = _resolve_path(root, data["dataset_root"])

        infer_source_raw = infer.get("source")
        if infer_source_raw:
            infer_source = _resolve_path(root, infer_source_raw)
        else:
            with val_data_yaml.open(encoding="utf-8") as f:
                bench = yaml.safe_load(f)
            infer_source = val_dataset_root / bench["test"]

        return cls(
            task=str(model.get("task", "classify")),
            data_yaml=_resolve_path(root, data["yaml_path"]),
            dataset_root=_resolve_path(root, data["dataset_root"]),
            val_data_yaml=val_data_yaml,
            val_dataset_root=val_dataset_root,
            model_weights=model["weights"],
            imgsz=model["imgsz"],
            epochs=train["epochs"],
            batch=train["batch"],
            patience=train["patience"],
            device=resolve_device_setting(train["device"]),
            gpu_resource_fraction=_parse_gpu_resource_fraction(train),
            preflight_cpu_epochs=int(train.get("preflight_cpu_epochs", 1)),
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
            val_metric_conf=float(validate.get("metric_conf", 0.001)),
            val_iou=validate["iou"],
            val_split=validate["split"],
            infer_conf=infer["conf"],
            infer_iou=infer["iou"],
            infer_save_dir=_resolve_path(root, infer["save_dir"]),
            infer_source=infer_source,
            mlflow_tracking_uri=tracking_uri,
            mlflow_experiment=mlflow_cfg["experiment_name"],
            mlflow_run_name=mlflow_cfg.get("run_name"),
        )

    def train_args(self) -> dict[str, Any]:
        train_data = str(self.dataset_root) if self.task == "classify" else str(self.data_yaml)
        args: dict[str, Any] = {
            "task": self.task,
            "data": train_data,
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
        }

        if self.task == "detect":
            args.update(self.augmentations.to_ultralytics_args())
            albumentations = build_albumentations(self.augmentations)
            if albumentations:
                args["augmentations"] = albumentations

        return args
