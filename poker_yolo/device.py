"""Resolve PyTorch / Ultralytics training device (GPU if available, else CPU)."""

from __future__ import annotations

import os


def strict_cuda_required() -> bool:
    """True when REQUIRE_CUDA env requests GPU-only mode."""
    return os.environ.get("REQUIRE_CUDA", "").strip().lower() in {"1", "true", "yes", "on"}


def detect_training_device(*, strict_cuda: bool | None = None) -> str:
    """Return Ultralytics device string: GPU index ('0', …) or 'cpu'.

    Precedence: explicit ``POKER_YOLO_DEVICE`` env → first available CUDA GPU → CPU.
    If ``strict_cuda`` is True and no GPU is found, raises RuntimeError.
    When ``strict_cuda`` is None, reads ``REQUIRE_CUDA`` from the environment.
    """
    if strict_cuda is None:
        strict_cuda = strict_cuda_required()

    env_device = os.environ.get("POKER_YOLO_DEVICE", "").strip()
    if env_device:
        return env_device

    try:
        import torch
    except ImportError:
        if strict_cuda:
            raise RuntimeError("PyTorch is not installed; CUDA is required") from None
        return "cpu"

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        if count > 0:
            return "0"

    if strict_cuda:
        raise RuntimeError(
            "CUDA is not available. Install NVIDIA driver + Container Toolkit, "
            "or run with GPU support disabled and device auto (CPU fallback)."
        )
    return "cpu"


def describe_device(device: str | None = None) -> str:
    """Human-readable description for logs."""
    device = device or detect_training_device()
    if device == "cpu":
        try:
            import torch

            return f"CPU (torch {torch.__version__}, CUDA not available)"
        except ImportError:
            return "CPU"

    try:
        import torch

        if torch.cuda.is_available():
            index = int(device) if device.isdigit() else 0
            name = torch.cuda.get_device_name(index)
            count = torch.cuda.device_count()
            suffix = f", {count} GPU(s) visible" if count > 1 else ""
            return f"GPU {index}: {name} (torch {torch.__version__}{suffix})"
    except (ImportError, ValueError, RuntimeError):
        pass
    return f"device={device}"


def resolve_device_setting(yaml_device: str) -> str:
    """Map config ``train.device`` (e.g. ``auto``) to a concrete device string."""
    if yaml_device != "auto":
        return yaml_device
    return detect_training_device()
