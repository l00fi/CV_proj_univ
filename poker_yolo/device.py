"""Resolve PyTorch / Ultralytics training device (GPU if available, else CPU)."""

from __future__ import annotations

import logging
import os
import re
import subprocess

import psutil

logger = logging.getLogger(__name__)


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


def release_cuda_memory() -> None:
    """Best-effort GPU memory cleanup between heavy Ultralytics runs."""
    try:
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _cuda_device_index(device: str) -> int | None:
    if device == "cpu":
        return None
    if device.isdigit():
        return int(device)
    return 0


def _set_gpu_power_limit_watts(device_index: int, fraction: float) -> int | None:
    """Set NVIDIA power limit to ``fraction`` of max TDP. Returns target watts or None."""
    if fraction <= 0 or fraction > 1:
        return None
    query = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(device_index),
            "--query-gpu=power.max_limit",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if query.returncode != 0:
        return None
    match = re.search(r"([\d.]+)", query.stdout or "")
    if not match:
        return None
    max_watts = float(match.group(1))
    target_watts = max(1, int(max_watts * fraction))
    result = subprocess.run(
        ["nvidia-smi", "-i", str(device_index), "-pl", str(target_watts)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        logger.debug("nvidia-smi power limit failed: %s", (result.stderr or "").strip())
        return None
    return target_watts


def ram_budget_gb(fraction: float) -> tuple[float, float]:
    """Return (budget_gb, total_gb) for ``fraction`` of visible system RAM."""
    if fraction <= 0 or fraction > 1:
        raise ValueError(f"resource fraction must be in (0, 1], got {fraction}")
    total_bytes = int(psutil.virtual_memory().total)
    budget_bytes = max(1, int(total_bytes * fraction))
    return budget_bytes / (1024**3), total_bytes / (1024**3)


def log_ram_training_budget(fraction: float) -> None:
    """Log RAM budget; hard cap must come from Docker ``mem_limit`` (see POKER_YOLO_MEM_LIMIT).

    ``RLIMIT_AS`` is intentionally not used: PyTorch/Ultralytics need a large virtual
    address space and fail with tiny allocations when VM is capped at ~80% of RAM.
    """
    budget_gb, total_gb = ram_budget_gb(fraction)
    logger.info(
        "Training limit: RAM budget %.0f%% (~%.1f / %.1f GB visible). "
        "Hard cap: set POKER_YOLO_MEM_LIMIT in .env or docker compose (e.g. %.0fg)",
        fraction * 100,
        budget_gb,
        total_gb,
        max(1, int(budget_gb)),
    )


def effective_training_workers(workers: int, fraction: float | None) -> int:
    """Reduce DataLoader workers to stay within the RAM budget (soft limit)."""
    if fraction is None or workers <= 0:
        return workers
    return max(0, min(workers, int(workers * fraction + 0.5)))


def apply_gpu_training_limits(device: str, fraction: float) -> None:
    """Cap GPU use during training (VRAM fraction + power limit when supported).

    ``fraction`` of 0.8 reserves ~20% headroom for the OS / desktop / other jobs.
    Only applied when ``device`` is a CUDA GPU.
    """
    if fraction <= 0 or fraction > 1:
        raise ValueError(f"gpu_resource_fraction must be in (0, 1], got {fraction}")

    device_index = _cuda_device_index(device)
    if device_index is None:
        return

    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not installed; skipping GPU resource limits")
        return

    if not torch.cuda.is_available():
        logger.warning("CUDA not available; skipping GPU resource limits")
        return

    torch.cuda.set_per_process_memory_fraction(fraction, device_index)
    logger.info(
        "GPU training limit: CUDA memory capped at %.0f%% on device %s",
        fraction * 100,
        device,
    )

    target_watts = _set_gpu_power_limit_watts(device_index, fraction)
    if target_watts is not None:
        logger.info(
            "GPU training limit: power limit set to %d W (~%.0f%% of TDP) on GPU %s",
            target_watts,
            fraction * 100,
            device_index,
        )
    else:
        logger.info(
            "GPU training limit: power limit not applied (nvidia-smi unavailable or denied); "
            "memory cap still active",
        )


def apply_training_resource_limits(device: str, fraction: float) -> None:
    """Apply GPU VRAM/power caps and log RAM budget (Docker mem_limit for hard RAM cap)."""
    log_ram_training_budget(fraction)
    if device != "cpu":
        apply_gpu_training_limits(device, fraction)
