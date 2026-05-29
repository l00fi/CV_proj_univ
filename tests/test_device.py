"""Tests for automatic GPU/CPU device resolution."""

from __future__ import annotations

import os

import pytest

from poker_yolo.device import detect_training_device, resolve_device_setting


def test_detect_training_device_uses_env_override(monkeypatch) -> None:
    monkeypatch.setenv("POKER_YOLO_DEVICE", "1")
    assert detect_training_device() == "1"


def test_detect_training_device_gpu_when_cuda_available(monkeypatch, mocker) -> None:
    monkeypatch.delenv("POKER_YOLO_DEVICE", raising=False)
    mocker.patch("torch.cuda.is_available", return_value=True)
    mocker.patch("torch.cuda.device_count", return_value=2)
    assert detect_training_device() == "0"


def test_detect_training_device_cpu_fallback(monkeypatch, mocker) -> None:
    monkeypatch.delenv("POKER_YOLO_DEVICE", raising=False)
    mocker.patch("torch.cuda.is_available", return_value=False)
    assert detect_training_device() == "cpu"


def test_detect_training_device_strict_raises_without_cuda(monkeypatch, mocker) -> None:
    monkeypatch.delenv("POKER_YOLO_DEVICE", raising=False)
    mocker.patch("torch.cuda.is_available", return_value=False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        detect_training_device(strict_cuda=True)


def test_resolve_device_setting_auto_uses_detect(mocker) -> None:
    mocker.patch("poker_yolo.device.detect_training_device", return_value="cpu")
    assert resolve_device_setting("auto") == "cpu"


def test_strict_cuda_required_reads_env(monkeypatch) -> None:
    from poker_yolo.device import strict_cuda_required

    monkeypatch.setenv("REQUIRE_CUDA", "1")
    assert strict_cuda_required() is True
    monkeypatch.setenv("REQUIRE_CUDA", "0")
    assert strict_cuda_required() is False


def test_apply_gpu_training_limits_skips_cpu() -> None:
    from poker_yolo.device import apply_gpu_training_limits

    apply_gpu_training_limits("cpu", 0.8)


def test_apply_gpu_training_limits_sets_memory_fraction(mocker) -> None:
    from poker_yolo.device import apply_gpu_training_limits

    mocker.patch("torch.cuda.is_available", return_value=True)
    set_fraction = mocker.patch("torch.cuda.set_per_process_memory_fraction")
    mocker.patch("poker_yolo.device._set_gpu_power_limit_watts", return_value=180)

    apply_gpu_training_limits("0", 0.8)

    set_fraction.assert_called_once_with(0.8, 0)


def test_apply_gpu_training_limits_invalid_fraction() -> None:
    from poker_yolo.device import apply_gpu_training_limits

    with pytest.raises(ValueError, match="gpu_resource_fraction"):
        apply_gpu_training_limits("0", 1.5)


def test_ram_budget_gb() -> None:
    from poker_yolo.device import ram_budget_gb

    budget, total = ram_budget_gb(0.8)
    assert total > 0
    assert budget == pytest.approx(total * 0.8, rel=1e-6)


def test_effective_training_workers() -> None:
    from poker_yolo.device import effective_training_workers

    assert effective_training_workers(4, 0.8) == 3
    assert effective_training_workers(4, None) == 4
    assert effective_training_workers(0, 0.8) == 0


def test_apply_training_resource_limits_applies_ram_and_gpu(mocker) -> None:
    from poker_yolo.device import apply_training_resource_limits

    ram = mocker.patch("poker_yolo.device.log_ram_training_budget")
    gpu = mocker.patch("poker_yolo.device.apply_gpu_training_limits")

    apply_training_resource_limits("0", 0.8)

    ram.assert_called_once_with(0.8)
    gpu.assert_called_once_with("0", 0.8)


def test_apply_training_resource_limits_cpu_skips_gpu(mocker) -> None:
    from poker_yolo.device import apply_training_resource_limits

    ram = mocker.patch("poker_yolo.device.log_ram_training_budget")
    gpu = mocker.patch("poker_yolo.device.apply_gpu_training_limits")

    apply_training_resource_limits("cpu", 0.8)

    ram.assert_called_once_with(0.8)
    gpu.assert_not_called()
