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
