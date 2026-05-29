"""Tests for results.csv parsing."""

from __future__ import annotations

from pathlib import Path

from poker_yolo.training_curves import parse_training_results_csv, results_csv_path


def test_parse_training_results_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "epoch,time,train/loss,metrics/accuracy_top1,metrics/accuracy_top5,val/loss\n"
        "1,10,1.9,0.5,0.9,1.2\n"
        "2,20,1.4,0.6,0.95,0.9\n",
        encoding="utf-8",
    )
    curves = parse_training_results_csv(csv_path)
    assert len(curves) == 2
    assert curves[0]["epoch"] == 1
    assert curves[0]["train_loss"] == 1.9
    assert curves[1]["top5"] == 0.95


def test_results_csv_path_from_weights(tmp_path: Path) -> None:
    weights = tmp_path / "run" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.touch()
    assert results_csv_path(weights) == (tmp_path / "run" / "results.csv").resolve()
