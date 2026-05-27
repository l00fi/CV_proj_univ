"""Tests for structured logging setup."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from poker_yolo.logging_config import JsonFormatter, log_action, setup_logging


def test_json_formatter_includes_action_and_details() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.action = "train.start"
    record.details = {"epochs": 10}

    payload = json.loads(formatter.format(record))
    assert payload["action"] == "train.start"
    assert payload["details"]["epochs"] == 10
    assert payload["message"] == "hello"


def test_setup_logging_writes_jsonl(tmp_path: Path) -> None:
    setup_logging(tmp_path, json_file=True, console_json=False)
    log_action(logging.getLogger("test"), logging.INFO, "test.event", "test message", foo="bar")

    log_file = tmp_path / "poker-yolo.jsonl"
    assert log_file.exists()
    line = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert line["action"] == "test.event"
    assert line["details"]["foo"] == "bar"
