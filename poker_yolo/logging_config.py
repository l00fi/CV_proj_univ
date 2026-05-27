from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line (Grafana Loki / JSON datasource friendly)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "action"):
            payload["action"] = record.action
        if hasattr(record, "details") and record.details:
            payload["details"] = record.details
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(
    log_dir: Path,
    level: str = "INFO",
    json_file: bool = True,
    console_json: bool = False,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level.upper())

    console = logging.StreamHandler(sys.stdout)
    if console_json:
        console.setFormatter(JsonFormatter())
    else:
        console.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
    root.addHandler(console)

    if json_file:
        file_handler = RotatingFileHandler(
            log_dir / "poker-yolo.jsonl",
            maxBytes=10 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)


def log_action(logger: logging.Logger, level: int, action: str, message: str, **details: Any) -> None:
    """Structured log line for a key pipeline action."""
    logger.log(level, message, extra={"action": action, "details": details or None})
