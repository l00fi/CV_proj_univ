"""Context manager for a single MLflow run per pipeline phase."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import mlflow

from poker_yolo.config import Config
from poker_yolo.mlflow_utils import log_config, setup_mlflow


@contextmanager
def mlflow_phase(config: Config, run_name: str, *, log_params: bool = True) -> Iterator[None]:
    """Start MLflow run, optionally log config, always end run on exit."""
    setup_mlflow(config, run_name=run_name)
    if log_params:
        log_config(config)
    try:
        yield
    finally:
        if mlflow.active_run() is not None:
            mlflow.end_run()
