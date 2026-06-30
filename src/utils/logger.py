"""
src/utils/logger.py — Structured JSON-capable logger for the pipeline.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from utils.config import load_config


class _JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if hasattr(record, "extra"):
            payload.update(record.extra)  # type: ignore[arg-type]
        return json.dumps(payload)


def get_logger(name: str, config_path: str | Path | None = None) -> logging.Logger:
    """Return a configured logger.

    Parameters
    ----------
    name:
        Logger name (typically ``__name__`` of the calling module).
    config_path:
        Optional path to config file.
    """
    cfg = load_config(config_path).get("logging", {})
    level_str: str = cfg.get("level", "INFO")
    log_file: str | None = cfg.get("log_file")
    json_format: bool = cfg.get("json_format", True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(getattr(logging, level_str.upper(), logging.INFO))

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, level_str.upper(), logging.INFO))
    if json_format:
        ch.setFormatter(_JsonFormatter())
    else:
        ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s"))
    logger.addHandler(ch)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setLevel(getattr(logging, level_str.upper(), logging.INFO))
        fh.setFormatter(_JsonFormatter() if json_format else logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
        ))
        logger.addHandler(fh)

    logger.propagate = False
    return logger


class Timer:
    """Simple context-manager wall-clock timer."""

    def __init__(self, description: str = "Task") -> None:
        self.description = description
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed = time.perf_counter() - self._start

    def __str__(self) -> str:
        return f"{self.description}: {self.elapsed:.2f}s"
