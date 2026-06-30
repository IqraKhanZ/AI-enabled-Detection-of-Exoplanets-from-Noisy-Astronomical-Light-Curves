"""
src/pipeline/logging_config.py
================================
Centralized logging setup for the exoplanet detection pipeline.

This module provides rich-formatted console output and structured
JSON file logging for every stage of the pipeline.  It builds on
top of the base :mod:`utils.logger` machinery while adding pipeline-
specific convenience helpers for structured log entries.

Functions
---------
setup_pipeline_logging
log_target_start
log_target_done
log_target_error
log_pipeline_summary
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Rich is an optional but recommended dependency for pretty terminal output.
try:
    from rich.console import Console
    from rich.logging import RichHandler
    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RICH_AVAILABLE = False

try:
    from utils.config import get, load_config
except Exception:  # pragma: no cover – standalone use
    def get(k: str, d: Any = None) -> Any:  # type: ignore[misc]
        return d

    def load_config(config_path: str | Path | None = None) -> dict:  # type: ignore[misc]
        return {}


# ---------------------------------------------------------------------------
# Private formatters
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects with structured fields."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload: dict[str, Any] = {
            "ts":     datetime.now(tz=timezone.utc).isoformat(),
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Merge any extra structured fields attached to the record
        for key, val in record.__dict__.items():
            if key.startswith("pipeline_") or key.startswith("target_"):
                payload[key] = val
        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_pipeline_logging(config_path: str | Path | None = None) -> logging.Logger:
    """Initialise and return the root pipeline logger.

    Sets up two handlers:

    * **Console** – uses :class:`rich.logging.RichHandler` when available,
      otherwise falls back to a plain :class:`logging.StreamHandler`.
      Output is human-readable with colours/markup.
    * **File** – writes newline-delimited JSON to the path specified in
      ``config.yaml`` under ``logging.log_file``.

    Parameters
    ----------
    config_path:
        Optional path to ``config.yaml``.  If *None* the default config
        discovered by :func:`utils.config.load_config` is used.

    Returns
    -------
    logging.Logger
        The configured root pipeline logger (name ``"pipeline"``).
    """
    try:
        cfg_logging = load_config(config_path).get("logging", {})
    except Exception:
        cfg_logging = {}

    level_str: str = cfg_logging.get("level", "INFO")
    log_file: str | None = cfg_logging.get("log_file", "outputs/pipeline.log")
    json_format: bool = cfg_logging.get("json_format", True)
    level: int = getattr(logging, level_str.upper(), logging.INFO)

    logger = logging.getLogger("pipeline")
    if logger.handlers:
        # Already configured – return early to avoid duplicate handlers.
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # ── Console handler ─────────────────────────────────────────────────────
    if _RICH_AVAILABLE:
        console = Console(stderr=False, highlight=True)
        console_handler: logging.Handler = RichHandler(
            console=console,
            show_time=True,
            show_level=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True,
            log_time_format="[%X]",
        )
        console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        if json_format:
            console_handler.setFormatter(_JsonFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
            )
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # ── File handler (JSON) ──────────────────────────────────────────────────
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(_JsonFormatter())
        logger.addHandler(file_handler)

    logger.info(
        "Pipeline logging initialised",
        extra={"pipeline_log_file": str(log_file), "pipeline_level": level_str},
    )
    return logger


def log_target_start(
    logger: logging.Logger,
    tic_id: int | str,
    sector: int | str,
) -> None:
    """Emit a structured log entry when processing of a target begins.

    Parameters
    ----------
    logger:
        The logger to use (typically the one returned by
        :func:`setup_pipeline_logging`).
    tic_id:
        TESS Input Catalogue identifier.
    sector:
        TESS observation sector number.
    """
    logger.info(
        "START TIC %s sector=%s",
        tic_id,
        sector,
        extra={
            "target_tic_id":  int(tic_id),
            "target_sector":  int(sector),
            "target_phase":   "start",
            "target_ts":      time.time(),
        },
    )


def log_target_done(
    logger: logging.Logger,
    tic_id: int | str,
    elapsed_s: float,
    label: str | int,
    confidence: float,
) -> None:
    """Emit a structured log entry when a target has been fully processed.

    Parameters
    ----------
    logger:
        The logger to use.
    tic_id:
        TESS Input Catalogue identifier.
    elapsed_s:
        Wall-clock processing time in seconds.
    label:
        Predicted class label (integer index or string name).
    confidence:
        Pipeline confidence score in [0, 1].
    """
    _LABEL_NAMES: dict[int, str] = {
        0: "PLANET", 1: "ECLIPSING_BINARY", 2: "BLEND", 3: "NOISE"
    }
    label_name: str = (
        _LABEL_NAMES.get(int(label), str(label))
        if isinstance(label, (int, float))
        else str(label)
    )
    logger.info(
        "DONE  TIC %s  label=%s  confidence=%.3f  elapsed=%.1fs",
        tic_id,
        label_name,
        confidence,
        elapsed_s,
        extra={
            "target_tic_id":    int(tic_id),
            "target_phase":     "done",
            "target_label":     label_name,
            "target_confidence": round(float(confidence), 4),
            "target_elapsed_s": round(float(elapsed_s), 2),
        },
    )


def log_target_error(
    logger: logging.Logger,
    tic_id: int | str,
    error: Exception,
) -> None:
    """Emit a structured error log when processing of a target fails.

    Parameters
    ----------
    logger:
        The logger to use.
    tic_id:
        TESS Input Catalogue identifier of the failing target.
    error:
        The caught exception.
    """
    logger.error(
        "ERROR TIC %s: %s — %s",
        tic_id,
        type(error).__name__,
        error,
        exc_info=True,
        extra={
            "target_tic_id":    int(tic_id),
            "target_phase":     "error",
            "target_error_type": type(error).__name__,
            "target_error_msg":  str(error),
        },
    )


def log_pipeline_summary(
    logger: logging.Logger,
    n_total: int,
    n_success: int,
    n_failed: int,
    total_time_s: float,
) -> None:
    """Emit a final summary log entry after the full pipeline completes.

    Parameters
    ----------
    logger:
        The logger to use.
    n_total:
        Total number of targets attempted.
    n_success:
        Number of targets successfully processed.
    n_failed:
        Number of targets that raised an error.
    total_time_s:
        Total wall-clock run time in seconds.
    """
    rate = n_total / max(total_time_s, 1e-6)
    logger.info(
        "PIPELINE SUMMARY | total=%d  success=%d  failed=%d  "
        "time=%.1fs  rate=%.2f tgt/s",
        n_total,
        n_success,
        n_failed,
        total_time_s,
        rate,
        extra={
            "pipeline_phase":         "summary",
            "pipeline_n_total":       n_total,
            "pipeline_n_success":     n_success,
            "pipeline_n_failed":      n_failed,
            "pipeline_total_time_s":  round(total_time_s, 2),
            "pipeline_rate_tgt_per_s": round(rate, 4),
        },
    )


# ---------------------------------------------------------------------------
# CLI entry-point (diagnostic / test)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    _parser = argparse.ArgumentParser(
        description="Initialise pipeline logging and emit test messages."
    )
    _parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    _args = _parser.parse_args()

    _log = setup_pipeline_logging(_args.config)
    log_target_start(_log, tic_id=12345678, sector=1)
    log_target_done(_log, tic_id=12345678, elapsed_s=42.3, label=0, confidence=0.91)
    log_target_error(_log, tic_id=99999999, error=RuntimeError("test error"))
    log_pipeline_summary(_log, n_total=100, n_success=97, n_failed=3, total_time_s=300.0)
