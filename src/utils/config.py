"""
src/utils/config.py — Configuration loader for the pipeline.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration file and return as a nested dict.

    Parameters
    ----------
    config_path:
        Explicit path to the config file.  Falls back to
        ``configs/config.yaml`` relative to the project root.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def get(key_path: str, default: Any = None, config_path: str | Path | None = None) -> Any:
    """Retrieve a nested config value using dot-separated keys.

    Example
    -------
    >>> get("model.transformer_d_model")
    128
    """
    cfg = load_config(config_path)
    parts = key_path.split(".")
    node: Any = cfg
    for part in parts:
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return default
    return node


def project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return _DEFAULT_CONFIG_PATH.parents[1]


def data_path(*parts: str) -> Path:
    """Resolve a path under data_root defined in the config."""
    root = project_root()
    data_root = get("paths.data_root", "data")
    return root / data_root / Path(*parts) if parts else root / data_root
