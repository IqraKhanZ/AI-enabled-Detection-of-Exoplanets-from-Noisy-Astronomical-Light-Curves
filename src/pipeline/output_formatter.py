"""
src/pipeline/output_formatter.py
=================================
Standardized output formatting for the exoplanet detection pipeline.

Provides functions to format individual results into canonical dicts,
persist them as per-target JSON files, aggregate all results into a
DataFrame, and save aggregate CSVs.

Typical usage
-------------
>>> result = format_result(tic_id=12345, label=0, class_probs=[0.9, 0.05, 0.03, 0.02],
...                        confidence_dict={"pipeline_confidence": 0.88},
...                        param_dict={"period_days": 3.52}, snr_dict={"snr": 12.1},
...                        fap_dict={"fap": 1e-6}, centroid_result=None,
...                        contamination_result=None)
>>> save_result(result, "outputs/")
>>> df = aggregate_results("outputs/")
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from utils.config import get, project_root
    from utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:  # pragma: no cover – standalone use
    import logging
    logger = logging.getLogger(__name__)

    def get(k: str, d: Any = None) -> Any:  # type: ignore[misc]
        return d

    def project_root() -> Path:  # type: ignore[misc]
        return Path(".")


# ---------------------------------------------------------------------------
# Class label mapping (must match training convention)
# ---------------------------------------------------------------------------
_LABEL_NAMES: dict[int, str] = {
    0: "PLANET",
    1: "ECLIPSING_BINARY",
    2: "BLEND",
    3: "NOISE",
}

_CONFIDENCE_BINS: list[tuple[float, str]] = [
    (0.90, "VERY_HIGH"),
    (0.75, "HIGH"),
    (0.55, "MEDIUM"),
    (0.0,  "LOW"),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _confidence_level(pipeline_confidence: float) -> str:
    """Map a scalar confidence in [0, 1] to a human-readable tier.

    Parameters
    ----------
    pipeline_confidence:
        Scalar value in [0, 1].

    Returns
    -------
    str
        One of ``VERY_HIGH``, ``HIGH``, ``MEDIUM``, or ``LOW``.
    """
    for threshold, label in _CONFIDENCE_BINS:
        if pipeline_confidence >= threshold:
            return label
    return "LOW"


def _safe_float(value: Any, default: float = float("nan")) -> float:
    """Safely cast *value* to float, returning *default* on failure.

    Parameters
    ----------
    value:
        Any value to attempt conversion.
    default:
        Fallback value when conversion fails (default: NaN).

    Returns
    -------
    float
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _make_serialisable(obj: Any) -> Any:
    """Recursively convert numpy scalars / NaN to JSON-safe types.

    Parameters
    ----------
    obj:
        Any Python object.

    Returns
    -------
    Any
        JSON-serialisable equivalent.
    """
    if isinstance(obj, float) and obj != obj:  # NaN check via identity
        return None
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if v != v else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(x) for x in obj]
    return obj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def format_result(
    tic_id: int | str,
    label: int,
    class_probs: np.ndarray | list[float],
    confidence_dict: dict[str, Any],
    param_dict: dict[str, Any],
    snr_dict: dict[str, Any],
    fap_dict: dict[str, Any],
    centroid_result: dict[str, Any] | None,
    contamination_result: dict[str, Any] | None,
    processing_time_s: float = float("nan"),
) -> dict[str, Any]:
    """Build a canonical result dictionary for a single pipeline target.

    Parameters
    ----------
    tic_id:
        TESS Input Catalogue identifier.
    label:
        Integer class prediction (0=PLANET, 1=ECLIPSING_BINARY,
        2=BLEND, 3=NOISE).
    class_probs:
        Array-like of length 4 with softmax probabilities per class.
    confidence_dict:
        Output from ``scoring.confidence.compute_confidence``.  Must
        contain at least the key ``pipeline_confidence`` (float in [0,1]).
    param_dict:
        Transit parameter estimates.  Expected keys (all optional):
        ``period_days``, ``period_err``, ``depth_ppm``, ``depth_err``,
        ``duration_hrs``, ``duration_err``.
    snr_dict:
        Signal-to-noise information.  Expected key: ``snr`` (float).
    fap_dict:
        False-alarm probability.  Expected key: ``fap`` (float).
    centroid_result:
        Centroid analysis output.  Expected key:
        ``centroid_shift_arcsec`` (float).  May be ``None``.
    contamination_result:
        Contamination analysis output.  Expected keys:
        ``contamination_ratio`` (float), ``is_contaminated`` (bool).
        May be ``None``.
    processing_time_s:
        Wall-clock seconds taken to process this target.

    Returns
    -------
    dict[str, Any]
        Flat dictionary with **all** pipeline outputs:

        * ``tic_id`` – target identifier
        * ``predicted_label`` – integer class index
        * ``predicted_label_name`` – human-readable class name
        * ``confidence_level`` – VERY_HIGH / HIGH / MEDIUM / LOW
        * ``pipeline_confidence`` – scalar [0, 1]
        * ``planet_prob``, ``eb_prob``, ``blend_prob``, ``noise_prob``
        * ``period_days``, ``period_err``
        * ``depth_ppm``, ``depth_err``
        * ``duration_hrs``, ``duration_err``
        * ``snr``, ``fap``
        * ``centroid_shift_arcsec``
        * ``contamination_ratio``, ``is_contaminated``
        * ``processing_time_s``
    """
    probs = list(class_probs) if not isinstance(class_probs, list) else class_probs
    probs = [_safe_float(p) for p in probs]
    while len(probs) < 4:
        probs.append(float("nan"))

    pipeline_conf = _safe_float(confidence_dict.get("pipeline_confidence", 0.0))

    result: dict[str, Any] = {
        # ── Identity ──────────────────────────────────────────────────────
        "tic_id": int(tic_id),
        # ── Classification ────────────────────────────────────────────────
        "predicted_label":      int(label),
        "predicted_label_name": _LABEL_NAMES.get(int(label), "UNKNOWN"),
        "confidence_level":     _confidence_level(pipeline_conf),
        "pipeline_confidence":  pipeline_conf,
        "planet_prob":          probs[0],
        "eb_prob":              probs[1],
        "blend_prob":           probs[2],
        "noise_prob":           probs[3],
        # ── Transit parameters ────────────────────────────────────────────
        "period_days":    _safe_float(param_dict.get("period_days")),
        "period_err":     _safe_float(param_dict.get("period_err")),
        "depth_ppm":      _safe_float(param_dict.get("depth_ppm")),
        "depth_err":      _safe_float(param_dict.get("depth_err")),
        "duration_hrs":   _safe_float(param_dict.get("duration_hrs")),
        "duration_err":   _safe_float(param_dict.get("duration_err")),
        # ── Detection quality ─────────────────────────────────────────────
        "snr": _safe_float(snr_dict.get("snr")),
        "fap": _safe_float(fap_dict.get("fap")),
        # ── Centroid & contamination ───────────────────────────────────────
        "centroid_shift_arcsec": (
            _safe_float(centroid_result.get("centroid_shift_arcsec"))
            if centroid_result
            else float("nan")
        ),
        "contamination_ratio": (
            _safe_float(contamination_result.get("contamination_ratio"))
            if contamination_result
            else float("nan")
        ),
        "is_contaminated": (
            bool(contamination_result.get("is_contaminated", False))
            if contamination_result
            else False
        ),
        # ── Timing ────────────────────────────────────────────────────────
        "processing_time_s": _safe_float(processing_time_s),
    }
    return result


def save_result(result_dict: dict[str, Any], output_dir: str | Path) -> Path:
    """Persist a single result dictionary as a JSON file.

    The file is named ``tic_<tic_id>.json`` and is written to
    *output_dir*, which is created if it does not exist.

    Parameters
    ----------
    result_dict:
        As returned by :func:`format_result`.
    output_dir:
        Directory under which the file will be saved.

    Returns
    -------
    Path
        The absolute path of the saved JSON file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tic_id = result_dict.get("tic_id", "unknown")
    out_path = out_dir / f"tic_{tic_id}.json"

    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(_make_serialisable(result_dict), fh, indent=2)

    logger.debug("Saved result JSON: %s", out_path)
    return out_path


def aggregate_results(output_dir: str | Path) -> pd.DataFrame:
    """Read all per-target JSON files and return a combined DataFrame.

    Parameters
    ----------
    output_dir:
        Directory containing ``tic_*.json`` files.

    Returns
    -------
    pd.DataFrame
        One row per target, one column per output field.  Returns an
        empty :class:`~pandas.DataFrame` if no JSON files are found.
    """
    out_dir = Path(output_dir)
    json_files = sorted(out_dir.glob("tic_*.json"))

    if not json_files:
        logger.warning("No tic_*.json files found in %s", out_dir)
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for jf in json_files:
        try:
            with jf.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            rows.append(data)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", jf, exc)

    df = pd.DataFrame(rows)
    if "tic_id" in df.columns:
        df = df.sort_values("tic_id").reset_index(drop=True)

    logger.info("Aggregated %d results from %s", len(df), out_dir)
    return df


def save_aggregate_csv(output_dir: str | Path) -> Path:
    """Aggregate all JSON results and save ``pipeline_results.csv``.

    Parameters
    ----------
    output_dir:
        Directory containing ``tic_*.json`` files.  The CSV is written
        to the same directory as ``pipeline_results.csv``.

    Returns
    -------
    Path
        Path to the saved CSV file.
    """
    df = aggregate_results(output_dir)
    out_dir = Path(output_dir)
    csv_path = out_dir / "pipeline_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved aggregate CSV (%d rows) to %s", len(df), csv_path)
    return csv_path


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    _parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-target pipeline result JSONs (tic_*.json) into "
            "a single pipeline_results.csv."
        )
    )
    _parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root() / get("paths.outputs", "outputs")),
        help="Directory containing tic_*.json files (default: outputs/).",
    )
    _args = _parser.parse_args()

    _csv_path = save_aggregate_csv(_args.output_dir)
    print(f"Saved: {_csv_path}")
