"""
src/acquisition/download_labels.py
====================================
Fetches and constructs the labeled training dataset.

Queries the NASA Exoplanet Archive TOI table (via direct URL or
``astroquery.nasa_exoplanet_archive``), maps TFOPWG dispositions to integer
class labels, optionally merges a curated override file, and writes the
consolidated labels to ``data/raw/labels/toi_labels.csv``.

Label mapping
-------------
* CP / PC  -> 0  (PLANET)
* FP / EB  -> 1  (ECLIPSING_BINARY)
* FP / BEB or BD -> 2  (BLEND)
* FA / null / unknown -> 3  (NOISE)

Usage
-----
.. code-block:: bash

    python src/acquisition/download_labels.py \\
        --output-dir data/raw/labels \\
        --curated-file data/raw/labels/curated_labels.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.config import get, load_config, project_root  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOI_TAP_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+*+from+toi&format=csv"
)

OUTPUT_COLUMNS = [
    "tic_id",
    "toi_id",
    "tfopwg_disp",
    "label",
    "label_name",
    "period_days",
    "depth_ppm",
    "duration_hrs",
]

LABEL_NAMES = {
    0: "PLANET",
    1: "ECLIPSING_BINARY",
    2: "BLEND",
    3: "NOISE",
}

# TFOPWG disposition sub-type keywords -> label
_EB_KEYWORDS = {"eb", "eclipsing binary"}
_BLEND_KEYWORDS = {"beb", "bd", "blend", "background eclipsing binary"}


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def _disposition_to_label(disp: str) -> int:
    """Map a TFOPWG disposition string to an integer class label.

    Parameters
    ----------
    disp : str
        Raw disposition string from the TOI table (e.g. ``'CP'``, ``'FP'``).

    Returns
    -------
    int
        One of 0 (PLANET), 1 (ECLIPSING_BINARY), 2 (BLEND), 3 (NOISE).
    """
    if not isinstance(disp, str) or disp.strip() == "":
        return 3  # NOISE

    disp_clean = disp.strip().upper()

    if disp_clean in ("CP", "PC", "KP"):
        return 0  # PLANET

    if disp_clean in ("FA",):
        return 3  # NOISE / False Alarm

    if disp_clean == "FP":
        # Default FP without sub-type -> NOISE
        return 3

    # Check sub-type keyword in parenthetical or free-text
    lower = disp.lower()
    for kw in _EB_KEYWORDS:
        if kw in lower:
            return 1  # ECLIPSING_BINARY
    for kw in _BLEND_KEYWORDS:
        if kw in lower:
            return 2  # BLEND

    return 3  # fallback -> NOISE


def _map_fp_subtype(row: pd.Series) -> int:
    """Refine FP label using sub-disposition columns if available.

    Parameters
    ----------
    row : pd.Series
        A row from the TOI dataframe, potentially containing
        ``'tfopwg_disp'`` and ``'tfopwg_disposition'`` columns.

    Returns
    -------
    int
        Refined label.
    """
    base = _disposition_to_label(str(row.get("tfopwg_disp", "")))
    if base != 3:
        return base

    # Try to read a sub-disposition column
    for col in ("tfopwg_disposition", "disp_pis", "disp_sg1a", "disp_sg1b"):
        val = str(row.get(col, ""))
        if val and val.lower() not in ("nan", "none", ""):
            lower = val.lower()
            for kw in _EB_KEYWORDS:
                if kw in lower:
                    return 1
            for kw in _BLEND_KEYWORDS:
                if kw in lower:
                    return 2

    return base


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _fetch_toi_table_url() -> Optional[pd.DataFrame]:
    """Download TOI table from NASA Exoplanet Archive TAP endpoint.

    Returns
    -------
    pd.DataFrame or None
        Parsed dataframe, or ``None`` on failure.
    """
    logger.info("Fetching TOI table from NASA Exoplanet Archive TAP ...")
    try:
        resp = requests.get(TOI_TAP_URL, timeout=120)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text), low_memory=False)
        logger.info("Fetched %d rows from TOI table.", len(df))
        return df
    except Exception as exc:
        logger.error("Failed to fetch TOI table via URL: %s", exc)
        return None


def _fetch_toi_table_astroquery() -> Optional[pd.DataFrame]:
    """Download TOI table via astroquery.nasa_exoplanet_archive.

    Returns
    -------
    pd.DataFrame or None
        Parsed dataframe, or ``None`` on failure.
    """
    try:
        from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
        logger.info("Fetching TOI table via astroquery ...")
        tbl = NasaExoplanetArchive.query_criteria(table="toi", select="*")
        df = tbl.to_pandas()
        logger.info("Fetched %d rows via astroquery.", len(df))
        return df
    except Exception as exc:
        logger.error("astroquery fetch failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Column normalisation helpers
# ---------------------------------------------------------------------------

def _safe_col(df: pd.DataFrame, candidates: list[str], default: object = None) -> pd.Series:
    """Return the first matching column from *candidates*, else a constant Series.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    candidates : list[str]
        Column name candidates to try (case-insensitive).
    default : object
        Fill value when no candidate is found.

    Returns
    -------
    pd.Series
        The found column or a Series of *default*.
    """
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return df[lower_map[cand.lower()]]
    return pd.Series([default] * len(df), index=df.index)


def _build_output_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Transform the raw TOI dataframe into the pipeline label format.

    Parameters
    ----------
    raw : pd.DataFrame
        Raw TOI table from NASA Exoplanet Archive.

    Returns
    -------
    pd.DataFrame
        Dataframe with columns matching :data:`OUTPUT_COLUMNS`.
    """
    df = raw.copy()
    # Normalise column names
    df.columns = [c.lower().strip() for c in df.columns]

    tic_col = _safe_col(df, ["tic_id", "ticid", "tid"])
    toi_col = _safe_col(df, ["toi", "toi_id", "toipfx"], default="")
    disp_col = _safe_col(df, ["tfopwg_disp", "tfopwg_disposition", "disp"], default="")
    period_col = _safe_col(df, ["pl_orbper", "period", "toi_period"], default=float("nan"))
    depth_col = _safe_col(df, ["pl_trandep", "depth_ppm", "toi_depth"], default=float("nan"))
    dur_col = _safe_col(df, ["pl_trandur", "duration_hrs", "toi_duration"], default=float("nan"))

    out = pd.DataFrame({
        "tic_id": tic_col.values,
        "toi_id": toi_col.values,
        "tfopwg_disp": disp_col.astype(str).values,
        "period_days": pd.to_numeric(period_col, errors="coerce").values,
        "depth_ppm": pd.to_numeric(depth_col, errors="coerce").values,
        "duration_hrs": pd.to_numeric(dur_col, errors="coerce").values,
    })

    out["tic_id"] = pd.to_numeric(out["tic_id"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["tic_id"])

    # Apply label mapping (with FP sub-type refinement using full df)
    labels = []
    for i, row in out.iterrows():
        full_row = df.iloc[out.index.get_loc(i)] if isinstance(df.index, pd.RangeIndex) else df.loc[i]
        labels.append(_map_fp_subtype(full_row))
    out["label"] = labels
    out["label_name"] = out["label"].map(LABEL_NAMES)

    return out[OUTPUT_COLUMNS]


def _merge_curated(base: pd.DataFrame, curated_path: Path) -> pd.DataFrame:
    """Merge curated override labels into *base*, prioritising curated rows.

    Parameters
    ----------
    base : pd.DataFrame
        Auto-fetched labels.
    curated_path : Path
        CSV with at minimum ``tic_id`` and ``label`` columns.

    Returns
    -------
    pd.DataFrame
        Merged dataframe.
    """
    try:
        curated = pd.read_csv(curated_path, low_memory=False)
        curated.columns = [c.lower().strip() for c in curated.columns]
        logger.info("Loaded %d curated labels from %s", len(curated), curated_path)
    except Exception as exc:
        logger.error("Failed to load curated labels: %s", exc)
        return base

    curated["tic_id"] = pd.to_numeric(curated["tic_id"], errors="coerce").astype("Int64")
    curated = curated.dropna(subset=["tic_id"])

    # Remove curated TIC IDs from base, then append curated rows
    curated_ids = set(curated["tic_id"].tolist())
    base_filtered = base[~base["tic_id"].isin(curated_ids)].copy()

    # Align columns
    for col in OUTPUT_COLUMNS:
        if col not in curated.columns:
            if col in ("label", "depth_ppm", "period_days", "duration_hrs"):
                curated[col] = float("nan")
            else:
                curated[col] = ""

    if "label_name" not in curated.columns or curated["label_name"].isna().any():
        curated["label"] = pd.to_numeric(curated.get("label", 3), errors="coerce").fillna(3).astype(int)
        curated["label_name"] = curated["label"].map(LABEL_NAMES)

    merged = pd.concat([base_filtered, curated[OUTPUT_COLUMNS]], ignore_index=True)
    logger.info(
        "Merged: %d base rows + %d curated rows = %d total",
        len(base_filtered), len(curated), len(merged),
    )
    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_labels(
    output_dir: Optional[Path] = None,
    curated_file: Optional[Path] = None,
) -> pd.DataFrame:
    """Fetch TOI labels from NASA Exoplanet Archive and write to CSV.

    Parameters
    ----------
    output_dir : Path, optional
        Directory for output CSV.  Defaults to ``data/raw/labels/``.
    curated_file : Path, optional
        Path to optional curated override CSV.

    Returns
    -------
    pd.DataFrame
        Final label dataframe.

    Raises
    ------
    RuntimeError
        If neither the URL fetch nor astroquery fetch succeed.
    """
    root = project_root()
    if output_dir is None:
        output_dir = root / get("data.raw_dir", "data/raw") / "labels"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch raw TOI table
    raw = _fetch_toi_table_url()
    if raw is None:
        logger.warning("URL fetch failed, falling back to astroquery ...")
        raw = _fetch_toi_table_astroquery()
    if raw is None:
        raise RuntimeError("Could not fetch TOI table from NASA Exoplanet Archive.")

    label_df = _build_output_df(raw)
    logger.info("Built label dataframe: %d rows.", len(label_df))

    # Print class distribution summary
    counts = label_df["label_name"].value_counts()
    logger.info("Label distribution:\n%s", counts.to_string())

    # Merge curated overrides if available
    if curated_file is None:
        curated_file = output_dir / "curated_labels.csv"
    curated_path = Path(curated_file)
    if curated_path.exists():
        label_df = _merge_curated(label_df, curated_path)
    else:
        logger.info("No curated labels file found at %s. Skipping merge.", curated_path)

    out_path = output_dir / "toi_labels.csv"
    label_df.to_csv(out_path, index=False)
    logger.info("Labels saved to %s", out_path)

    return label_df


def run(*args, **kwargs) -> pd.DataFrame:
    """Orchestrator entry point mapping to download_labels."""
    output_dir = kwargs.get("output_dir")
    if output_dir is not None:
        output_dir = Path(output_dir)
    curated_file = kwargs.get("curated_file")
    if curated_file is not None:
        curated_file = Path(curated_file)
    return download_labels(output_dir=output_dir, curated_file=curated_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch and label TESS TOI data from NASA Exoplanet Archive.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to write toi_labels.csv.",
    )
    parser.add_argument(
        "--curated-file", type=str, default=None,
        help="Path to curated_labels.csv override file.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to configs/config.yaml.",
    )
    return parser


if __name__ == "__main__":
    _parser = _build_parser()
    _args = _parser.parse_args()

    try:
        load_config(_args.config)
    except Exception as _e:
        logger.warning("Could not load config: %s. Using built-in defaults.", _e)

    _output_dir = Path(_args.output_dir) if _args.output_dir else None
    _curated = Path(_args.curated_file) if _args.curated_file else None

    _df = download_labels(output_dir=_output_dir, curated_file=_curated)
    print(f"\nTotal labeled targets: {len(_df)}")
    print(_df["label_name"].value_counts().to_string())
