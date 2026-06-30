"""
src/acquisition/gaia_crossmatch.py
=====================================
Cross-matches pipeline targets against Gaia DR3.

Reads the download manifest to obtain TIC IDs, queries the Gaia DR3
catalog (``gaiadr3.gaia_source``) via ADQL using ``astroquery.gaia``,
retrieves key astrometric and photometric columns, computes the number of
neighbor sources within the aperture and a contamination ratio, and writes
results to ``data/interim/gaia_matches.csv``.

The contamination ratio is defined as::

    contamination_ratio = sum(neighbor_flux) / target_flux

where ``flux ~ 10^(-0.4 * G_mag)``.

Usage
-----
.. code-block:: bash

    python src/acquisition/gaia_crossmatch.py \\
        --manifest data/raw/lightcurves/manifest.csv \\
        --output   data/interim/gaia_matches.csv \\
        --radius-arcsec 21 \\
        --batch-size 50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.config import get, load_config, project_root  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

try:
    from astroquery.gaia import Gaia
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
    Gaia.ROW_LIMIT = -1
except ImportError as _exc:
    logger.error("astroquery[gaia] is not installed: %s", _exc)
    Gaia = None  # type: ignore[assignment]

try:
    from astroquery.mast import Catalogs as MastCatalogs
except ImportError:
    MastCatalogs = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GAIA_COLUMNS = [
    "source_id",
    "ra",
    "dec",
    "parallax",
    "parallax_error",
    "phot_g_mean_mag",
    "bp_rp",
    "ruwe",
    "phot_variable_flag",
]

OUTPUT_COLUMNS = [
    "tic_id",
    "gaia_source_id",
    "ra",
    "dec",
    "parallax",
    "parallax_error",
    "phot_g_mean_mag",
    "bp_rp",
    "ruwe",
    "phot_variable_flag",
    "neighbor_count",
    "contamination_ratio",
    "separation_arcsec",
    "query_status",
]

RETRY_DELAY = 5  # seconds between retries
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# TIC coordinate lookup
# ---------------------------------------------------------------------------

def _get_tic_coords(tic_id: int) -> Optional[tuple[float, float]]:
    """Look up RA/Dec for a TIC ID via MAST TIC catalog.

    Parameters
    ----------
    tic_id : int
        TIC identifier.

    Returns
    -------
    tuple[float, float] or None
        (ra_deg, dec_deg) in decimal degrees, or ``None`` on failure.
    """
    if MastCatalogs is None:
        return None
    try:
        result = MastCatalogs.query_criteria(catalog="TIC", ID=tic_id)
        if result is None or len(result) == 0:
            return None
        ra = float(result["ra"][0])
        dec = float(result["dec"][0])
        return ra, dec
    except Exception as exc:
        logger.debug("TIC %d coord lookup failed: %s", tic_id, exc)
        return None


# ---------------------------------------------------------------------------
# Gaia query
# ---------------------------------------------------------------------------

def _flux_from_mag(g_mag: float) -> float:
    """Convert Gaia G-band magnitude to relative flux.

    Parameters
    ----------
    g_mag : float
        Gaia G-band magnitude.

    Returns
    -------
    float
        Relative flux (arbitrary units, ratio-safe).
    """
    return 10.0 ** (-0.4 * g_mag)


def _query_gaia_cone(
    ra_deg: float,
    dec_deg: float,
    radius_arcsec: float,
) -> Optional[pd.DataFrame]:
    """Query Gaia DR3 in a cone around (ra_deg, dec_deg).

    Parameters
    ----------
    ra_deg : float
        Right ascension in decimal degrees.
    dec_deg : float
        Declination in decimal degrees.
    radius_arcsec : float
        Search radius in arcseconds.

    Returns
    -------
    pd.DataFrame or None
        Gaia sources within the cone, or ``None`` on failure.
    """
    if Gaia is None:
        return None

    radius_deg = radius_arcsec / 3600.0
    adql = (
        "SELECT "
        + ", ".join(f"gs.{c}" for c in GAIA_COLUMNS)
        + ", DISTANCE(POINT('ICRS', gs.ra, gs.dec), "
        + f"POINT('ICRS', {ra_deg}, {dec_deg})) * 3600.0 AS separation_arcsec "
        + "FROM gaiadr3.gaia_source AS gs "
        + "WHERE CONTAINS("
        + "POINT('ICRS', gs.ra, gs.dec), "
        + f"CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg})"
        + ") = 1"
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            job = Gaia.launch_job(adql)
            result = job.get_results()
            if result is None or len(result) == 0:
                return pd.DataFrame(columns=GAIA_COLUMNS + ["separation_arcsec"])
            return result.to_pandas()
        except Exception as exc:
            logger.warning("Gaia query attempt %d failed: %s", attempt, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return None


def _build_match_row(
    tic_id: int,
    gaia_df: Optional[pd.DataFrame],
    radius_arcsec: float,
) -> dict:
    """Build an output row from a Gaia cone-search result.

    Parameters
    ----------
    tic_id : int
        TIC identifier.
    gaia_df : pd.DataFrame or None
        Cone-search result.  ``None`` if the query failed.
    radius_arcsec : float
        Cone search radius used.

    Returns
    -------
    dict
        Row matching :data:`OUTPUT_COLUMNS`.
    """
    base: dict = {c: np.nan for c in OUTPUT_COLUMNS}
    base["tic_id"] = tic_id
    base["query_status"] = "FAILED"

    if gaia_df is None:
        base["query_status"] = "QUERY_ERROR"
        return base

    if len(gaia_df) == 0:
        base["query_status"] = "NO_MATCH"
        base["neighbor_count"] = 0
        base["contamination_ratio"] = 0.0
        return base

    # Sort by separation; nearest source = target
    gaia_df = gaia_df.sort_values("separation_arcsec").reset_index(drop=True)
    target = gaia_df.iloc[0]
    neighbors = gaia_df.iloc[1:]

    base["gaia_source_id"] = int(target["source_id"]) if pd.notna(target.get("source_id")) else np.nan
    base["ra"] = float(target.get("ra", np.nan))
    base["dec"] = float(target.get("dec", np.nan))
    base["parallax"] = float(target.get("parallax", np.nan))
    base["parallax_error"] = float(target.get("parallax_error", np.nan))
    base["phot_g_mean_mag"] = float(target.get("phot_g_mean_mag", np.nan))
    base["bp_rp"] = float(target.get("bp_rp", np.nan))
    base["ruwe"] = float(target.get("ruwe", np.nan))
    base["phot_variable_flag"] = str(target.get("phot_variable_flag", ""))
    base["separation_arcsec"] = float(target.get("separation_arcsec", 0.0))
    base["neighbor_count"] = len(neighbors)

    # Contamination ratio
    target_g = base["phot_g_mean_mag"]
    if not np.isnan(target_g) and len(neighbors) > 0:
        target_flux = _flux_from_mag(target_g)
        neighbor_mags = pd.to_numeric(neighbors["phot_g_mean_mag"], errors="coerce").dropna()
        neighbor_flux_sum = sum(_flux_from_mag(m) for m in neighbor_mags)
        base["contamination_ratio"] = neighbor_flux_sum / max(target_flux, 1e-30)
    else:
        base["contamination_ratio"] = 0.0

    base["query_status"] = "OK"
    return base


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_crossmatch(
    manifest_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    radius_arcsec: float = 21.0,
    batch_size: int = 50,
) -> pd.DataFrame:
    """Cross-match all targets in the manifest against Gaia DR3.

    Parameters
    ----------
    manifest_path : Path, optional
        Path to ``manifest.csv``.
    output_path : Path, optional
        Destination CSV for Gaia matches.
    radius_arcsec : float
        Cone search radius in arcseconds.
    batch_size : int
        Number of targets to query per batch (for logging / batched sleep).

    Returns
    -------
    pd.DataFrame
        Gaia cross-match results.
    """
    root = project_root()
    if manifest_path is None:
        manifest_path = root / get("data.raw_dir", "data/raw") / "lightcurves" / "manifest.csv"
    if output_path is None:
        output_path = root / get("data.interim_dir", "data/interim") / "gaia_matches.csv"

    manifest_path = Path(manifest_path)
    output_path = Path(output_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path, low_memory=False)
    tic_ids = sorted(manifest["tic_id"].dropna().astype(int).unique().tolist())
    logger.info("Cross-matching %d unique TIC IDs against Gaia DR3.", len(tic_ids))

    rows: list[dict] = []
    total_batches = (len(tic_ids) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        batch = tic_ids[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        logger.info(
            "Processing batch %d/%d (%d targets)...",
            batch_idx + 1, total_batches, len(batch),
        )

        for tic_id in tqdm(batch, desc=f"Batch {batch_idx+1}", unit="target"):
            coords = _get_tic_coords(tic_id)
            if coords is None:
                row: dict = {c: np.nan for c in OUTPUT_COLUMNS}
                row["tic_id"] = tic_id
                row["query_status"] = "NO_COORDS"
                rows.append(row)
                continue

            ra_deg, dec_deg = coords
            gaia_df = _query_gaia_cone(ra_deg, dec_deg, radius_arcsec)
            row = _build_match_row(tic_id, gaia_df, radius_arcsec)
            rows.append(row)

        # Brief pause between batches to respect rate limits
        if batch_idx < total_batches - 1:
            time.sleep(1.0)

    result_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    logger.info("Gaia cross-match complete. Results saved to %s", output_path)

    # Summary
    status_counts = result_df["query_status"].value_counts()
    logger.info("Cross-match status summary:\n%s", status_counts.to_string())
    print("\n=== Gaia Cross-Match Summary ===")
    for status, n in status_counts.items():
        print(f"  {status:25s}: {n}")

    return result_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-match pipeline targets against Gaia DR3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to manifest.csv.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for gaia_matches.csv.",
    )
    parser.add_argument(
        "--radius-arcsec", type=float, default=21.0,
        help="Cone search radius in arcseconds.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="Number of targets per query batch.",
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

    _manifest = Path(_args.manifest) if _args.manifest else None
    _output = Path(_args.output) if _args.output else None

    _df = run_crossmatch(
        manifest_path=_manifest,
        output_path=_output,
        radius_arcsec=_args.radius_arcsec,
        batch_size=_args.batch_size,
    )
    print(f"\nTotal Gaia matches written: {len(_df)}")
