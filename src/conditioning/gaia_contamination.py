"""
gaia_contamination.py
=====================
Gaia DR3 contamination flagging for TESS targets.

Neighbouring Gaia DR3 sources within the TESS photometric aperture dilute the
transit signal and may cause false positives.  This module quantifies the
contamination ratio from Gaia G-band magnitudes and flags targets where the
contamination exceeds 10 %.

The Gaia cross-match CSV is expected to have columns:
    ``tic_id``, ``source_id``, ``ra``, ``dec``, ``phot_g_mean_mag``,
    ``angular_distance_arcsec``, ``is_target`` (1 for the target, 0 for neighbours).

Classes / Dataclasses
---------------------
ContaminationResult
    Container for contamination analysis results.

Functions
---------
compute_contamination(tic_id, gaia_csv_path, aperture_radius_arcsec, threshold)
    Load Gaia matches and return a ContaminationResult.

Author: Exoplanet Detection Pipeline
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Local imports
try:
    from utils.config import load_config, get, project_root
    from utils.logger import get_logger
except ImportError:
    import logging as _logging

    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return _logging.getLogger(name)

    def load_config(path: Optional[str] = None) -> dict:  # type: ignore[misc]
        return {}

    def get(config: dict, key: str, default=None):  # type: ignore[misc]
        keys = key.split(".")
        val = config
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k, default)
            else:
                return default
        return val

    def project_root() -> Path:  # type: ignore[misc]
        return Path(__file__).resolve().parents[2]


logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
DEFAULT_APERTURE_RADIUS_ARCSEC = 42.0   # ~2 TESS pixels
DEFAULT_CONTAMINATION_THRESHOLD = 0.10   # 10 %


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ContaminationResult:
    """Results from Gaia DR3 contamination analysis.

    Attributes
    ----------
    tic_id : str
        TIC identifier of the target star.
    contamination_ratio : float
        Ratio of neighbour flux to target flux: ``Σ F_neighbours / F_target``.
    neighbor_count : int
        Number of Gaia sources inside the aperture (excluding the target).
    is_contaminated : bool
        ``True`` if ``contamination_ratio > threshold``.
    threshold : float
        Contamination threshold used (default 0.10).
    target_flux : float
        Estimated flux of the target from its Gaia G magnitude.
    neighbor_flux_sum : float
        Sum of neighbour fluxes within the aperture.
    neighbor_list : list of str
        Gaia ``source_id`` values for neighbours inside the aperture.
    target_g_mag : float
        Gaia G magnitude of the target.
    aperture_radius_arcsec : float
        Aperture radius used (arcsec).
    """

    tic_id: str = ""
    contamination_ratio: float = 0.0
    neighbor_count: int = 0
    is_contaminated: bool = False
    threshold: float = DEFAULT_CONTAMINATION_THRESHOLD
    target_flux: float = 0.0
    neighbor_flux_sum: float = 0.0
    neighbor_list: list = field(default_factory=list)
    target_g_mag: float = float("nan")
    aperture_radius_arcsec: float = DEFAULT_APERTURE_RADIUS_ARCSEC


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _gmag_to_flux(g_mag: float) -> float:
    """Convert Gaia G magnitude to a relative (arbitrary-scale) flux.

    Uses the standard formula: ``F = 10^{-0.4 * G}``.

    Parameters
    ----------
    g_mag : float
        Gaia G-band magnitude.

    Returns
    -------
    float
        Relative flux value.
    """
    return float(10.0 ** (-0.4 * g_mag))


def _gmag_array_to_flux(g_mags: np.ndarray) -> np.ndarray:
    """Vectorised version of _gmag_to_flux."""
    return 10.0 ** (-0.4 * np.asarray(g_mags, dtype=float))


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def compute_contamination(
    tic_id: str,
    gaia_csv_path: str,
    aperture_radius_arcsec: float = DEFAULT_APERTURE_RADIUS_ARCSEC,
    threshold: float = DEFAULT_CONTAMINATION_THRESHOLD,
    config: Optional[dict] = None,
) -> ContaminationResult:
    """Compute the Gaia contamination ratio for a given TIC target.

    Loads all Gaia DR3 cross-matched sources for *tic_id* from a CSV file,
    separates the target from neighbours, converts G magnitudes to fluxes,
    and computes the contamination ratio within *aperture_radius_arcsec*.

    Parameters
    ----------
    tic_id : str
        TIC identifier string (e.g. ``'123456789'``).
    gaia_csv_path : str
        Path to the Gaia cross-match CSV file.  Expected columns:
        ``tic_id``, ``source_id``, ``phot_g_mean_mag``,
        ``angular_distance_arcsec``, and optionally ``is_target``.
    aperture_radius_arcsec : float
        Radius of the photometric aperture in arcseconds.
    threshold : float
        Contamination ratio above which ``is_contaminated`` is set to
        ``True``.
    config : dict, optional
        Pipeline configuration.

    Returns
    -------
    ContaminationResult
        Contamination analysis result.  If Gaia data for *tic_id* is not
        found, returns a default result with ``contamination_ratio=0.0``.
    """
    if config is not None:
        aperture_radius_arcsec = float(
            get(config, "conditioning.gaia.aperture_radius_arcsec", aperture_radius_arcsec)
        )
        threshold = float(get(config, "conditioning.gaia.threshold", threshold))

    tic_id_str = str(tic_id)

    # ------------------------------------------------------------------
    # Load CSV
    # ------------------------------------------------------------------
    csv_path = Path(gaia_csv_path)
    if not csv_path.exists():
        logger.warning("Gaia CSV not found: %s. Returning default result.", gaia_csv_path)
        return ContaminationResult(tic_id=tic_id_str, threshold=threshold)

    try:
        df = pd.read_csv(str(csv_path), dtype={"tic_id": str, "source_id": str})
    except Exception as exc:
        logger.error("Failed to read Gaia CSV %s: %s", gaia_csv_path, exc)
        return ContaminationResult(tic_id=tic_id_str, threshold=threshold)

    # Normalise column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # ------------------------------------------------------------------
    # Filter to this TIC ID
    # ------------------------------------------------------------------
    if "tic_id" not in df.columns:
        logger.warning("Column 'tic_id' not found in %s.", gaia_csv_path)
        return ContaminationResult(tic_id=tic_id_str, threshold=threshold)

    target_df = df[df["tic_id"].astype(str) == tic_id_str].copy()
    if target_df.empty:
        logger.warning("TIC %s not found in Gaia CSV.", tic_id_str)
        return ContaminationResult(tic_id=tic_id_str, threshold=threshold)

    # ------------------------------------------------------------------
    # Identify the target row
    # ------------------------------------------------------------------
    if "is_target" in target_df.columns:
        target_rows = target_df[target_df["is_target"].astype(int) == 1]
        neighbour_rows = target_df[target_df["is_target"].astype(int) == 0]
    else:
        # Assume the brightest (smallest G mag) entry is the target
        if "phot_g_mean_mag" not in target_df.columns:
            logger.warning("No 'phot_g_mean_mag' column. Returning default.")
            return ContaminationResult(tic_id=tic_id_str, threshold=threshold)
        min_idx = target_df["phot_g_mean_mag"].idxmin()
        target_rows = target_df.loc[[min_idx]]
        neighbour_rows = target_df.drop(index=min_idx)

    if target_rows.empty:
        logger.warning("No target row identified for TIC %s.", tic_id_str)
        return ContaminationResult(tic_id=tic_id_str, threshold=threshold)

    target_g = float(target_rows.iloc[0]["phot_g_mean_mag"])
    target_flux = _gmag_to_flux(target_g)

    # ------------------------------------------------------------------
    # Filter neighbours to aperture
    # ------------------------------------------------------------------
    dist_col = None
    for cname in ["angular_distance_arcsec", "angular_distance", "separation_arcsec"]:
        if cname in neighbour_rows.columns:
            dist_col = cname
            break

    if dist_col is None:
        logger.warning(
            "No angular distance column found; using all neighbours as within aperture."
        )
        within_aperture = neighbour_rows
    else:
        within_aperture = neighbour_rows[
            neighbour_rows[dist_col].astype(float) <= aperture_radius_arcsec
        ]

    # Filter to valid G magnitudes
    within_aperture = within_aperture.dropna(subset=["phot_g_mean_mag"])

    n_neighbors = len(within_aperture)
    if n_neighbors == 0:
        logger.info("TIC %s: no Gaia neighbours within %.1f arcsec.", tic_id_str, aperture_radius_arcsec)
        return ContaminationResult(
            tic_id=tic_id_str,
            contamination_ratio=0.0,
            neighbor_count=0,
            is_contaminated=False,
            threshold=threshold,
            target_flux=target_flux,
            neighbor_flux_sum=0.0,
            neighbor_list=[],
            target_g_mag=target_g,
            aperture_radius_arcsec=aperture_radius_arcsec,
        )

    # ------------------------------------------------------------------
    # Compute contamination
    # ------------------------------------------------------------------
    neighbour_g_mags = np.asarray(within_aperture["phot_g_mean_mag"].values, dtype=float)
    neighbour_fluxes = _gmag_array_to_flux(neighbour_g_mags)
    neighbor_flux_sum = float(np.nansum(neighbour_fluxes))

    if target_flux > 0:
        contamination_ratio = neighbor_flux_sum / target_flux
    else:
        contamination_ratio = 0.0
        logger.warning("Target flux is zero or negative for TIC %s.", tic_id_str)

    is_contaminated = contamination_ratio > threshold

    # Source IDs of neighbours
    sid_col = "source_id" if "source_id" in within_aperture.columns else None
    if sid_col:
        neighbor_ids = [str(sid) for sid in within_aperture[sid_col].tolist()]
    else:
        neighbor_ids = [str(i) for i in range(n_neighbors)]

    logger.info(
        "TIC %s: contamination=%.4f (%.1f %%), n_neighbours=%d, contaminated=%s",
        tic_id_str,
        contamination_ratio,
        contamination_ratio * 100,
        n_neighbors,
        is_contaminated,
    )

    return ContaminationResult(
        tic_id=tic_id_str,
        contamination_ratio=contamination_ratio,
        neighbor_count=n_neighbors,
        is_contaminated=is_contaminated,
        threshold=threshold,
        target_flux=target_flux,
        neighbor_flux_sum=neighbor_flux_sum,
        neighbor_list=neighbor_ids,
        target_g_mag=target_g,
        aperture_radius_arcsec=aperture_radius_arcsec,
    )


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def batch_compute_contamination(
    tic_ids: list,
    gaia_csv_path: str,
    aperture_radius_arcsec: float = DEFAULT_APERTURE_RADIUS_ARCSEC,
    threshold: float = DEFAULT_CONTAMINATION_THRESHOLD,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """Compute contamination for a list of TIC IDs.

    Parameters
    ----------
    tic_ids : list of str
        TIC identifiers to process.
    gaia_csv_path : str
        Path to the Gaia cross-match CSV.
    aperture_radius_arcsec : float
        Aperture radius in arcseconds.
    threshold : float
        Contamination threshold.
    config : dict, optional
        Pipeline configuration.

    Returns
    -------
    pd.DataFrame
        One row per TIC ID with contamination metrics.
    """
    records = []
    for tic_id in tic_ids:
        result = compute_contamination(
            tic_id, gaia_csv_path, aperture_radius_arcsec, threshold, config
        )
        records.append(
            {
                "tic_id": result.tic_id,
                "contamination_ratio": result.contamination_ratio,
                "neighbor_count": result.neighbor_count,
                "is_contaminated": result.is_contaminated,
                "target_g_mag": result.target_g_mag,
                "target_flux": result.target_flux,
                "neighbor_flux_sum": result.neighbor_flux_sum,
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gaia DR3 contamination flagging for TESS targets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Single target
    single = sub.add_parser("single", help="Process a single TIC ID.")
    single.add_argument("tic_id", type=str, help="TIC identifier.")
    single.add_argument("gaia_csv", type=str, help="Path to Gaia cross-match CSV.")
    single.add_argument(
        "--aperture", type=float, default=DEFAULT_APERTURE_RADIUS_ARCSEC,
        help="Aperture radius (arcsec).",
    )
    single.add_argument(
        "--threshold", type=float, default=DEFAULT_CONTAMINATION_THRESHOLD,
        help="Contamination ratio threshold.",
    )
    single.add_argument("--config", type=str, default=None)
    single.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    # Batch
    batch = sub.add_parser("batch", help="Process multiple TIC IDs from a file.")
    batch.add_argument("tic_list", type=str, help="Text file with one TIC ID per line.")
    batch.add_argument("gaia_csv", type=str, help="Path to Gaia cross-match CSV.")
    batch.add_argument(
        "--output", "-o", type=str, default="contamination_results.csv",
        help="Output CSV file.",
    )
    batch.add_argument(
        "--aperture", type=float, default=DEFAULT_APERTURE_RADIUS_ARCSEC,
    )
    batch.add_argument(
        "--threshold", type=float, default=DEFAULT_CONTAMINATION_THRESHOLD,
    )
    batch.add_argument("--config", type=str, default=None)
    batch.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    return parser


def main(argv: Optional[list] = None) -> None:
    """Entry point for the Gaia contamination CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config: Optional[dict] = None
    if args.config:
        config = load_config(args.config)

    if args.command == "single":
        result = compute_contamination(
            args.tic_id, args.gaia_csv,
            aperture_radius_arcsec=args.aperture,
            threshold=args.threshold,
            config=config,
        )
        print(f"TIC {result.tic_id} contamination results:")
        print(f"  Target G mag          : {result.target_g_mag:.3f}")
        print(f"  Target flux (rel)     : {result.target_flux:.6e}")
        print(f"  Neighbour count       : {result.neighbor_count}")
        print(f"  Neighbour flux sum    : {result.neighbor_flux_sum:.6e}")
        print(f"  Contamination ratio   : {result.contamination_ratio:.4f} ({result.contamination_ratio*100:.2f}%)")
        print(f"  Is contaminated?      : {result.is_contaminated}")
        if result.neighbor_list:
            print(f"  Neighbour source IDs  : {', '.join(result.neighbor_list[:5])}" +
                  (f" ... (+{len(result.neighbor_list)-5} more)" if len(result.neighbor_list) > 5 else ""))

    elif args.command == "batch":
        tic_ids = Path(args.tic_list).read_text().splitlines()
        tic_ids = [t.strip() for t in tic_ids if t.strip()]
        logger.info("Processing %d TIC IDs.", len(tic_ids))
        df = batch_compute_contamination(
            tic_ids, args.gaia_csv,
            aperture_radius_arcsec=args.aperture,
            threshold=args.threshold,
            config=config,
        )
        df.to_csv(args.output, index=False)
        n_flagged = int(df["is_contaminated"].sum())
        print(f"Processed {len(df)} targets. {n_flagged} flagged as contaminated.")
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
