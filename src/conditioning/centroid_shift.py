"""
centroid_shift.py
=================
Pixel-level centroid-shift analysis from TESS Target Pixel Files (TPFs).

Computes the flux-weighted centroid position for in-transit and
out-of-transit cadences and quantifies the centroid shift in both
pixel and arcsecond units (TESS pixel scale: 21 arcsec/pixel).

A significant centroid shift during transit suggests the transit signal
originates from a background eclipsing binary rather than the target star.

Classes / Dataclasses
---------------------
CentroidResult
    Container for centroid shift results.

Functions
---------
compute_centroid(pixel_data, time_mask) -> (row, col)
    Flux-weighted centroid for a cadence subset.
extract_centroid_shift(tpf_path, transit_mask) -> CentroidResult
    Main entry: loads TPF and returns a CentroidResult.

Author: Exoplanet Detection Pipeline
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import lightkurve as lk
    from lightkurve import TessTargetPixelFile

    _LK_AVAILABLE = True
except ImportError:
    _LK_AVAILABLE = False
    warnings.warn(
        "lightkurve is not installed. centroid_shift will not function.",
        ImportWarning,
        stacklevel=2,
    )

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

# TESS pixel scale in arcseconds per pixel
TESS_PIXEL_SCALE_ARCSEC = 21.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CentroidResult:
    """Results from centroid shift analysis.

    Attributes
    ----------
    centroid_shift_pixels : float
        Euclidean centroid displacement (pixels) between in-transit and
        out-of-transit centroids.
    centroid_shift_arcsec : float
        Centroid displacement in arcseconds.
    in_transit_centroid : tuple of float
        (row, col) centroid during in-transit cadences.
    oot_centroid : tuple of float
        (row, col) centroid during out-of-transit cadences.
    all_centroids_row : np.ndarray
        Centroid row for every cadence.
    all_centroids_col : np.ndarray
        Centroid col for every cadence.
    n_in_transit : int
        Number of in-transit cadences used.
    n_oot : int
        Number of out-of-transit cadences used.
    pixel_scale_arcsec : float
        TESS pixel scale used (arcsec/pixel).
    """

    centroid_shift_pixels: float = 0.0
    centroid_shift_arcsec: float = 0.0
    in_transit_centroid: Tuple[float, float] = (0.0, 0.0)
    oot_centroid: Tuple[float, float] = (0.0, 0.0)
    all_centroids_row: np.ndarray = field(default_factory=lambda: np.array([]))
    all_centroids_col: np.ndarray = field(default_factory=lambda: np.array([]))
    n_in_transit: int = 0
    n_oot: int = 0
    pixel_scale_arcsec: float = TESS_PIXEL_SCALE_ARCSEC


# ---------------------------------------------------------------------------
# Core centroid computation
# ---------------------------------------------------------------------------


def compute_centroid(
    pixel_data: np.ndarray,
) -> Tuple[float, float]:
    """Compute the flux-weighted centroid of a 2-D pixel flux array.

    Parameters
    ----------
    pixel_data : np.ndarray
        Stack of pixel frames with shape ``(n_cadences, n_rows, n_cols)``,
        **or** a single 2-D frame of shape ``(n_rows, n_cols)``.
        NaN and negative pixels are treated as zero weight.

    Returns
    -------
    tuple of float
        ``(centroid_row, centroid_col)`` — flux-weighted centre of light.
        Returns ``(nan, nan)`` if the total flux is zero.
    """
    if pixel_data.ndim == 3:
        # Sum over cadences
        frame = np.nansum(pixel_data, axis=0)
    else:
        frame = pixel_data.copy()

    # Clip negatives to zero
    frame = np.where(frame > 0, frame, 0.0)
    total = float(np.nansum(frame))

    if total == 0.0:
        return (float("nan"), float("nan"))

    n_rows, n_cols = frame.shape
    row_coords = np.arange(n_rows)[:, np.newaxis] * np.ones((1, n_cols))
    col_coords = np.arange(n_cols)[np.newaxis, :] * np.ones((n_rows, 1))

    cen_row = float(np.nansum(frame * row_coords) / total)
    cen_col = float(np.nansum(frame * col_coords) / total)
    return (cen_row, cen_col)


def compute_per_cadence_centroids(
    pixel_cube: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute flux-weighted centroid for each cadence.

    Parameters
    ----------
    pixel_cube : np.ndarray
        Array of shape ``(n_cadences, n_rows, n_cols)``.

    Returns
    -------
    tuple of np.ndarray
        ``(centroid_rows, centroid_cols)`` arrays of length ``n_cadences``.
    """
    n_cadences = pixel_cube.shape[0]
    cen_rows = np.full(n_cadences, np.nan)
    cen_cols = np.full(n_cadences, np.nan)

    for i in range(n_cadences):
        frame = pixel_cube[i]
        row, col = compute_centroid(frame)
        cen_rows[i] = row
        cen_cols[i] = col

    return cen_rows, cen_cols


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------


def extract_centroid_shift(
    tpf_path: str,
    transit_mask: Optional[np.ndarray] = None,
    pixel_scale_arcsec: float = TESS_PIXEL_SCALE_ARCSEC,
    config: Optional[dict] = None,
) -> Optional[CentroidResult]:
    """Extract centroid shift from a TESS Target Pixel File.

    Parameters
    ----------
    tpf_path : str
        Path to the TESS TPF FITS file.
    transit_mask : np.ndarray of bool, optional
        Boolean array of length ``n_cadences`` with ``True`` for in-transit
        cadences.  If ``None``, returns the full-time centroid only, and
        ``centroid_shift_pixels`` will be 0.
    pixel_scale_arcsec : float
        TESS pixel scale in arcseconds per pixel.
    config : dict, optional
        Pipeline configuration.

    Returns
    -------
    CentroidResult or None
        Centroid shift results, or ``None`` if the TPF cannot be loaded.
    """
    if not _LK_AVAILABLE:
        logger.error("lightkurve is not installed. Cannot extract centroids.")
        return None

    if config is not None:
        pixel_scale_arcsec = float(
            get(config, "conditioning.centroid.pixel_scale_arcsec", pixel_scale_arcsec)
        )

    # ------------------------------------------------------------------
    # Load TPF
    # ------------------------------------------------------------------
    tpf_path_obj = Path(tpf_path)
    if not tpf_path_obj.exists():
        logger.warning("TPF file not found: %s. Returning None.", tpf_path)
        return None

    try:
        tpf = lk.read(str(tpf_path_obj))
    except Exception as exc:
        logger.warning(
            "Failed to open TPF at %s: %s. Returning None.", tpf_path, exc
        )
        return None

    if not isinstance(tpf, TessTargetPixelFile):
        try:
            # Force wrap
            tpf = TessTargetPixelFile(str(tpf_path_obj))
        except Exception as exc:
            logger.warning("Cannot read as TessTargetPixelFile: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Extract pixel data cube
    # ------------------------------------------------------------------
    try:
        flux_cube = np.asarray(tpf.flux.value, dtype=float)
    except Exception as exc:
        logger.warning("Cannot read flux from TPF: %s", exc)
        return None

    n_cadences, n_rows, n_cols = flux_cube.shape
    logger.info(
        "TPF loaded: %d cadences, %dx%d pixels.", n_cadences, n_rows, n_cols
    )

    # ------------------------------------------------------------------
    # Validate/align transit_mask
    # ------------------------------------------------------------------
    if transit_mask is not None:
        transit_mask = np.asarray(transit_mask, dtype=bool)
        if len(transit_mask) != n_cadences:
            logger.warning(
                "transit_mask length %d != TPF cadences %d. Trimming/padding.",
                len(transit_mask),
                n_cadences,
            )
            if len(transit_mask) > n_cadences:
                transit_mask = transit_mask[:n_cadences]
            else:
                pad = np.zeros(n_cadences - len(transit_mask), dtype=bool)
                transit_mask = np.concatenate([transit_mask, pad])
        oot_mask = ~transit_mask
    else:
        transit_mask = np.zeros(n_cadences, dtype=bool)
        oot_mask = np.ones(n_cadences, dtype=bool)

    # ------------------------------------------------------------------
    # Quality mask (remove bad cadences)
    # ------------------------------------------------------------------
    try:
        quality = np.asarray(tpf.quality.value, dtype=int)
        good = quality == 0
    except Exception:
        good = np.ones(n_cadences, dtype=bool)

    transit_mask = transit_mask & good
    oot_mask = oot_mask & good

    n_in = int(transit_mask.sum())
    n_oot = int(oot_mask.sum())
    logger.info("In-transit cadences: %d, OOT cadences: %d", n_in, n_oot)

    # ------------------------------------------------------------------
    # Compute per-cadence centroids
    # ------------------------------------------------------------------
    all_rows, all_cols = compute_per_cadence_centroids(flux_cube)

    # ------------------------------------------------------------------
    # Average centroids for transit / OOT
    # ------------------------------------------------------------------
    if n_in > 0:
        in_row = float(np.nanmean(all_rows[transit_mask]))
        in_col = float(np.nanmean(all_cols[transit_mask]))
    else:
        in_row = float(np.nanmean(all_rows))
        in_col = float(np.nanmean(all_cols))
        logger.warning("No in-transit cadences found; using full-time centroid.")

    if n_oot > 0:
        oot_row = float(np.nanmean(all_rows[oot_mask]))
        oot_col = float(np.nanmean(all_cols[oot_mask]))
    else:
        oot_row = float(np.nanmean(all_rows))
        oot_col = float(np.nanmean(all_cols))
        logger.warning("No OOT cadences found; using full-time centroid.")

    # ------------------------------------------------------------------
    # Compute shift
    # ------------------------------------------------------------------
    d_row = in_row - oot_row
    d_col = in_col - oot_col
    shift_pixels = float(np.sqrt(d_row ** 2 + d_col ** 2))
    shift_arcsec = shift_pixels * pixel_scale_arcsec

    logger.info(
        "Centroid shift: %.4f pixels = %.2f arcsec", shift_pixels, shift_arcsec
    )

    return CentroidResult(
        centroid_shift_pixels=shift_pixels,
        centroid_shift_arcsec=shift_arcsec,
        in_transit_centroid=(in_row, in_col),
        oot_centroid=(oot_row, oot_col),
        all_centroids_row=all_rows,
        all_centroids_col=all_cols,
        n_in_transit=n_in,
        n_oot=n_oot,
        pixel_scale_arcsec=pixel_scale_arcsec,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Centroid shift analysis from TESS Target Pixel Files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("tpf_file", type=str, help="Path to TESS TPF FITS file.")
    parser.add_argument(
        "--transit-mask-npz", type=str, default=None,
        help=".npz file with boolean 'transit_mask' key.",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output .npz file for centroid arrays. Defaults to <tpf>_centroids.npz.",
    )
    parser.add_argument(
        "--pixel-scale", type=float, default=TESS_PIXEL_SCALE_ARCSEC,
        help="TESS pixel scale in arcsec/pixel.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Pipeline config YAML.",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: Optional[list] = None) -> None:
    """Entry point for the centroid shift CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config: Optional[dict] = None
    if args.config:
        config = load_config(args.config)

    transit_mask: Optional[np.ndarray] = None
    if args.transit_mask_npz:
        tm = np.load(args.transit_mask_npz, allow_pickle=False)
        transit_mask = tm["transit_mask"].astype(bool)

    result = extract_centroid_shift(
        args.tpf_file,
        transit_mask=transit_mask,
        pixel_scale_arcsec=args.pixel_scale,
        config=config,
    )

    if result is None:
        print("ERROR: Could not extract centroid from TPF. Check logs.")
        return

    output = args.output or (Path(args.tpf_file).stem + "_centroids.npz")
    np.savez(
        output,
        centroid_shift_pixels=result.centroid_shift_pixels,
        centroid_shift_arcsec=result.centroid_shift_arcsec,
        in_transit_centroid_row=result.in_transit_centroid[0],
        in_transit_centroid_col=result.in_transit_centroid[1],
        oot_centroid_row=result.oot_centroid[0],
        oot_centroid_col=result.oot_centroid[1],
        all_centroids_row=result.all_centroids_row,
        all_centroids_col=result.all_centroids_col,
    )
    print(f"Centroid result saved to: {output}")
    print(f"  Shift (pixels)  : {result.centroid_shift_pixels:.4f}")
    print(f"  Shift (arcsec)  : {result.centroid_shift_arcsec:.2f}")
    print(f"  In-transit cen  : ({result.in_transit_centroid[0]:.3f}, {result.in_transit_centroid[1]:.3f})")
    print(f"  OOT centroid    : ({result.oot_centroid[0]:.3f}, {result.oot_centroid[1]:.3f})")
    print(f"  N in-transit    : {result.n_in_transit}")
    print(f"  N OOT           : {result.n_oot}")


if __name__ == "__main__":
    main()
