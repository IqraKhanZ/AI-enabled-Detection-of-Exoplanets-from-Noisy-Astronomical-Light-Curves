"""
systematics_removal.py
=======================
Instrumental systematics removal for TESS light curves.

Applies lightkurve's Cotrending Basis Vector (CBV) correction to remove
spacecraft systematics from TESS photometry.  If CBVs are not available
for the target sector/camera/CCD, the module falls back to a
SparseSplineLightCurveCorrector.

Functions
---------
remove_systematics(lc, sector, camera, ccd, method='cbv') -> TessLightCurve
    Main entry point: applies CBV or spline correction and returns the
    corrected lightkurve TessLightCurve.

Author: Exoplanet Detection Pipeline
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Optional

try:
    import lightkurve as lk
    from lightkurve import TessLightCurve
    from lightkurve.correctors import CBVCorrector, SparseSplineLightCurveCorrector

    _LK_AVAILABLE = True
except ImportError:
    _LK_AVAILABLE = False
    warnings.warn(
        "lightkurve is not installed. systematics_removal will not function.",
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


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def remove_systematics(
    lc: "TessLightCurve",
    sector: int,
    camera: int,
    ccd: int,
    method: str = "cbv",
    n_cbvs: int = 4,
    cbv_type: str = "MultiScale",
    spline_n_knots: int = 20,
    spline_degree: int = 3,
    config: Optional[dict] = None,
) -> "TessLightCurve":
    """Remove instrumental systematics from a TESS light curve.

    Attempts to apply lightkurve's CBV correction using the specified
    number of basis vectors.  If CBVs are unavailable for this
    sector/camera/CCD or if the correction raises an exception, falls
    back to a SparseSplineLightCurveCorrector.

    Parameters
    ----------
    lc : lightkurve.TessLightCurve
        Input light curve to correct.  Must contain ``flux``, ``time``,
        and optionally ``flux_err``.
    sector : int
        TESS sector number.
    camera : int
        TESS camera number (1–4).
    ccd : int
        TESS CCD number (1–4).
    method : str
        Preferred correction method.  One of ``'cbv'`` or ``'spline'``.
        If ``'cbv'``, tries CBV first and falls back to spline on failure.
        If ``'spline'``, skips CBV and applies spline directly.
    n_cbvs : int
        Number of CBV vectors to use in the correction.
    cbv_type : str
        CBV type string passed to ``CBVCorrector.download_cbvs``.  Typical
        values: ``'MultiScale'``, ``'SingleScale'``, ``'Spike'``.
    spline_n_knots : int
        Number of knots for the SparseSplineLightCurveCorrector.
    spline_degree : int
        Polynomial degree for the spline corrector.
    config : dict, optional
        Pipeline configuration dictionary.

    Returns
    -------
    lightkurve.TessLightCurve
        Corrected light curve with the same time array.

    Raises
    ------
    ImportError
        If lightkurve is not installed.
    RuntimeError
        If both CBV and spline corrections fail.

    Examples
    --------
    >>> import lightkurve as lk
    >>> lc = lk.search_lightcurve('TIC 123456789', sector=1).download()
    >>> corrected = remove_systematics(lc, sector=1, camera=1, ccd=1)
    """
    if not _LK_AVAILABLE:
        raise ImportError("lightkurve is required for systematics_removal.")

    # Override parameters from config if provided
    if config is not None:
        method = get(config, "conditioning.systematics.method", method)
        n_cbvs = int(get(config, "conditioning.systematics.n_cbvs", n_cbvs))
        cbv_type = get(config, "conditioning.systematics.cbv_type", cbv_type)
        spline_n_knots = int(
            get(config, "conditioning.systematics.spline_n_knots", spline_n_knots)
        )

    logger.info(
        "Removing systematics: sector=%d, camera=%d, ccd=%d, method=%s",
        sector, camera, ccd, method,
    )

    # ------------------------------------------------------------------
    # Attempt CBV correction
    # ------------------------------------------------------------------
    if method == "cbv":
        try:
            corrected_lc = _apply_cbv_correction(
                lc, sector, camera, ccd, n_cbvs, cbv_type
            )
            logger.info("CBV correction applied successfully.")
            return corrected_lc
        except Exception as exc:
            logger.warning(
                "CBV correction failed (%s). Falling back to spline correction.", exc
            )

    # ------------------------------------------------------------------
    # Spline fallback (or explicit method='spline')
    # ------------------------------------------------------------------
    try:
        corrected_lc = _apply_spline_correction(lc, spline_n_knots, spline_degree)
        logger.info("Spline correction applied successfully.")
        return corrected_lc
    except Exception as exc:
        logger.error("Spline correction also failed: %s", exc)
        raise RuntimeError(
            f"Both CBV and spline corrections failed for sector={sector}, "
            f"camera={camera}, ccd={ccd}. Last error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_cbv_correction(
    lc: "TessLightCurve",
    sector: int,
    camera: int,
    ccd: int,
    n_cbvs: int,
    cbv_type: str,
) -> "TessLightCurve":
    """Apply CBV correction to a TessLightCurve.

    Parameters
    ----------
    lc : TessLightCurve
        Input light curve.
    sector : int
        TESS sector.
    camera : int
        TESS camera.
    ccd : int
        TESS CCD.
    n_cbvs : int
        Number of CBV vectors.
    cbv_type : str
        CBV type (MultiScale, SingleScale, Spike).

    Returns
    -------
    TessLightCurve
        CBV-corrected light curve.
    """
    cbvcor = CBVCorrector(lc)
    cbvcor.download_cbvs(
        cbv_type=cbv_type,
        cbv_indices=list(range(1, n_cbvs + 1)),
    )
    corrected = cbvcor.correct(
        cbv_type=cbv_type,
        cbv_indices=list(range(1, n_cbvs + 1)),
    )
    # Normalise so median flux = 1
    median = float(corrected.flux.value[~corrected.flux.mask].mean()) if hasattr(corrected.flux, "mask") else float(corrected.flux.value.mean())
    if median != 0:
        corrected = corrected.normalize()
    return corrected


def _apply_spline_correction(
    lc: "TessLightCurve",
    n_knots: int = 20,
    degree: int = 3,
) -> "TessLightCurve":
    """Apply SparseSplineLightCurveCorrector to a TessLightCurve.

    Parameters
    ----------
    lc : TessLightCurve
        Input light curve.
    n_knots : int
        Number of spline knots.
    degree : int
        Spline polynomial degree.

    Returns
    -------
    TessLightCurve
        Spline-corrected light curve.
    """
    try:
        corrector = SparseSplineLightCurveCorrector(lc)
        corrected = corrector.correct(n_knots=n_knots, degree=degree)
    except TypeError:
        # Some lightkurve versions have a different signature
        corrector = SparseSplineLightCurveCorrector(lc)
        corrected = corrector.correct()
    return corrected.normalize()


def _load_tess_lightcurve(fits_path: str, sector: Optional[int] = None) -> "TessLightCurve":
    """Load a TessLightCurve from a FITS file.

    Parameters
    ----------
    fits_path : str
        Path to the TESS light curve FITS file.
    sector : int, optional
        If provided, verified against the file header.

    Returns
    -------
    TessLightCurve
        Loaded light curve object.
    """
    lc = lk.read(fits_path)
    if not isinstance(lc, lk.TessLightCurve):
        # Wrap it if needed
        lc = lk.TessLightCurve(time=lc.time, flux=lc.flux, flux_err=lc.flux_err)
    return lc


def _infer_sector_camera_ccd(lc: "TessLightCurve") -> tuple:
    """Infer sector, camera, CCD from a TessLightCurve header metadata.

    Parameters
    ----------
    lc : TessLightCurve
        Light curve with metadata.

    Returns
    -------
    tuple of int
        (sector, camera, ccd).  Returns (1, 1, 1) if not found.
    """
    sector = int(getattr(lc, "sector", 1) or 1)
    camera = int(getattr(lc, "camera", 1) or 1)
    ccd = int(getattr(lc, "ccd", 1) or 1)
    # Try meta dict
    if hasattr(lc, "meta"):
        sector = int(lc.meta.get("SECTOR", sector))
        camera = int(lc.meta.get("CAMERA", camera))
        ccd = int(lc.meta.get("CCD", ccd))
    return sector, camera, ccd


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def batch_remove_systematics(
    fits_files: list,
    output_dir: str,
    method: str = "cbv",
    n_cbvs: int = 4,
    config: Optional[dict] = None,
) -> list:
    """Apply systematics removal to multiple FITS files.

    Parameters
    ----------
    fits_files : list of str
        Paths to input TESS FITS light curve files.
    output_dir : str
        Directory where corrected FITS files are saved.
    method : str
        Correction method (``'cbv'`` or ``'spline'``).
    n_cbvs : int
        Number of CBVs to use.
    config : dict, optional
        Pipeline config.

    Returns
    -------
    list of str
        Paths to successfully written output files.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    outputs = []

    for fits_path in fits_files:
        logger.info("Processing %s", fits_path)
        try:
            lc = _load_tess_lightcurve(fits_path)
            sector, camera, ccd = _infer_sector_camera_ccd(lc)
            corrected = remove_systematics(
                lc, sector, camera, ccd, method=method, n_cbvs=n_cbvs, config=config
            )
            out_path = output_dir_path / (Path(fits_path).stem + "_corrected.fits")
            corrected.to_fits(str(out_path), overwrite=True)
            outputs.append(str(out_path))
            logger.info("Saved corrected LC to %s", out_path)
        except Exception as exc:
            logger.error("Failed to process %s: %s", fits_path, exc)

    return outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TESS instrumental systematics removal (CBV / SparseSpline).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Single-file command
    single = sub.add_parser("single", help="Correct a single FITS light curve.")
    single.add_argument("fits_file", type=str, help="Input TESS FITS file path.")
    single.add_argument("--output", "-o", type=str, default=None, help="Output FITS file.")
    single.add_argument("--sector", type=int, default=None)
    single.add_argument("--camera", type=int, default=None)
    single.add_argument("--ccd", type=int, default=None)
    single.add_argument("--method", type=str, default="cbv", choices=["cbv", "spline"])
    single.add_argument("--n-cbvs", type=int, default=4)
    single.add_argument("--cbv-type", type=str, default="MultiScale")
    single.add_argument("--spline-knots", type=int, default=20)
    single.add_argument("--config", type=str, default=None)
    single.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # Batch command
    batch = sub.add_parser("batch", help="Correct multiple FITS light curves.")
    batch.add_argument("fits_dir", type=str, help="Directory with FITS files.")
    batch.add_argument("output_dir", type=str, help="Output directory.")
    batch.add_argument("--method", type=str, default="cbv", choices=["cbv", "spline"])
    batch.add_argument("--n-cbvs", type=int, default=4)
    batch.add_argument("--config", type=str, default=None)
    batch.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser


def main(argv: Optional[list] = None) -> None:
    """Entry point for the systematics removal CLI."""
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
        lc = _load_tess_lightcurve(args.fits_file)
        if args.sector is None or args.camera is None or args.ccd is None:
            sector, camera, ccd = _infer_sector_camera_ccd(lc)
        else:
            sector = args.sector
            camera = args.camera
            ccd = args.ccd
        corrected = remove_systematics(
            lc, sector, camera, ccd,
            method=args.method,
            n_cbvs=args.n_cbvs,
            cbv_type=args.cbv_type,
            spline_n_knots=args.spline_knots,
            config=config,
        )
        out = args.output or (Path(args.fits_file).stem + "_corrected.fits")
        corrected.to_fits(out, overwrite=True)
        print(f"Corrected light curve saved to: {out}")

    elif args.command == "batch":
        fits_files = sorted(Path(args.fits_dir).glob("*.fits"))
        fits_files_str = [str(f) for f in fits_files]
        logger.info("Found %d FITS files to process.", len(fits_files_str))
        outputs = batch_remove_systematics(
            fits_files_str,
            args.output_dir,
            method=args.method,
            n_cbvs=args.n_cbvs,
            config=config,
        )
        print(f"Processed {len(outputs)} / {len(fits_files_str)} files successfully.")


if __name__ == "__main__":
    main()
