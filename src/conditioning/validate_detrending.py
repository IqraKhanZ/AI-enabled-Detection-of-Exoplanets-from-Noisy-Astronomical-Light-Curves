"""
validate_detrending.py
======================
Visual validation of detrending output against known planet transits.

Reads confirmed-planet TIC IDs from the labels CSV, loads raw and
detrended light curves for each, and generates 4-panel comparison
figures:
  1. Raw flux
  2. GP / wavelet trend model
  3. Detrended flux
  4. Phase-folded (BLS best period) detrended flux

Figures are saved to ``reports/detrending_validation/`` in the
project root.

Author: Exoplanet Detection Pipeline
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
# Default paths / style
# ---------------------------------------------------------------------------
_DARK_STYLE = "dark_background"
_DPI = 150
_FIGSIZE = (16, 10)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _find_raw_lc(tic_id: str, data_root: Path) -> Optional[Path]:
    """Search for a raw light curve file for *tic_id*.

    Tries common naming conventions in *data_root* and its subdirectories.

    Parameters
    ----------
    tic_id : str
        TIC identifier.
    data_root : Path
        Root directory to search.

    Returns
    -------
    Path or None
        Path to the found file, or None.
    """
    patterns = [
        f"tic{tic_id}*.fits",
        f"tic{tic_id}*.npz",
        f"*{tic_id}*.fits",
        f"*{tic_id}*.npz",
        f"TIC{tic_id}*.fits",
    ]
    for pat in patterns:
        matches = sorted(data_root.rglob(pat))
        if matches:
            return matches[0]
    return None


def _find_detrended_lc(tic_id: str, data_root: Path) -> Optional[Path]:
    """Search for a detrended light curve file for *tic_id*."""
    patterns = [
        f"tic{tic_id}*_detrended*.npz",
        f"tic{tic_id}*_gp*.npz",
        f"*{tic_id}*detrend*.npz",
    ]
    for pat in patterns:
        matches = sorted(data_root.rglob(pat))
        if matches:
            return matches[0]
    return None


def _load_fits_lc(path: Path):
    """Load time, flux, flux_err from a FITS or .npz file."""
    if path.suffix == ".npz":
        d = np.load(str(path), allow_pickle=False)
        time = d["time"]
        flux = d.get("flux", d.get("detrended_flux", d.get("FLUX", None)))
        trend = d.get("trend_model", d.get("gp_mean", None))
        err = d.get("flux_err", d.get("FLUX_ERR", None))
        if flux is None:
            raise ValueError(f"No flux key found in {path}")
        return (
            np.asarray(time, dtype=float),
            np.asarray(flux, dtype=float),
            np.asarray(err, dtype=float) if err is not None else np.zeros_like(flux),
            np.asarray(trend, dtype=float) if trend is not None else None,
        )
    # FITS
    from astropy.io import fits as af
    with af.open(str(path)) as hdul:
        for hdu in hdul:
            if hasattr(hdu, "columns") and hdu.data is not None and hdu.columns is not None:
                cols = [c.name for c in hdu.columns]
                t = np.asarray(hdu.data["TIME"], dtype=float)
                f_col = "PDCSAP_FLUX" if "PDCSAP_FLUX" in cols else "FLUX"
                e_col = "PDCSAP_FLUX_ERR" if "PDCSAP_FLUX_ERR" in cols else "FLUX_ERR"
                trend_col = "TREND_MODEL" if "TREND_MODEL" in cols else None
                f = np.asarray(hdu.data[f_col], dtype=float)
                e = np.asarray(hdu.data[e_col], dtype=float) if e_col in cols else np.zeros_like(f)
                trend = np.asarray(hdu.data[trend_col], dtype=float) if trend_col else None
                ok = np.isfinite(t) & np.isfinite(f)
                return t[ok], f[ok], e[ok], trend[ok] if trend is not None else None
    raise ValueError(f"Cannot read {path}")


def _phase_fold_simple(time, flux, period, t0):
    """Simple phase fold for plotting."""
    phase = ((time - t0) % period) / period
    phase = np.where(phase >= 0.5, phase - 1.0, phase)
    idx = np.argsort(phase)
    return phase[idx], flux[idx]


def _run_bls_minimal(time, flux, flux_err):
    """Run BLS and return best_period, best_t0."""
    from astropy.timeseries import BoxLeastSquares
    import astropy.units as u
    ok = np.isfinite(time) & np.isfinite(flux)
    t, f, e = time[ok], flux[ok], flux_err[ok] if flux_err is not None else np.ones_like(flux[ok]) * 1e-3
    periods = np.linspace(0.5, 15.0, 5000)
    durations = np.linspace(0.02, 0.3, 20)
    try:
        bls = BoxLeastSquares(t * u.day, f, dy=e)
        result = bls.power(periods * u.day, durations * u.day)
        best_idx = int(np.argmax(result.power))
        return float(result.period[best_idx].value), float(result.transit_time[best_idx].value)
    except Exception as exc:
        logger.warning("BLS failed in validate_detrending: %s", exc)
        return float(np.ptp(time) / 3.0), float(time[0])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _make_validation_plot(
    tic_id: str,
    raw_time: np.ndarray,
    raw_flux: np.ndarray,
    trend_model: Optional[np.ndarray],
    detrended_time: np.ndarray,
    detrended_flux: np.ndarray,
    detrended_flux_err: np.ndarray,
    output_path: Path,
) -> None:
    """Generate and save the 4-panel validation figure.

    Parameters
    ----------
    tic_id : str
        TIC identifier (used for title).
    raw_time, raw_flux : np.ndarray
        Raw light curve data.
    trend_model : np.ndarray or None
        GP or wavelet trend model. If None, panel 2 shows a flat line.
    detrended_time, detrended_flux, detrended_flux_err : np.ndarray
        Detrended light curve data.
    output_path : Path
        File path to save the figure.
    """
    plt.style.use(_DARK_STYLE)
    fig, axes = plt.subplots(4, 1, figsize=_FIGSIZE, sharex=False)
    fig.suptitle(f"Detrending Validation — TIC {tic_id}", fontsize=14, color="white")

    # ---- Panel 1: Raw flux ----
    ax = axes[0]
    ok_raw = np.isfinite(raw_flux)
    ax.scatter(
        raw_time[ok_raw], raw_flux[ok_raw],
        s=1, alpha=0.5, color="steelblue", label="Raw flux",
    )
    if trend_model is not None:
        ok_t = np.isfinite(trend_model)
        ax.plot(raw_time[ok_t], trend_model[ok_t], color="tomato", lw=1.5, label="Trend model")
    ax.set_ylabel("Flux", color="white", fontsize=9)
    ax.set_title("Raw flux + trend model", color="white", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(labelsize=7, colors="white")

    # ---- Panel 2: Trend model ----
    ax = axes[1]
    if trend_model is not None:
        ok_t = np.isfinite(trend_model)
        ax.plot(raw_time[ok_t], trend_model[ok_t], color="tomato", lw=1.5)
        ax.set_title("Trend model only", color="white", fontsize=9)
    else:
        ax.axhline(1.0, color="grey", lw=1)
        ax.set_title("No trend model available", color="white", fontsize=9)
    ax.set_ylabel("Trend", color="white", fontsize=9)
    ax.tick_params(labelsize=7, colors="white")

    # ---- Panel 3: Detrended flux ----
    ax = axes[2]
    ok_det = np.isfinite(detrended_flux)
    ax.scatter(
        detrended_time[ok_det], detrended_flux[ok_det],
        s=1, alpha=0.5, color="gold", label="Detrended",
    )
    ax.axhline(float(np.nanmedian(detrended_flux)), color="grey", lw=0.8, linestyle="--")
    ax.set_ylabel("Detrended flux", color="white", fontsize=9)
    ax.set_title("Detrended flux", color="white", fontsize=9)
    ax.tick_params(labelsize=7, colors="white")

    # ---- Panel 4: Phase-folded ----
    ax = axes[3]
    try:
        ok_det2 = np.isfinite(detrended_flux) & np.isfinite(detrended_time)
        med_f = float(np.nanmedian(detrended_flux[ok_det2]))
        norm_flux = detrended_flux[ok_det2] / med_f if med_f != 0 else detrended_flux[ok_det2]
        period, t0 = _run_bls_minimal(
            detrended_time[ok_det2], norm_flux,
            detrended_flux_err[ok_det2] if detrended_flux_err is not None else None,
        )
        phase, flux_folded = _phase_fold_simple(
            detrended_time[ok_det2], norm_flux, period, t0
        )
        ax.scatter(phase, flux_folded, s=1, alpha=0.3, color="cyan")
        # Median bin
        bins = np.linspace(-0.5, 0.5, 100)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_medians = np.array([
            float(np.median(flux_folded[(phase >= bins[i]) & (phase < bins[i + 1])]))
            if np.any((phase >= bins[i]) & (phase < bins[i + 1]))
            else float(np.nanmedian(flux_folded))
            for i in range(len(bins) - 1)
        ])
        ax.plot(bin_centers, bin_medians, color="white", lw=1.5)
        ax.set_title(f"Phase-folded (P={period:.4f} d)", color="white", fontsize=9)
    except Exception as exc:
        logger.warning("Phase-fold panel failed for TIC %s: %s", tic_id, exc)
        ax.set_title("Phase-fold failed", color="white", fontsize=9)

    ax.set_xlabel("Phase", color="white", fontsize=9)
    ax.set_ylabel("Norm. flux", color="white", fontsize=9)
    ax.tick_params(labelsize=7, colors="white")

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=_DPI, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    logger.info("Saved validation plot to %s", output_path)


# ---------------------------------------------------------------------------
# Main validation routine
# ---------------------------------------------------------------------------


def validate_detrending(
    labels_csv: str,
    raw_data_dir: str,
    detrended_data_dir: str,
    output_dir: Optional[str] = None,
    n_samples: int = 20,
    config: Optional[dict] = None,
) -> None:
    """Generate detrending validation plots for confirmed planets.

    Parameters
    ----------
    labels_csv : str
        Path to the labels CSV with columns ``tic_id`` and ``label`` (or
        ``disposition``).  Only rows with label 0 (PLANET) are used.
    raw_data_dir : str
        Directory containing raw FITS or .npz light curve files.
    detrended_data_dir : str
        Directory containing detrended .npz files.
    output_dir : str, optional
        Output directory for PNG plots.  Defaults to
        ``<project_root>/reports/detrending_validation/``.
    n_samples : int
        Maximum number of targets to process.
    config : dict, optional
        Pipeline configuration.
    """
    if config is not None:
        n_samples = int(get(config, "validation.n_samples", n_samples))

    proj_root = project_root()
    if output_dir is None:
        output_dir_path = proj_root / "reports" / "detrending_validation"
    else:
        output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    raw_root = Path(raw_data_dir)
    det_root = Path(detrended_data_dir)

    # ------------------------------------------------------------------
    # Load labels
    # ------------------------------------------------------------------
    try:
        labels_df = pd.read_csv(labels_csv, dtype={"tic_id": str})
    except Exception as exc:
        logger.error("Failed to load labels CSV %s: %s", labels_csv, exc)
        return

    labels_df.columns = [c.strip().lower() for c in labels_df.columns]

    label_col = None
    for cname in ["label", "disposition", "class", "category"]:
        if cname in labels_df.columns:
            label_col = cname
            break

    if label_col is None:
        logger.error("No label column found in %s", labels_csv)
        return

    # Filter to confirmed planets (label == 0)
    planet_df = labels_df[labels_df[label_col].astype(int) == 0]
    if planet_df.empty:
        logger.warning("No confirmed-planet entries found (label=0).")
        return

    tic_ids = planet_df["tic_id"].astype(str).tolist()
    logger.info("Found %d confirmed planets; will process up to %d.", len(tic_ids), n_samples)

    processed = 0
    skipped = 0
    for tic_id in tic_ids:
        if processed >= n_samples:
            break

        raw_path = _find_raw_lc(tic_id, raw_root)
        det_path = _find_detrended_lc(tic_id, det_root)

        if raw_path is None:
            logger.warning("No raw LC found for TIC %s. Skipping.", tic_id)
            skipped += 1
            continue

        try:
            raw_time, raw_flux, raw_err, raw_trend = _load_fits_lc(raw_path)
        except Exception as exc:
            logger.warning("Cannot load raw LC for TIC %s: %s", tic_id, exc)
            skipped += 1
            continue

        if det_path is not None:
            try:
                det_time, det_flux, det_err, det_trend = _load_fits_lc(det_path)
                trend_model = raw_trend if raw_trend is not None else det_trend
            except Exception as exc:
                logger.warning("Cannot load detrended LC for TIC %s: %s", tic_id, exc)
                det_time, det_flux, det_err = raw_time, raw_flux, raw_err
                trend_model = raw_trend
        else:
            logger.info("No detrended LC found for TIC %s. Using raw.", tic_id)
            det_time, det_flux, det_err = raw_time, raw_flux, raw_err
            trend_model = raw_trend

        # Normalise
        med_raw = float(np.nanmedian(raw_flux))
        if med_raw != 0:
            raw_flux_n = raw_flux / med_raw
            trend_n = (raw_trend / med_raw) if raw_trend is not None else None
        else:
            raw_flux_n = raw_flux
            trend_n = raw_trend

        med_det = float(np.nanmedian(det_flux))
        if med_det != 0:
            det_flux_n = det_flux / med_det
            det_err_n = det_err / med_det
        else:
            det_flux_n = det_flux
            det_err_n = det_err

        out_file = output_dir_path / f"tic{tic_id}_validation.png"
        try:
            _make_validation_plot(
                tic_id=tic_id,
                raw_time=raw_time,
                raw_flux=raw_flux_n,
                trend_model=trend_n,
                detrended_time=det_time,
                detrended_flux=det_flux_n,
                detrended_flux_err=det_err_n,
                output_path=out_file,
            )
            processed += 1
        except Exception as exc:
            logger.error("Failed to make plot for TIC %s: %s", tic_id, exc)
            skipped += 1

    logger.info("Validation complete: %d plots saved, %d skipped.", processed, skipped)
    print(f"Validation plots: {processed} saved to {output_dir_path}, {skipped} skipped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate detrending output against known confirmed planets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("labels_csv", type=str, help="Labels CSV with tic_id and label columns.")
    parser.add_argument("raw_data_dir", type=str, help="Directory with raw light curve files.")
    parser.add_argument("detrended_data_dir", type=str, help="Directory with detrended .npz files.")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for PNG plots.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=20,
        help="Maximum number of targets to plot.",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: Optional[list] = None) -> None:
    """Entry point for the detrending validation CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config: Optional[dict] = None
    if args.config:
        config = load_config(args.config)

    validate_detrending(
        labels_csv=args.labels_csv,
        raw_data_dir=args.raw_data_dir,
        detrended_data_dir=args.detrended_data_dir,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        config=config,
    )


if __name__ == "__main__":
    main()
