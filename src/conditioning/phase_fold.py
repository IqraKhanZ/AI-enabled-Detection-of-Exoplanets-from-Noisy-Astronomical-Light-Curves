"""
phase_fold.py
=============
BLS periodogram search and phase-folding utility.

Uses ``astropy.timeseries.BoxLeastSquares`` to identify the best-fit
transit period, epoch, duration, and depth from a cleaned light curve.
After BLS, the module produces three standard Astronet-style light curve
representations:

* **global_view** – 200-bin median-binned phase-folded flux over the full
  orbital period.
* **local_view** – 50-bin median-binned flux centred on the transit (±1.5×
  transit duration).
* **river_plot** – 2-D array of shape ``(n_cycles, 200)`` where each row
  corresponds to one complete orbital cycle.

All views are normalised so that the transit minimum is 0 and out-of-transit
baseline is 1.

Functions
---------
run_bls(time, flux, flux_err, config) -> dict
    Run BLS and return the best-fit parameters.
phase_fold(time, flux, flux_err, period, t0) -> tuple
    Phase-fold a light curve.
make_global_view(phase, flux, n_bins) -> np.ndarray
    Median-binned global view.
make_local_view(phase, flux, transit_duration, n_bins) -> np.ndarray
    Median-binned local view.
make_river_plot(time, flux, period, t0, n_bins) -> np.ndarray
    2-D river plot.
fold_lightcurve(time, flux, flux_err, config) -> PhaseResult
    Master function that runs BLS and produces all views.

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
from astropy.timeseries import BoxLeastSquares
import astropy.units as u

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
# Default configuration constants
# ---------------------------------------------------------------------------
DEFAULT_PERIOD_MIN = 0.5       # days
DEFAULT_PERIOD_MAX = 15.0      # days
DEFAULT_N_PERIODS = 10_000
DEFAULT_GLOBAL_BINS = 200
DEFAULT_LOCAL_BINS = 50
DEFAULT_LOCAL_HALF_WIDTH = 1.5  # multiples of transit duration


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PhaseResult:
    """All outputs from the BLS search and phase-folding pipeline.

    Attributes
    ----------
    best_period : float
        Best-fit orbital period (days).
    best_t0 : float
        Best-fit mid-transit epoch (days, same time system as input).
    best_duration : float
        Best-fit transit duration (days).
    best_depth : float
        Best-fit transit depth (fractional).
    bls_power : np.ndarray
        BLS power spectrum array.
    bls_periods : np.ndarray
        Period array corresponding to *bls_power*.
    bls_stats : dict
        Dictionary of BLS statistics (depth, depth_err, snr, etc.).
    phase : np.ndarray
        Phase array in [−0.5, 0.5) corresponding to the input time array.
    global_view : np.ndarray
        200-bin normalised global-view array.
    local_view : np.ndarray
        50-bin normalised local-view array.
    river_plot : np.ndarray
        2-D river plot of shape (n_cycles, 200).
    n_transits_observed : int
        Number of complete orbital cycles observed.
    """

    best_period: float = 0.0
    best_t0: float = 0.0
    best_duration: float = 0.0
    best_depth: float = 0.0
    bls_power: np.ndarray = field(default_factory=lambda: np.array([]))
    bls_periods: np.ndarray = field(default_factory=lambda: np.array([]))
    bls_stats: dict = field(default_factory=dict)
    phase: np.ndarray = field(default_factory=lambda: np.array([]))
    global_view: np.ndarray = field(default_factory=lambda: np.zeros(DEFAULT_GLOBAL_BINS))
    local_view: np.ndarray = field(default_factory=lambda: np.zeros(DEFAULT_LOCAL_BINS))
    river_plot: np.ndarray = field(default_factory=lambda: np.zeros((1, DEFAULT_GLOBAL_BINS)))
    n_transits_observed: int = 0


# ---------------------------------------------------------------------------
# BLS search
# ---------------------------------------------------------------------------


def run_bls(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period_min: float = DEFAULT_PERIOD_MIN,
    period_max: float = DEFAULT_PERIOD_MAX,
    n_periods: int = DEFAULT_N_PERIODS,
    duration_grid: Optional[np.ndarray] = None,
    config: Optional[dict] = None,
) -> dict:
    """Run astropy BoxLeastSquares on a light curve.

    Parameters
    ----------
    time : np.ndarray
        Observation times (days).
    flux : np.ndarray
        Normalised flux (dimensionless, baseline ≈ 1).
    flux_err : np.ndarray
        Flux uncertainties.
    period_min : float
        Minimum period to search (days).
    period_max : float
        Maximum period to search (days).
    n_periods : int
        Number of period grid points.
    duration_grid : np.ndarray, optional
        Array of transit durations (days) to search.  Defaults to a
        log-spaced grid from 0.01 to 0.5 days.
    config : dict, optional
        Pipeline configuration.

    Returns
    -------
    dict
        Keys: ``best_period``, ``best_t0``, ``best_duration``,
        ``best_depth``, ``best_depth_err``, ``bls_power``,
        ``bls_periods``, ``snr``, ``log_likelihood``.
    """
    if config is not None:
        period_min = float(get(config, "conditioning.bls.period_min", period_min))
        period_max = float(get(config, "conditioning.bls.period_max", period_max))
        n_periods = int(get(config, "conditioning.bls.n_periods", n_periods))

    if duration_grid is None:
        max_duration = min(0.3, period_min * 0.8)
        duration_grid = np.linspace(0.01, max_duration, 30)

    periods = np.linspace(period_min, period_max, n_periods)

    # Remove NaNs
    ok = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    if ok.sum() < 10:
        logger.warning("Too few finite points for BLS (%d).", int(ok.sum()))
        return _empty_bls_stats(periods)

    t_ok = time[ok]
    f_ok = flux[ok]
    e_ok = flux_err[ok]

    logger.info(
        "Running BLS: %d cadences, periods=[%.2f,%.2f] days, n=%d",
        ok.sum(), period_min, period_max, n_periods,
    )

    try:
        bls = BoxLeastSquares(t_ok * u.day, f_ok, dy=e_ok)
        result = bls.power(periods * u.day, duration_grid * u.day, objective="snr")

        best_idx = int(np.argmax(result.power))
        best_period = float(result.period[best_idx].to(u.day).value)
        best_t0 = float(result.transit_time[best_idx].to(u.day).value)
        best_duration = float(result.duration[best_idx].to(u.day).value)
        best_depth = float(result.depth[best_idx])

        # Compute SNR
        stats = bls.compute_stats(
            result.period[best_idx],
            result.duration[best_idx],
            result.transit_time[best_idx],
        )
        snr = float(stats["depth"][0] / stats["depth"][1]) if stats["depth"][1] != 0 else 0.0
        depth_err = float(stats["depth"][1])

        logger.info(
            "BLS best: period=%.4f days, t0=%.4f, duration=%.4f days, depth=%.6f, SNR=%.2f",
            best_period, best_t0, best_duration, best_depth, snr,
        )

        return {
            "best_period": best_period,
            "best_t0": best_t0,
            "best_duration": best_duration,
            "best_depth": best_depth,
            "best_depth_err": depth_err,
            "bls_power": np.asarray(result.power),
            "bls_periods": np.asarray(result.period.to(u.day).value),
            "snr": snr,
            "log_likelihood": float(np.nanmax(result.log_likelihood)
                                    if hasattr(result, "log_likelihood") else np.nan),
            "stats": stats,
        }

    except Exception as exc:
        logger.error("BLS failed: %s", exc)
        return _empty_bls_stats(periods)


def _empty_bls_stats(periods: np.ndarray) -> dict:
    """Return an empty BLS stats dict."""
    return {
        "best_period": float(np.median(periods)),
        "best_t0": 0.0,
        "best_duration": 0.1,
        "best_depth": 0.0,
        "best_depth_err": 0.0,
        "bls_power": np.zeros_like(periods),
        "bls_periods": periods,
        "snr": 0.0,
        "log_likelihood": np.nan,
        "stats": {},
    }


# ---------------------------------------------------------------------------
# Phase folding
# ---------------------------------------------------------------------------


def phase_fold(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    t0: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phase-fold a light curve.

    Computes the phase ``φ = ((t - t0) mod P) / P`` in the range [−0.5, 0.5).

    Parameters
    ----------
    time : np.ndarray
        Observation times (days).
    flux : np.ndarray
        Flux values.
    flux_err : np.ndarray
        Flux uncertainties.
    period : float
        Orbital period (days).
    t0 : float
        Mid-transit epoch (days).

    Returns
    -------
    tuple of np.ndarray
        ``(phase, flux_sorted, flux_err_sorted)`` sorted by phase.
    """
    if period <= 0:
        raise ValueError(f"Period must be positive, got {period}.")
    phase = ((time - t0) % period) / period
    # Map to [-0.5, 0.5)
    phase = np.where(phase >= 0.5, phase - 1.0, phase)
    sort_idx = np.argsort(phase)
    return phase[sort_idx], flux[sort_idx], flux_err[sort_idx]


# ---------------------------------------------------------------------------
# Binning utilities
# ---------------------------------------------------------------------------


def _median_bin(
    phase: np.ndarray,
    flux: np.ndarray,
    n_bins: int,
    phase_lo: float = -0.5,
    phase_hi: float = 0.5,
) -> np.ndarray:
    """Median-bin a phase-folded flux into *n_bins* equal-width bins.

    Parameters
    ----------
    phase : np.ndarray
        Phase values in [phase_lo, phase_hi).
    flux : np.ndarray
        Corresponding flux values.
    n_bins : int
        Number of bins.
    phase_lo, phase_hi : float
        Phase range.

    Returns
    -------
    np.ndarray
        Array of length *n_bins* containing median flux per bin.  Empty
        bins are filled with the global median.
    """
    bin_edges = np.linspace(phase_lo, phase_hi, n_bins + 1)
    binned = np.full(n_bins, np.nan)
    global_median = float(np.nanmedian(flux))

    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        if mask.sum() > 0:
            binned[i] = float(np.median(flux[mask]))

    # Fill NaN bins
    nan_mask = ~np.isfinite(binned)
    if nan_mask.any():
        binned[nan_mask] = global_median

    return binned


def _normalise_view(view: np.ndarray) -> np.ndarray:
    """Normalise a view so out-of-transit = 1, transit minimum = 0.

    Parameters
    ----------
    view : np.ndarray
        Raw median-binned flux view.

    Returns
    -------
    np.ndarray
        Normalised view in [0, 1].
    """
    baseline = float(np.nanmedian(view))
    minimum = float(np.nanmin(view))
    rng = baseline - minimum
    if rng < 1e-12:
        return np.ones_like(view)
    return (view - minimum) / rng


# ---------------------------------------------------------------------------
# Global view
# ---------------------------------------------------------------------------


def make_global_view(
    phase: np.ndarray,
    flux: np.ndarray,
    n_bins: int = DEFAULT_GLOBAL_BINS,
) -> np.ndarray:
    """Produce a global-view light curve representation.

    Parameters
    ----------
    phase : np.ndarray
        Phase values in [−0.5, 0.5).
    flux : np.ndarray
        Flux values (same ordering as *phase*).
    n_bins : int
        Number of phase bins.

    Returns
    -------
    np.ndarray
        Normalised median-binned phase-folded flux of length *n_bins*.
    """
    binned = _median_bin(phase, flux, n_bins, -0.5, 0.5)
    return _normalise_view(binned)


# ---------------------------------------------------------------------------
# Local view
# ---------------------------------------------------------------------------


def make_local_view(
    phase: np.ndarray,
    flux: np.ndarray,
    transit_duration: float,
    period: float,
    n_bins: int = DEFAULT_LOCAL_BINS,
    half_width_factor: float = DEFAULT_LOCAL_HALF_WIDTH,
) -> np.ndarray:
    """Produce a local-view transit light curve representation.

    Centres on phase=0 and uses a window of ±*half_width_factor* ×
    transit duration (in phase units).

    Parameters
    ----------
    phase : np.ndarray
        Phase values in [−0.5, 0.5).
    flux : np.ndarray
        Flux values.
    transit_duration : float
        Transit duration in days.
    period : float
        Orbital period in days.
    n_bins : int
        Number of bins in the local view.
    half_width_factor : float
        Half-window width as a multiple of the transit duration.

    Returns
    -------
    np.ndarray
        Normalised median-binned local view of length *n_bins*.
    """
    if period <= 0:
        period = 1.0
    half_width_phase = half_width_factor * (transit_duration / period)
    half_width_phase = min(half_width_phase, 0.49)

    binned = _median_bin(
        phase, flux, n_bins,
        -half_width_phase, half_width_phase,
    )
    return _normalise_view(binned)


# ---------------------------------------------------------------------------
# River plot
# ---------------------------------------------------------------------------


def make_river_plot(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    n_bins: int = DEFAULT_GLOBAL_BINS,
) -> np.ndarray:
    """Produce a 2-D river plot.

    Each row represents one orbital cycle, with the flux median-binned into
    *n_bins* phase bins.

    Parameters
    ----------
    time : np.ndarray
        Observation times (days).
    flux : np.ndarray
        Flux values.
    period : float
        Orbital period (days).
    t0 : float
        Mid-transit epoch (days).
    n_bins : int
        Number of phase bins per cycle.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(n_cycles, n_bins)``.  Rows are sorted by cycle
        index (ascending time).  Empty rows are filled with 1.0.
    """
    if period <= 0:
        raise ValueError("period must be positive.")

    # Compute cycle index for each cadence
    cycle_idx = np.floor((time - t0) / period).astype(int)
    phase = ((time - t0) % period) / period
    phase = np.where(phase >= 0.5, phase - 1.0, phase)

    unique_cycles = np.unique(cycle_idx)
    n_cycles = len(unique_cycles)
    if n_cycles == 0:
        return np.ones((1, n_bins))

    river = np.ones((n_cycles, n_bins))
    bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)

    for row, cyc in enumerate(unique_cycles):
        mask = cycle_idx == cyc
        ph_cyc = phase[mask]
        fl_cyc = flux[mask]
        if len(ph_cyc) == 0:
            continue
        for b in range(n_bins):
            bin_mask = (ph_cyc >= bin_edges[b]) & (ph_cyc < bin_edges[b + 1])
            if bin_mask.sum() > 0:
                river[row, b] = float(np.median(fl_cyc[bin_mask]))

    return river


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------


def fold_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    config: Optional[dict] = None,
    period_min: float = DEFAULT_PERIOD_MIN,
    period_max: float = DEFAULT_PERIOD_MAX,
    n_periods: int = DEFAULT_N_PERIODS,
    global_bins: int = DEFAULT_GLOBAL_BINS,
    local_bins: int = DEFAULT_LOCAL_BINS,
) -> PhaseResult:
    """Run BLS and produce all phase-folded light curve views.

    Parameters
    ----------
    time : np.ndarray
        Observation times (days).
    flux : np.ndarray
        Normalised flux.
    flux_err : np.ndarray
        Flux uncertainties.
    config : dict, optional
        Pipeline configuration.
    period_min : float
        Minimum BLS search period (days).
    period_max : float
        Maximum BLS search period (days).
    n_periods : int
        Number of trial periods.
    global_bins : int
        Number of bins in the global view.
    local_bins : int
        Number of bins in the local view.

    Returns
    -------
    PhaseResult
        All outputs including BLS parameters and folded views.
    """
    if config is not None:
        global_bins = int(get(config, "conditioning.phase_fold.global_bins", global_bins))
        local_bins = int(get(config, "conditioning.phase_fold.local_bins", local_bins))

    # 1. Run BLS
    bls_result = run_bls(
        time, flux, flux_err,
        period_min=period_min,
        period_max=period_max,
        n_periods=n_periods,
        config=config,
    )

    period = bls_result["best_period"]
    t0 = bls_result["best_t0"]
    duration = bls_result["best_duration"]
    depth = bls_result["best_depth"]

    # 2. Phase fold
    ok = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    phase, flux_s, _ = phase_fold(time[ok], flux[ok], flux_err[ok], period, t0)

    # 3. Build views
    global_view = make_global_view(phase, flux_s, global_bins)
    local_view = make_local_view(phase, flux_s, duration, period, local_bins)
    river = make_river_plot(time[ok], flux[ok], period, t0, global_bins)

    # 4. Estimate number of transits observed
    time_span = float(np.ptp(time[ok]))
    n_transits = max(1, int(np.floor(time_span / period)))

    return PhaseResult(
        best_period=period,
        best_t0=t0,
        best_duration=duration,
        best_depth=depth,
        bls_power=bls_result["bls_power"],
        bls_periods=bls_result["bls_periods"],
        bls_stats=bls_result,
        phase=phase,
        global_view=global_view,
        local_view=local_view,
        river_plot=river,
        n_transits_observed=n_transits,
    )


run = fold_lightcurve


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_lightcurve(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load time, flux, flux_err from a FITS or .npz file."""
    if path.endswith(".npz"):
        data = np.load(path, allow_pickle=False)
        return data["time"], data["flux"], data["flux_err"]
    from astropy.io import fits as af
    with af.open(path) as hdul:
        for hdu in hdul:
            if hasattr(hdu, "columns") and hdu.data is not None:
                cols = [c.name for c in hdu.columns]
                t = np.asarray(hdu.data["TIME"], dtype=float)
                f_col = "PDCSAP_FLUX" if "PDCSAP_FLUX" in cols else "FLUX"
                e_col = "PDCSAP_FLUX_ERR" if "PDCSAP_FLUX_ERR" in cols else "FLUX_ERR"
                f = np.asarray(hdu.data[f_col], dtype=float)
                e = np.asarray(hdu.data[e_col], dtype=float)
                ok = np.isfinite(t) & np.isfinite(f)
                return t[ok], f[ok], np.where(np.isfinite(e[ok]), e[ok], np.nanmedian(e))
    raise ValueError(f"Cannot read light curve from {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BLS periodogram search and phase-folding for TESS light curves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_file", type=str, help="FITS or .npz light curve.")
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output .npz file. Defaults to <input>_phasefolded.npz.",
    )
    parser.add_argument("--period-min", type=float, default=DEFAULT_PERIOD_MIN)
    parser.add_argument("--period-max", type=float, default=DEFAULT_PERIOD_MAX)
    parser.add_argument("--n-periods", type=int, default=DEFAULT_N_PERIODS)
    parser.add_argument("--global-bins", type=int, default=DEFAULT_GLOBAL_BINS)
    parser.add_argument("--local-bins", type=int, default=DEFAULT_LOCAL_BINS)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: Optional[list] = None) -> None:
    """Entry point for the BLS/phase-fold CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config: Optional[dict] = None
    if args.config:
        config = load_config(args.config)

    output = args.output or (Path(args.input_file).stem + "_phasefolded.npz")

    logger.info("Loading light curve from %s", args.input_file)
    time, flux, flux_err = _load_lightcurve(args.input_file)

    # Normalise
    med = float(np.nanmedian(flux))
    if med != 0.0:
        flux = flux / med
        flux_err = flux_err / med

    result = fold_lightcurve(
        time, flux, flux_err,
        config=config,
        period_min=args.period_min,
        period_max=args.period_max,
        n_periods=args.n_periods,
        global_bins=args.global_bins,
        local_bins=args.local_bins,
    )

    np.savez(
        output,
        best_period=result.best_period,
        best_t0=result.best_t0,
        best_duration=result.best_duration,
        best_depth=result.best_depth,
        bls_power=result.bls_power,
        bls_periods=result.bls_periods,
        phase=result.phase,
        global_view=result.global_view,
        local_view=result.local_view,
        river_plot=result.river_plot,
        n_transits_observed=result.n_transits_observed,
    )
    logger.info("Saved phase-fold result to %s", output)
    print(f"Phase-fold result saved to: {output}")
    print(f"  Best period   : {result.best_period:.6f} days")
    print(f"  Best epoch t0 : {result.best_t0:.6f} days")
    print(f"  Duration      : {result.best_duration * 24:.3f} hours")
    print(f"  Depth         : {result.best_depth * 1e6:.1f} ppm")
    print(f"  SNR           : {result.bls_stats.get('snr', 0):.2f}")
    print(f"  N transits    : {result.n_transits_observed}")


if __name__ == "__main__":
    main()
