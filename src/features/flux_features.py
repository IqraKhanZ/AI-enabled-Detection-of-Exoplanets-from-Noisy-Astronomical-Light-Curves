"""
flux_features.py
================
Flux-based statistical feature extraction for exoplanet transit characterisation.

Extracts a comprehensive set of photometric and transit-geometry features from
a phase-folded light curve and the associated BLS statistics.  These features
are used as input to the classical ML classifiers and can also augment the
representation vector of a neural network.

Features computed
-----------------
transit_depth_ppm
    Transit depth in parts-per-million (from BLS best-fit depth).
transit_duration_hrs
    Transit duration converted to hours.
orbital_period_days
    Best-fit orbital period from BLS.
transit_snr
    Signal-to-noise ratio from BLS power.
odd_even_depth_difference
    Ratio of odd-transit depth to even-transit depth minus 1.  A value
    near −1 or >>1 is diagnostic of an eclipsing binary.
secondary_eclipse_depth_ppm
    Depth of the secondary eclipse (at phase ≈ 0.5) in ppm.
oot_scatter
    Standard deviation of the out-of-transit flux (baseline noise level).
flux_skewness
    Skewness of the (normalised) flux distribution.
flux_kurtosis
    Excess kurtosis of the flux distribution.
autocorrelation_lag1
    Pearson autocorrelation at lag-1.
n_transits_observed
    Number of complete orbital cycles observed.
transit_depth_consistency
    Standard deviation of individual per-transit depths (in ppm).
limb_darkening_asymmetry
    Ratio of ingress to egress duration (from trapezoid-like estimation).

Functions
---------
extract_flux_features(time, flux, flux_err, phase_result, bls_stats) -> dict
    Master extraction function.

Author: Exoplanet Detection Pipeline
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
from scipy.stats import skew, kurtosis

# Local imports
try:
    from utils.logger import get_logger
except ImportError:
    import logging as _logging

    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return _logging.getLogger(name)


# Import phase_fold result type — allow soft import
try:
    from conditioning.phase_fold import PhaseResult
except ImportError:
    try:
        from phase_fold import PhaseResult  # type: ignore[no-redef]
    except ImportError:
        PhaseResult = None  # type: ignore[misc,assignment]

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _oot_mask(phase: np.ndarray, transit_duration: float, period: float) -> np.ndarray:
    """Boolean mask selecting out-of-transit cadences.

    Parameters
    ----------
    phase : np.ndarray
        Phase array in [−0.5, 0.5).
    transit_duration : float
        Transit duration in days.
    period : float
        Orbital period in days.

    Returns
    -------
    np.ndarray of bool
        ``True`` for out-of-transit cadences.
    """
    half_width = min(0.5 * transit_duration / (period + 1e-12), 0.49)
    return np.abs(phase) > half_width


def _secondary_eclipse_depth(
    phase: np.ndarray,
    flux: np.ndarray,
    transit_duration: float,
    period: float,
    secondary_phase: float = 0.5,
) -> float:
    """Estimate the secondary eclipse depth at phase ≈ secondary_phase.

    Parameters
    ----------
    phase : np.ndarray
        Phase values in [−0.5, 0.5).
    flux : np.ndarray
        Normalised flux (baseline ≈ 1).
    transit_duration : float
        Transit duration in days.
    period : float
        Orbital period in days.
    secondary_phase : float
        Expected phase of the secondary eclipse (default 0.5 for circular orbit).

    Returns
    -------
    float
        Secondary eclipse depth in ppm (positive = dip below baseline).
    """
    half_width = min(0.5 * transit_duration / (period + 1e-12), 0.10)
    # Shift phase so secondary_phase maps to 0
    phase_shifted = phase - secondary_phase
    phase_shifted = np.where(phase_shifted < -0.5, phase_shifted + 1.0, phase_shifted)
    phase_shifted = np.where(phase_shifted >= 0.5, phase_shifted - 1.0, phase_shifted)

    in_secondary = np.abs(phase_shifted) < half_width
    if in_secondary.sum() < 3:
        return 0.0

    oot = _oot_mask(phase, transit_duration, period)
    baseline = float(np.nanmedian(flux[oot])) if oot.sum() > 0 else float(np.nanmedian(flux))
    if baseline == 0:
        return 0.0

    sec_flux = float(np.nanmedian(flux[in_secondary]))
    return float((baseline - sec_flux) / baseline * 1e6)


def _individual_transit_depths(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
) -> np.ndarray:
    """Compute the depth of each individual transit.

    Parameters
    ----------
    time : np.ndarray
        Observation times.
    flux : np.ndarray
        Normalised flux.
    period : float
        Orbital period (days).
    t0 : float
        Mid-transit epoch (days).
    duration : float
        Transit duration (days).

    Returns
    -------
    np.ndarray
        Array of per-transit depths (fractional, positive = dip).
        Returns empty array if fewer than 2 transits found.
    """
    if period <= 0:
        return np.array([])

    transit_centers = np.arange(t0, time[-1] + period, period)
    transit_centers = transit_centers[transit_centers >= time[0] - period]
    depths = []

    for tc in transit_centers:
        in_transit = np.abs(time - tc) < (0.5 * duration)
        oot = (np.abs(time - tc) > 0.5 * duration) & (np.abs(time - tc) < 2.0 * duration)
        if in_transit.sum() < 2 or oot.sum() < 5:
            continue
        baseline = float(np.nanmedian(flux[oot]))
        depth = float(baseline - np.nanmedian(flux[in_transit]))
        if np.isfinite(depth):
            depths.append(depth)

    return np.asarray(depths)


def _odd_even_depth_difference(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
) -> float:
    """Compute the odd/even transit depth ratio minus 1.

    For a genuine planet the depths should be equal (ratio ≈ 1, result ≈ 0).
    A significantly non-zero value suggests an eclipsing binary.

    Parameters
    ----------
    time, flux : np.ndarray
        Light curve arrays.
    period, t0, duration : float
        BLS parameters.

    Returns
    -------
    float
        ``(odd_depth / even_depth) - 1``, or 0 if insufficient data.
    """
    if period <= 0:
        return 0.0

    transit_centers = np.arange(t0, time[-1] + period, period)
    transit_centers = transit_centers[transit_centers >= time[0] - period]
    odd_depths = []
    even_depths = []

    for k, tc in enumerate(transit_centers):
        in_transit = np.abs(time - tc) < (0.5 * duration)
        oot = (np.abs(time - tc) > 0.5 * duration) & (np.abs(time - tc) < 2.0 * duration)
        if in_transit.sum() < 2 or oot.sum() < 3:
            continue
        baseline = float(np.nanmedian(flux[oot]))
        depth = float(baseline - np.nanmedian(flux[in_transit]))
        if not np.isfinite(depth):
            continue
        if (k + 1) % 2 == 1:
            odd_depths.append(depth)
        else:
            even_depths.append(depth)

    if not odd_depths or not even_depths:
        return 0.0
    odd_mean = float(np.mean(odd_depths))
    even_mean = float(np.mean(even_depths))
    if even_mean == 0:
        return 0.0
    return float((odd_mean / even_mean) - 1.0)


def _limb_darkening_asymmetry(local_view: np.ndarray) -> float:
    """Estimate ingress vs egress asymmetry from the local view.

    Finds the transit minimum bin and measures the gradient steepness on
    each side.  A symmetric transit (as expected for a planet with limb
    darkening) returns a value near 1.  An asymmetric profile (blended
    eclipsing binary) can deviate significantly.

    Parameters
    ----------
    local_view : np.ndarray
        50-bin normalised local view (transit at minimum).

    Returns
    -------
    float
        Ingress / egress duration ratio.  Values far from 1.0 indicate
        asymmetric transits.
    """
    n = len(local_view)
    if n < 5:
        return 1.0

    min_idx = int(np.argmin(local_view))
    transit_level = float(local_view[min_idx])
    baseline = float(np.nanmedian(local_view))
    depth = baseline - transit_level
    if depth < 1e-6:
        return 1.0

    half_depth_level = baseline - 0.5 * depth

    # Ingress: left side of minimum — find where flux crosses half depth
    ingress_idx = min_idx
    for i in range(min_idx, -1, -1):
        if local_view[i] >= half_depth_level:
            ingress_idx = i
            break

    # Egress: right side
    egress_idx = min_idx
    for i in range(min_idx, n):
        if local_view[i] >= half_depth_level:
            egress_idx = i
            break

    ingress_bins = float(min_idx - ingress_idx)
    egress_bins = float(egress_idx - min_idx)

    if egress_bins == 0:
        return 1.0
    return float(ingress_bins / egress_bins)


# ---------------------------------------------------------------------------
# Master feature extraction function
# ---------------------------------------------------------------------------


def extract_flux_features(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    phase_result,
    bls_stats: dict,
) -> dict:
    """Extract flux-based statistical features from a phase-folded light curve.

    Parameters
    ----------
    time : np.ndarray
        Full observation time array (days).
    flux : np.ndarray
        Normalised flux array (baseline ≈ 1).
    flux_err : np.ndarray
        Flux uncertainties.
    phase_result : PhaseResult
        Result from ``conditioning.phase_fold.fold_lightcurve``.  Must have
        attributes: ``best_period``, ``best_t0``, ``best_duration``,
        ``best_depth``, ``phase``, ``global_view``, ``local_view``,
        ``n_transits_observed``.
    bls_stats : dict
        BLS statistics dict from ``conditioning.phase_fold.run_bls``.

    Returns
    -------
    dict
        Feature dictionary with the following keys (all float or int):
        ``transit_depth_ppm``, ``transit_duration_hrs``,
        ``orbital_period_days``, ``transit_snr``,
        ``odd_even_depth_difference``, ``secondary_eclipse_depth_ppm``,
        ``oot_scatter``, ``flux_skewness``, ``flux_kurtosis``,
        ``autocorrelation_lag1``, ``n_transits_observed``,
        ``transit_depth_consistency``, ``limb_darkening_asymmetry``.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)

    # ------------------------------------------------------------------
    # Extract BLS parameters
    # ------------------------------------------------------------------
    period = float(getattr(phase_result, "best_period", bls_stats.get("best_period", 1.0)))
    t0 = float(getattr(phase_result, "best_t0", bls_stats.get("best_t0", time[0])))
    duration = float(getattr(phase_result, "best_duration", bls_stats.get("best_duration", 0.1)))
    depth = float(getattr(phase_result, "best_depth", bls_stats.get("best_depth", 0.0)))

    # Protect against pathological values
    period = max(period, 0.1)
    duration = max(duration, 1e-4)

    # ------------------------------------------------------------------
    # 1. Transit depth (ppm)
    # ------------------------------------------------------------------
    transit_depth_ppm = float(abs(depth) * 1e6)

    # ------------------------------------------------------------------
    # 2. Transit duration (hours)
    # ------------------------------------------------------------------
    transit_duration_hrs = float(duration * 24.0)

    # ------------------------------------------------------------------
    # 3. Orbital period (days)
    # ------------------------------------------------------------------
    orbital_period_days = period

    # ------------------------------------------------------------------
    # 4. Transit SNR
    # ------------------------------------------------------------------
    transit_snr = float(bls_stats.get("snr", 0.0))

    # ------------------------------------------------------------------
    # 5. Odd/even depth difference
    # ------------------------------------------------------------------
    ok = np.isfinite(time) & np.isfinite(flux)
    try:
        oe_diff = _odd_even_depth_difference(time[ok], flux[ok], period, t0, duration)
    except Exception as exc:
        logger.warning("odd_even_depth_difference failed: %s", exc)
        oe_diff = 0.0

    # ------------------------------------------------------------------
    # 6. Secondary eclipse depth (ppm)
    # ------------------------------------------------------------------
    phase = getattr(phase_result, "phase", None)
    if phase is not None and len(phase) == ok.sum():
        phase_arr = np.asarray(phase)
        flux_sorted_arr = flux[ok][np.argsort(((time[ok] - t0) % period) / period)]
        try:
            sec_depth = _secondary_eclipse_depth(
                phase_arr, flux[ok][np.argsort(phase_arr)], duration, period
            )
        except Exception as exc:
            logger.warning("secondary_eclipse_depth failed: %s", exc)
            sec_depth = 0.0
    else:
        sec_depth = 0.0

    # ------------------------------------------------------------------
    # 7. OOT scatter
    # ------------------------------------------------------------------
    if phase is not None and len(phase) == ok.sum():
        oot_m = _oot_mask(np.asarray(phase), duration, period)
        flux_oot = flux[ok][np.argsort(phase)][oot_m] if len(phase) > 0 else flux[ok]
        oot_scatter = float(np.nanstd(flux_oot)) if len(flux_oot) > 0 else float(np.nanstd(flux[ok]))
    else:
        oot_scatter = float(np.nanstd(flux[ok]))

    # ------------------------------------------------------------------
    # 8. Flux statistics
    # ------------------------------------------------------------------
    flux_finite = flux[np.isfinite(flux)]
    try:
        flux_skewness = float(skew(flux_finite)) if len(flux_finite) > 3 else 0.0
    except Exception:
        flux_skewness = 0.0

    try:
        flux_kurtosis = float(kurtosis(flux_finite)) if len(flux_finite) > 3 else 0.0
    except Exception:
        flux_kurtosis = 0.0

    # ------------------------------------------------------------------
    # 9. Autocorrelation at lag-1
    # ------------------------------------------------------------------
    try:
        if len(flux_finite) > 2:
            mean_f = float(np.mean(flux_finite))
            var_f = float(np.var(flux_finite))
            if var_f > 0:
                acf_lag1 = float(np.mean((flux_finite[:-1] - mean_f) * (flux_finite[1:] - mean_f)) / var_f)
            else:
                acf_lag1 = 0.0
        else:
            acf_lag1 = 0.0
    except Exception:
        acf_lag1 = 0.0

    # ------------------------------------------------------------------
    # 10. Number of transits observed
    # ------------------------------------------------------------------
    n_transits = int(getattr(phase_result, "n_transits_observed", max(1, int(np.ptp(time[ok]) / period))))

    # ------------------------------------------------------------------
    # 11. Transit depth consistency
    # ------------------------------------------------------------------
    try:
        ind_depths = _individual_transit_depths(time[ok], flux[ok], period, t0, duration)
        if len(ind_depths) >= 2:
            depth_consistency = float(np.std(ind_depths) * 1e6)  # in ppm
        else:
            depth_consistency = 0.0
    except Exception as exc:
        logger.warning("transit_depth_consistency failed: %s", exc)
        depth_consistency = 0.0

    # ------------------------------------------------------------------
    # 12. Limb darkening asymmetry
    # ------------------------------------------------------------------
    local_view = getattr(phase_result, "local_view", None)
    if local_view is not None and len(local_view) > 4:
        try:
            ld_asym = _limb_darkening_asymmetry(np.asarray(local_view))
        except Exception as exc:
            logger.warning("limb_darkening_asymmetry failed: %s", exc)
            ld_asym = 1.0
    else:
        ld_asym = 1.0

    # ------------------------------------------------------------------
    # Assemble feature dict
    # ------------------------------------------------------------------
    features = {
        "transit_depth_ppm": float(transit_depth_ppm),
        "transit_duration_hrs": float(transit_duration_hrs),
        "orbital_period_days": float(orbital_period_days),
        "transit_snr": float(transit_snr),
        "odd_even_depth_difference": float(oe_diff),
        "secondary_eclipse_depth_ppm": float(sec_depth),
        "oot_scatter": float(oot_scatter),
        "flux_skewness": float(flux_skewness),
        "flux_kurtosis": float(flux_kurtosis),
        "autocorrelation_lag1": float(acf_lag1),
        "n_transits_observed": int(n_transits),
        "transit_depth_consistency": float(depth_consistency),
        "limb_darkening_asymmetry": float(ld_asym),
    }

    logger.debug("Extracted flux features: %s", features)
    return features


# ---------------------------------------------------------------------------
# Convenience: validate feature dict completeness
# ---------------------------------------------------------------------------
EXPECTED_FEATURES = [
    "transit_depth_ppm",
    "transit_duration_hrs",
    "orbital_period_days",
    "transit_snr",
    "odd_even_depth_difference",
    "secondary_eclipse_depth_ppm",
    "oot_scatter",
    "flux_skewness",
    "flux_kurtosis",
    "autocorrelation_lag1",
    "n_transits_observed",
    "transit_depth_consistency",
    "limb_darkening_asymmetry",
]def extract(time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray) -> np.ndarray:
    """Extract flux-based statistical features as a 1D numpy array."""
    # Build a simple PhaseResult internally
    from astropy.timeseries import BoxLeastSquares
    try:
        from conditioning.phase_fold import run_bls, fold_lightcurve
        phase_result = fold_lightcurve(time, flux, flux_err)
        bls_stats = phase_result.bls_stats
    except Exception:
        # Fallback if phase_fold is not fully available/fails
        try:
            bls = BoxLeastSquares(time, flux, flux_err)
            period_grid = np.linspace(0.5, 15.0, 1000)
            pg = bls.power(period_grid, 0.1)
            best_idx = np.argmax(pg.power)
            best_period = period_grid[best_idx]
            best_t0 = pg.transit_time[best_idx]
            best_duration = pg.duration[best_idx]
            best_depth = pg.depth[best_idx]
            snr = pg.depth[best_idx] / max(np.std(flux), 1e-6)
        except Exception:
            best_period = 3.0
            best_t0 = time[0]
            best_duration = 0.1
            best_depth = 0.001
            snr = 5.0
            
        class DummyPhaseResult:
            pass
        phase_result = DummyPhaseResult()
        phase_result.best_period = best_period
        phase_result.best_t0 = best_t0
        phase_result.best_duration = best_duration
        phase_result.best_depth = best_depth
        phase_result.phase = ((time - best_t0) % best_period) / best_period
        phase_result.phase = np.where(phase_result.phase >= 0.5, phase_result.phase - 1.0, phase_result.phase)
        phase_result.global_view = np.ones(200)
        phase_result.local_view = np.ones(50)
        phase_result.n_transits_observed = max(1, int(np.ptp(time) / best_period))
        bls_stats = {"snr": snr}

    feats = extract_flux_features(time, flux, flux_err, phase_result, bls_stats)
    return np.array([feats[k] for k in EXPECTED_FEATURES], dtype=np.float32)


def validate_feature_dict(features: dict) -> bool:
    """Check that all expected features are present.

    Parameters
    ----------
    features : dict
        Feature dictionary from :func:`extract_flux_features`.

    Returns
    -------
    bool
        ``True`` if all keys are present.
    """
    missing = [k for k in EXPECTED_FEATURES if k not in features]
    if missing:
        logger.warning("Missing features: %s", missing)
        return False
    return True


if __name__ == "__main__":
    # Quick smoke test
    import sys

    logging.basicConfig(level=logging.DEBUG)

    # Create dummy PhaseResult-like object
    class _DummyPhaseResult:
        best_period = 3.14
        best_t0 = 0.0
        best_duration = 0.1
        best_depth = 1e-3
        phase = np.linspace(-0.5, 0.5, 900)
        local_view = np.ones(50)
        local_view[20:30] = 0.998
        n_transits_observed = 8

    t = np.linspace(0, 27, 1000)
    f = np.ones(1000) + np.random.normal(0, 1e-3, 1000)
    fe = np.full(1000, 1e-3)
    pr = _DummyPhaseResult()
    bls = {
        "best_period": 3.14, "best_t0": 0.0, "best_duration": 0.1,
        "best_depth": 1e-3, "snr": 12.5,
    }
    feats = extract_flux_features(t, f, fe, pr, bls)
    print("Extracted features:")
    for k, v in feats.items():
        print(f"  {k}: {v}")
    ok = validate_feature_dict(feats)
    print(f"Validation: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
