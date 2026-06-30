"""
wavelet_detrend.py
==================
Wavelet-based multi-resolution detrending for exoplanet light curves.

Uses PyWavelets (pywt) to decompose a flux time series into multi-resolution
approximation and detail coefficients. The lowest-frequency approximation
coefficient encodes the long-term stellar trend, which is removed from the
observed flux. Medium-frequency detail coefficients can optionally be
thresholded with sigma-clipping to suppress stellar variability residuals.

Author: Exoplanet Detection Pipeline
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pywt
from astropy.io import fits
from scipy.interpolate import interp1d

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
# Constants / defaults
# ---------------------------------------------------------------------------
DEFAULT_WAVELET = "db8"
DEFAULT_LEVELS = 4
DEFAULT_SIGMA_THRESHOLD = 3.0


# ---------------------------------------------------------------------------
# Dataclass for results
# ---------------------------------------------------------------------------
@dataclass
class WaveletDetrendResult:
    """Container for wavelet detrending outputs.

    Attributes
    ----------
    time : np.ndarray
        Original time array (days).
    detrended_flux : np.ndarray
        Flux with the long-term trend removed (normalised around 0).
    trend_model : np.ndarray
        Reconstructed long-term trend (approximation component).
    flux_err : np.ndarray
        Preserved (unchanged) flux uncertainties.
    wavelet : str
        Wavelet family used.
    levels : int
        Number of decomposition levels.
    coefficients : list
        List of wavelet coefficient arrays [cA_N, cD_N, ..., cD_1].
    """

    time: np.ndarray
    detrended_flux: np.ndarray
    trend_model: np.ndarray
    flux_err: np.ndarray
    wavelet: str = DEFAULT_WAVELET
    levels: int = DEFAULT_LEVELS
    coefficients: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------
def detrend_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    config: Optional[dict] = None,
    wavelet: str = DEFAULT_WAVELET,
    levels: int = DEFAULT_LEVELS,
    sigma_threshold: Optional[float] = DEFAULT_SIGMA_THRESHOLD,
    threshold_levels: Optional[list] = None,
) -> WaveletDetrendResult:
    """Perform wavelet-based multi-resolution detrending on a light curve.

    Decomposes the flux into approximation and detail wavelet coefficients
    up to *levels* levels using the specified wavelet family.  The
    approximation (lowest-frequency) coefficient is set to zero so that the
    reconstructed signal has the long-term trend removed.  Optionally,
    medium-frequency detail coefficients (specified by *threshold_levels*)
    are soft-thresholded to further suppress stellar variability.

    Parameters
    ----------
    time : np.ndarray
        Array of observation times in days.  Must be 1-D and monotonically
        increasing.
    flux : np.ndarray
        Normalised (or raw) flux values with the same shape as *time*.
    flux_err : np.ndarray
        Flux uncertainties with the same shape as *flux*.  These are not
        modified by the detrending process.
    config : dict, optional
        Configuration dictionary loaded by ``utils.config.load_config``.
        Keys consulted: ``conditioning.wavelet``, ``conditioning.levels``,
        ``conditioning.sigma_threshold``.
    wavelet : str
        Wavelet family name accepted by ``pywt`` (e.g. ``'db8'``,
        ``'sym8'``).  Overridden by *config* if provided.
    levels : int
        Number of decomposition levels.  Overridden by *config* if provided.
    sigma_threshold : float or None
        If not ``None``, apply soft-thresholding to the detail coefficients
        listed in *threshold_levels* using this sigma multiplier.  Set to
        ``None`` to skip thresholding.
    threshold_levels : list[int] or None
        Which detail coefficient levels (1-indexed from finest) to threshold.
        Defaults to the two finest levels ``[1, 2]``.

    Returns
    -------
    WaveletDetrendResult
        Dataclass containing ``detrended_flux``, ``trend_model``,
        ``flux_err``, and intermediate wavelet coefficients.

    Raises
    ------
    ValueError
        If *time*, *flux*, and *flux_err* do not have the same length, or if
        *levels* is non-positive.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 30, 1000)
    >>> f = 1.0 + 0.01*np.sin(2*np.pi*t/15) + np.random.normal(0, 1e-3, 1000)
    >>> fe = np.full_like(f, 1e-3)
    >>> result = detrend_lightcurve(t, f, fe)
    >>> result.detrended_flux.shape
    (1000,)
    """
    # ------------------------------------------------------------------
    # 1. Resolve configuration
    # ------------------------------------------------------------------
    if config is not None:
        wavelet = get(config, "conditioning.wavelet", wavelet)
        levels = int(get(config, "conditioning.levels", levels))
        sigma_threshold = get(config, "conditioning.sigma_threshold", sigma_threshold)

    if threshold_levels is None:
        threshold_levels = [1, 2]

    # ------------------------------------------------------------------
    # 2. Input validation
    # ------------------------------------------------------------------
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)

    if time.shape != flux.shape or flux.shape != flux_err.shape:
        raise ValueError(
            f"time, flux, flux_err must have the same shape. "
            f"Got {time.shape}, {flux.shape}, {flux_err.shape}."
        )
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}.")

    n = len(flux)

    # ------------------------------------------------------------------
    # 3. Handle NaNs / gaps by linear interpolation before decomposition
    # ------------------------------------------------------------------
    finite_mask = np.isfinite(flux) & np.isfinite(time)
    if not np.all(finite_mask):
        logger.warning(
            "%d non-finite flux values detected. Interpolating for wavelet "
            "decomposition only; originals preserved in flux_err.",
            int((~finite_mask).sum()),
        )
        if finite_mask.sum() < 2:
            raise ValueError("Too few finite flux values for wavelet detrending.")
        interp_fn = interp1d(
            time[finite_mask],
            flux[finite_mask],
            kind="linear",
            fill_value="extrapolate",
        )
        flux_work = interp_fn(time)
    else:
        flux_work = flux.copy()

    # ------------------------------------------------------------------
    # 4. Cap decomposition levels to what pywt allows
    # ------------------------------------------------------------------
    max_level = pywt.dwt_max_level(n, wavelet)
    if levels > max_level:
        logger.warning(
            "Requested levels=%d exceeds maximum for wavelet '%s' with n=%d. "
            "Clamping to %d.",
            levels,
            wavelet,
            n,
            max_level,
        )
        levels = max_level

    # ------------------------------------------------------------------
    # 5. Wavelet decomposition
    # ------------------------------------------------------------------
    logger.debug(
        "Wavelet decomposition: wavelet=%s, levels=%d, n=%d", wavelet, levels, n
    )
    coeffs = pywt.wavedec(flux_work, wavelet=wavelet, level=levels, mode="periodization")
    # coeffs[0] = cA_N (approximation at coarsest scale)
    # coeffs[1] = cD_N, ..., coeffs[levels] = cD_1

    # ------------------------------------------------------------------
    # 6. Extract and zero-out the trend (approximation) component
    # ------------------------------------------------------------------
    coeffs_trend = [c.copy() for c in coeffs]
    coeffs_zero = [c.copy() for c in coeffs]

    # Zero the detail coefficients to isolate the trend
    for i in range(1, len(coeffs_trend)):
        coeffs_trend[i] = np.zeros_like(coeffs_trend[i])

    # Zero the approximation to detrend
    coeffs_zero[0] = np.zeros_like(coeffs_zero[0])

    # ------------------------------------------------------------------
    # 7. Optional soft-thresholding of medium-frequency detail coefficients
    # ------------------------------------------------------------------
    if sigma_threshold is not None:
        for lv in threshold_levels:
            # detail level lv (1 = finest, N = coarsest)
            idx = levels - lv + 1  # index in coeffs list
            if 1 <= idx < len(coeffs_zero):
                d = coeffs_zero[idx]
                sigma_mad = _mad_sigma(d)
                threshold = sigma_threshold * sigma_mad
                coeffs_zero[idx] = pywt.threshold(d, threshold, mode="soft")
                logger.debug(
                    "Thresholded detail level %d (idx=%d): sigma=%.4f, thr=%.4f",
                    lv,
                    idx,
                    sigma_mad,
                    threshold,
                )

    # ------------------------------------------------------------------
    # 8. Reconstruct trend and detrended flux
    # ------------------------------------------------------------------
    trend_reconstructed = pywt.waverec(coeffs_trend, wavelet=wavelet, mode="periodization")
    detrended_reconstructed = pywt.waverec(coeffs_zero, wavelet=wavelet, mode="periodization")

    # Trim or pad to original length (periodization mode may add one sample)
    trend_reconstructed = _match_length(trend_reconstructed, n)
    detrended_reconstructed = _match_length(detrended_reconstructed, n)

    # Restore NaN positions in original data
    detrended_flux = np.where(finite_mask, detrended_reconstructed, np.nan)
    trend_model = np.where(finite_mask, trend_reconstructed, np.nan)

    logger.info(
        "Wavelet detrending complete. Trend RMS=%.6f, Residual RMS=%.6f",
        float(np.nanstd(trend_model)),
        float(np.nanstd(detrended_flux)),
    )

    return WaveletDetrendResult(
        time=time,
        detrended_flux=detrended_flux,
        trend_model=trend_model,
        flux_err=flux_err,
        wavelet=wavelet,
        levels=levels,
        coefficients=coeffs,
    )


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _mad_sigma(arr: np.ndarray) -> float:
    """Estimate the standard deviation from the median absolute deviation."""
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return float(1.4826 * mad + 1e-12)


def _match_length(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Trim or zero-pad *arr* to *target_len*."""
    if len(arr) >= target_len:
        return arr[:target_len]
    pad = np.zeros(target_len - len(arr), dtype=arr.dtype)
    return np.concatenate([arr, pad])


# ---------------------------------------------------------------------------
# Class interface
# ---------------------------------------------------------------------------
class WaveletDetrend:
    """Scikit-learn-style fit/transform interface for wavelet detrending.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary (see :func:`detrend_lightcurve`).
    wavelet : str
        Wavelet family name.
    levels : int
        Number of decomposition levels.
    sigma_threshold : float or None
        Sigma multiplier for optional detail coefficient thresholding.
    threshold_levels : list[int] or None
        Detail levels to threshold (1 = finest).

    Examples
    --------
    >>> wdt = WaveletDetrend(wavelet='db8', levels=4)
    >>> result = wdt.fit_transform(time, flux, flux_err)
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        wavelet: str = DEFAULT_WAVELET,
        levels: int = DEFAULT_LEVELS,
        sigma_threshold: Optional[float] = DEFAULT_SIGMA_THRESHOLD,
        threshold_levels: Optional[list] = None,
    ) -> None:
        self.config = config
        self.wavelet = wavelet
        self.levels = levels
        self.sigma_threshold = sigma_threshold
        self.threshold_levels = threshold_levels
        self._result: Optional[WaveletDetrendResult] = None

    # ------------------------------------------------------------------
    def fit(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
    ) -> "WaveletDetrend":
        """Fit the wavelet detrending model to *flux*.

        Parameters
        ----------
        time : np.ndarray
            Observation times.
        flux : np.ndarray
            Flux values.
        flux_err : np.ndarray
            Flux uncertainties.

        Returns
        -------
        WaveletDetrend
            Self (for method chaining).
        """
        self._result = detrend_lightcurve(
            time=time,
            flux=flux,
            flux_err=flux_err,
            config=self.config,
            wavelet=self.wavelet,
            levels=self.levels,
            sigma_threshold=self.sigma_threshold,
            threshold_levels=self.threshold_levels,
        )
        return self

    # ------------------------------------------------------------------
    def transform(self) -> WaveletDetrendResult:
        """Return the detrending result from the last :meth:`fit` call.

        Returns
        -------
        WaveletDetrendResult
            The computed detrending result.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if self._result is None:
            raise RuntimeError("Call fit() before transform().")
        return self._result

    # ------------------------------------------------------------------
    def fit_transform(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
    ) -> WaveletDetrendResult:
        """Convenience: fit then transform in one call.

        Parameters
        ----------
        time : np.ndarray
            Observation times.
        flux : np.ndarray
            Flux values.
        flux_err : np.ndarray
            Flux uncertainties.

        Returns
        -------
        WaveletDetrendResult
            Detrending result.
        """
        return self.fit(time, flux, flux_err).transform()

    # ------------------------------------------------------------------
    def get_coefficients(self) -> list:
        """Return the raw wavelet coefficient list from the last fit.

        Returns
        -------
        list
            List of numpy arrays [cA_N, cD_N, ..., cD_1].
        """
        if self._result is None:
            raise RuntimeError("Call fit() first.")
        return self._result.coefficients

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"WaveletDetrend(wavelet='{self.wavelet}', levels={self.levels}, "
            f"sigma_threshold={self.sigma_threshold})"
        )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def _load_fits_lightcurve(
    fits_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load time, flux, flux_err from a TESS-style FITS light curve file.

    Parameters
    ----------
    fits_path : str
        Path to the FITS file containing a BINTABLE extension with columns
        TIME, PDCSAP_FLUX (or FLUX), and PDCSAP_FLUX_ERR (or FLUX_ERR).

    Returns
    -------
    tuple of np.ndarray
        (time, flux, flux_err) arrays with NaN entries removed.
    """
    with fits.open(fits_path) as hdul:
        for hdu in hdul:
            if hasattr(hdu, "columns") and hdu.columns is not None and hdu.data is not None:
                data = hdu.data
                colnames = [c.name for c in hdu.columns]
                time_col = "TIME" if "TIME" in colnames else colnames[0]
                flux_col = (
                    "PDCSAP_FLUX"
                    if "PDCSAP_FLUX" in colnames
                    else ("FLUX" if "FLUX" in colnames else colnames[1])
                )
                err_col = (
                    "PDCSAP_FLUX_ERR"
                    if "PDCSAP_FLUX_ERR" in colnames
                    else (
                        "FLUX_ERR"
                        if "FLUX_ERR" in colnames
                        else (colnames[2] if len(colnames) > 2 else flux_col)
                    )
                )
                time = np.asarray(data[time_col], dtype=float)
                flux = np.asarray(data[flux_col], dtype=float)
                flux_err = np.asarray(data[err_col], dtype=float)
                mask = np.isfinite(time) & np.isfinite(flux)
                if not np.any(mask):
                    raise ValueError("No valid (finite) data found in FITS file.")
                flux_err = np.where(np.isfinite(flux_err), flux_err, np.nanmedian(flux_err))
                return time[mask], flux[mask], flux_err[mask]
    raise ValueError(f"No usable BINTABLE extension found in {fits_path}")


def _save_result(result: WaveletDetrendResult, output_path: str) -> None:
    """Save the detrending result to a FITS file.

    Parameters
    ----------
    result : WaveletDetrendResult
        Detrending result to save.
    output_path : str
        Destination FITS file path.
    """
    from astropy.table import Table
    from astropy.io.fits import BinTableHDU, PrimaryHDU, HDUList

    tbl = Table(
        {
            "TIME": result.time,
            "DETRENDED_FLUX": result.detrended_flux,
            "TREND_MODEL": result.trend_model,
            "FLUX_ERR": result.flux_err,
        }
    )
    primary = PrimaryHDU()
    primary.header["WAVELET"] = result.wavelet
    primary.header["LEVELS"] = result.levels
    tbl_hdu = BinTableHDU(tbl, name="LIGHTCURVE")
    HDUList([primary, tbl_hdu]).writeto(output_path, overwrite=True)
    logger.info("Saved detrended light curve to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wavelet-based multi-resolution detrending for TESS light curves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("fits_file", type=str, help="Path to input FITS light curve.")
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output FITS file path. Defaults to <input>_detrended.fits.",
    )
    parser.add_argument(
        "--wavelet", type=str, default=DEFAULT_WAVELET,
        help="PyWavelets wavelet family.",
    )
    parser.add_argument(
        "--levels", type=int, default=DEFAULT_LEVELS,
        help="Number of decomposition levels.",
    )
    parser.add_argument(
        "--sigma-threshold", type=float, default=DEFAULT_SIGMA_THRESHOLD,
        help="Sigma for detail coefficient thresholding. Set 0 to disable.",
    )
    parser.add_argument(
        "--threshold-levels", nargs="+", type=int, default=[1, 2],
        help="Detail levels to threshold (1=finest).",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to pipeline config YAML.",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def main(argv: Optional[list] = None) -> None:
    """Entry point for the wavelet detrending CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config: Optional[dict] = None
    if args.config:
        config = load_config(args.config)

    output = args.output
    if output is None:
        p = Path(args.fits_file)
        output = str(p.parent / (p.stem + "_detrended.fits"))

    logger.info("Loading light curve from %s", args.fits_file)
    time, flux, flux_err = _load_fits_lightcurve(args.fits_file)
    logger.info("Loaded %d cadences.", len(time))

    sigma_threshold: Optional[float] = (
        args.sigma_threshold if args.sigma_threshold > 0 else None
    )

    result = detrend_lightcurve(
        time=time,
        flux=flux,
        flux_err=flux_err,
        config=config,
        wavelet=args.wavelet,
        levels=args.levels,
        sigma_threshold=sigma_threshold,
        threshold_levels=args.threshold_levels,
    )

    _save_result(result, output)
    print(f"Detrended light curve saved to: {output}")
    print(f"  Trend RMS      : {float(np.nanstd(result.trend_model)):.6f}")
    print(f"  Residual RMS   : {float(np.nanstd(result.detrended_flux)):.6f}")
    print(f"  Wavelet        : {result.wavelet}")
    print(f"  Levels         : {result.levels}")


if __name__ == "__main__":
    main()
