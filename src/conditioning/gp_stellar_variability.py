"""
gp_stellar_variability.py
==========================
Gaussian Process (GP) stellar variability modelling using celerite2.

Models correlated stellar variability signals (granulation, rotation,
pulsation) with a celerite2 SHO or Matern-3/2 kernel. The GP is fitted to
out-of-transit cadences via log-likelihood maximisation using
``scipy.optimize.minimize``. If optimisation fails, the module gracefully
falls back to a low-order polynomial detrending.

Classes
-------
GPVariabilityModel
    Main class with ``fit``, ``predict``, and ``get_residuals`` methods.

Functions
---------
_polynomial_fallback
    Polynomial detrending used when GP optimisation fails.
_log_likelihood
    Negative log-likelihood function for celerite2.

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
from scipy.optimize import minimize

try:
    import celerite2
    from celerite2 import terms as celerite2_terms

    _CELERITE2_AVAILABLE = True
except ImportError:
    _CELERITE2_AVAILABLE = False
    warnings.warn(
        "celerite2 is not installed. GP modelling will fall back to polynomial detrending.",
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
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class GPResult:
    """Outputs from a GP stellar variability model fit.

    Attributes
    ----------
    time : np.ndarray
        Full time array (days).
    gp_mean_prediction : np.ndarray
        GP predictive mean evaluated at *time*.
    residuals : np.ndarray
        ``flux - gp_mean_prediction``.
    gp_variance : np.ndarray
        GP predictive variance at *time* (same units as flux^2).
    kernel_params : dict
        Optimised kernel hyperparameters.
    converged : bool
        Whether the optimisation converged.
    fallback_used : bool
        ``True`` if polynomial fallback was used instead of GP.
    log_likelihood : float
        Final log-likelihood value (NaN if fallback used).
    """

    time: np.ndarray
    gp_mean_prediction: np.ndarray
    residuals: np.ndarray
    gp_variance: np.ndarray
    kernel_params: dict = field(default_factory=dict)
    converged: bool = False
    fallback_used: bool = False
    log_likelihood: float = float("nan")


# ---------------------------------------------------------------------------
# Helper: polynomial fallback
# ---------------------------------------------------------------------------


def _polynomial_fallback(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    degree: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a polynomial trend and return (prediction, residuals, variance).

    Parameters
    ----------
    time : np.ndarray
        Time values.
    flux : np.ndarray
        Flux values.
    flux_err : np.ndarray
        Flux uncertainties.
    degree : int
        Polynomial degree.

    Returns
    -------
    tuple of np.ndarray
        (gp_mean_prediction, residuals, gp_variance) where gp_variance is
        estimated as the median flux_err^2 broadcast to full length.
    """
    t_norm = (time - time.mean()) / (time.ptp() + 1e-12)
    weights = 1.0 / (flux_err ** 2 + 1e-12)
    coeffs = np.polyfit(t_norm, flux, deg=degree, w=np.sqrt(weights))
    prediction = np.polyval(coeffs, t_norm)
    residuals = flux - prediction
    variance = np.full_like(flux, np.median(flux_err) ** 2)
    return prediction, residuals, variance


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class GPVariabilityModel:
    """Gaussian Process model for stellar variability using celerite2.

    Models the stellar variability as a correlated noise process with a
    Simple Harmonic Oscillator (SHO) or Matern-3/2 kernel.  The GP is
    conditioned on the out-of-transit cadences and then used to predict
    the variability at all cadences.

    Parameters
    ----------
    kernel_type : str
        Kernel to use: ``'sho'`` (SHO) or ``'matern32'`` (Matern-3/2).
    log_sigma : float
        Initial log-amplitude of the kernel.
    log_rho : float
        Initial log-timescale (SHO: quality / rho; Matern: rho).
    log_Q : float
        Initial log-quality factor Q for SHO kernel (ignored for Matern).
    log_jitter : float
        Initial log-jitter term (extra white noise added in quadrature).
    poly_degree : int
        Polynomial degree for the fallback detrending.
    max_iter : int
        Maximum optimisation iterations (passed to scipy L-BFGS-B).
    config : dict, optional
        Pipeline configuration dictionary.

    Examples
    --------
    >>> gp = GPVariabilityModel(kernel_type='sho')
    >>> result = gp.fit(time, flux, flux_err, transit_mask=mask)
    >>> residuals = gp.get_residuals()
    """

    def __init__(
        self,
        kernel_type: str = "sho",
        log_sigma: float = -3.0,
        log_rho: float = 1.0,
        log_Q: float = 1.0,
        log_jitter: float = -10.0,
        poly_degree: int = 3,
        max_iter: int = 500,
        config: Optional[dict] = None,
    ) -> None:
        self.kernel_type = kernel_type
        self.log_sigma = log_sigma
        self.log_rho = log_rho
        self.log_Q = log_Q
        self.log_jitter = log_jitter
        self.poly_degree = poly_degree
        self.max_iter = max_iter
        self.config = config

        if config is not None:
            self.kernel_type = get(config, "conditioning.gp.kernel_type", self.kernel_type)
            self.log_sigma = float(get(config, "conditioning.gp.log_sigma", self.log_sigma))
            self.log_rho = float(get(config, "conditioning.gp.log_rho", self.log_rho))
            self.log_Q = float(get(config, "conditioning.gp.log_Q", self.log_Q))
            self.log_jitter = float(get(config, "conditioning.gp.log_jitter", self.log_jitter))

        self._gp: Optional[object] = None  # celerite2 GP object
        self._result: Optional[GPResult] = None

    # ------------------------------------------------------------------
    # Internal: build celerite2 GP
    # ------------------------------------------------------------------

    def _build_gp(self, log_sigma: float, log_rho: float, log_Q: float, log_jitter: float):
        """Construct a celerite2 GP object from log-parameters.

        Parameters
        ----------
        log_sigma : float
            Log-amplitude.
        log_rho : float
            Log-timescale or rho parameter.
        log_Q : float
            Log-quality factor (SHO only).
        log_jitter : float
            Log-jitter term.

        Returns
        -------
        celerite2.GaussianProcess
            Newly constructed GP object.
        """
        jitter = celerite2_terms.JitterTerm(log_sigma=log_jitter)
        if self.kernel_type == "sho":
            sigma = np.exp(log_sigma)
            rho = np.exp(log_rho)
            Q = np.exp(log_Q)
            # SHOTerm: w0 = 2*pi/rho, S0 derived from sigma and Q
            w0 = 2.0 * np.pi / rho
            S0 = (sigma ** 2) / (w0 * Q)
            term = celerite2_terms.SHOTerm(S0=S0, w0=w0, Q=Q)
        elif self.kernel_type == "matern32":
            term = celerite2_terms.Matern32Term(
                sigma=np.exp(log_sigma), rho=np.exp(log_rho)
            )
        else:
            raise ValueError(f"Unknown kernel_type '{self.kernel_type}'. Use 'sho' or 'matern32'.")

        kernel = term + jitter
        gp = celerite2.GaussianProcess(kernel, mean=0.0)
        return gp

    # ------------------------------------------------------------------
    # Internal: negative log-likelihood and gradient
    # ------------------------------------------------------------------

    def _neg_log_like(
        self,
        params: np.ndarray,
        time_oot: np.ndarray,
        flux_oot: np.ndarray,
        flux_err_oot: np.ndarray,
    ) -> float:
        """Compute negative log-likelihood for the celerite2 GP.

        Parameters
        ----------
        params : np.ndarray
            Parameter vector [log_sigma, log_rho, log_Q, log_jitter].
        time_oot, flux_oot, flux_err_oot : np.ndarray
            Out-of-transit data.

        Returns
        -------
        float
            Negative log-likelihood (inf on numerical failure).
        """
        log_sigma, log_rho, log_Q, log_jitter = params
        try:
            gp = self._build_gp(log_sigma, log_rho, log_Q, log_jitter)
            gp.compute(time_oot, diag=flux_err_oot ** 2 + 1e-12)
            ll = gp.log_likelihood(flux_oot)
            return -ll if np.isfinite(ll) else np.inf
        except Exception:
            return np.inf

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        transit_mask: Optional[np.ndarray] = None,
    ) -> "GPVariabilityModel":
        """Fit the GP to out-of-transit cadences.

        Parameters
        ----------
        time : np.ndarray
            Full observation time array (days).
        flux : np.ndarray
            Full flux array.
        flux_err : np.ndarray
            Full flux uncertainty array.
        transit_mask : np.ndarray of bool, optional
            Boolean array with ``True`` for **in-transit** cadences.  These
            cadences are excluded from the GP fit.  If ``None``, all
            cadences are used.

        Returns
        -------
        GPVariabilityModel
            Self (for method chaining).
        """
        time = np.asarray(time, dtype=float)
        flux = np.asarray(flux, dtype=float)
        flux_err = np.asarray(flux_err, dtype=float)

        if transit_mask is None:
            transit_mask = np.zeros(len(time), dtype=bool)
        else:
            transit_mask = np.asarray(transit_mask, dtype=bool)

        oot_mask = ~transit_mask & np.isfinite(flux) & np.isfinite(time)
        logger.info(
            "GP fit: %d total cadences, %d out-of-transit, kernel=%s",
            len(time),
            oot_mask.sum(),
            self.kernel_type,
        )

        if not _CELERITE2_AVAILABLE:
            logger.warning("celerite2 not available. Using polynomial fallback.")
            self._result = self._do_polynomial_fallback(time, flux, flux_err, oot_mask)
            return self

        time_oot = time[oot_mask]
        flux_oot = flux[oot_mask]
        err_oot = flux_err[oot_mask]

        # Remove mean from out-of-transit flux for GP fitting
        oot_mean = np.nanmedian(flux_oot)
        flux_oot_zm = flux_oot - oot_mean

        x0 = np.array([self.log_sigma, self.log_rho, self.log_Q, self.log_jitter])

        try:
            result = minimize(
                self._neg_log_like,
                x0,
                args=(time_oot, flux_oot_zm, err_oot),
                method="L-BFGS-B",
                options={"maxiter": self.max_iter, "ftol": 1e-10},
                bounds=[
                    (-15, 5),   # log_sigma
                    (-3, 10),   # log_rho
                    (-5, 5),    # log_Q
                    (-20, 0),   # log_jitter
                ],
            )
            converged = result.success
            best_params = result.x
            final_ll = -result.fun

            if not converged:
                logger.warning(
                    "GP optimisation did not fully converge: %s. "
                    "Using best-so-far parameters.",
                    result.message,
                )
        except Exception as exc:
            logger.error("GP optimisation raised an exception: %s. Using fallback.", exc)
            self._result = self._do_polynomial_fallback(time, flux, flux_err, oot_mask)
            return self

        # Build GP with optimal parameters and compute on full time array
        log_sigma, log_rho, log_Q, log_jitter = best_params
        try:
            gp = self._build_gp(log_sigma, log_rho, log_Q, log_jitter)
            gp.compute(time_oot, diag=err_oot ** 2 + 1e-12)
            gp.log_likelihood(flux_oot_zm)

            # Predict on full time grid
            mu, var = gp.predict(flux_oot_zm, t=time, return_var=True)
            gp_mean = mu + oot_mean
            gp_var = np.abs(var)

            self._gp = gp
            kernel_params = {
                "log_sigma": float(log_sigma),
                "log_rho": float(log_rho),
                "log_Q": float(log_Q),
                "log_jitter": float(log_jitter),
                "kernel_type": self.kernel_type,
            }

            self._result = GPResult(
                time=time,
                gp_mean_prediction=gp_mean,
                residuals=flux - gp_mean,
                gp_variance=gp_var,
                kernel_params=kernel_params,
                converged=converged,
                fallback_used=False,
                log_likelihood=float(final_ll),
            )
            logger.info(
                "GP fit complete. log_L=%.4f, converged=%s", final_ll, converged
            )
        except Exception as exc:
            logger.error(
                "GP prediction failed: %s. Falling back to polynomial.", exc
            )
            self._result = self._do_polynomial_fallback(time, flux, flux_err, oot_mask)

        return self

    # ------------------------------------------------------------------

    def _do_polynomial_fallback(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        oot_mask: np.ndarray,
    ) -> GPResult:
        """Internal wrapper for polynomial fallback.

        Parameters
        ----------
        time, flux, flux_err : np.ndarray
            Full arrays.
        oot_mask : np.ndarray of bool
            Out-of-transit mask.

        Returns
        -------
        GPResult
            Result with ``fallback_used=True``.
        """
        logger.info("Using polynomial (degree=%d) fallback.", self.poly_degree)
        t_oot = time[oot_mask]
        f_oot = flux[oot_mask]
        e_oot = flux_err[oot_mask]
        pred_oot, _, _ = _polynomial_fallback(t_oot, f_oot, e_oot, self.poly_degree)
        # Fit polynomial on OOT, evaluate on full time grid
        t_norm = (time - time[oot_mask].mean()) / (time[oot_mask].ptp() + 1e-12)
        t_norm_oot = (t_oot - t_oot.mean()) / (t_oot.ptp() + 1e-12)
        weights = 1.0 / (e_oot ** 2 + 1e-12)
        coeffs = np.polyfit(t_norm_oot, f_oot, deg=self.poly_degree, w=np.sqrt(weights))
        gp_mean = np.polyval(coeffs, t_norm)
        residuals = flux - gp_mean
        gp_var = np.full_like(flux, np.median(flux_err) ** 2)
        return GPResult(
            time=time,
            gp_mean_prediction=gp_mean,
            residuals=residuals,
            gp_variance=gp_var,
            kernel_params={"fallback": "polynomial", "degree": self.poly_degree},
            converged=False,
            fallback_used=True,
            log_likelihood=float("nan"),
        )

    # ------------------------------------------------------------------

    def predict(self, time: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate GP mean and variance at arbitrary time points.

        Parameters
        ----------
        time : np.ndarray
            Times at which to evaluate the GP.

        Returns
        -------
        tuple of np.ndarray
            ``(mean, variance)`` arrays.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if self._result is None:
            raise RuntimeError("Call fit() before predict().")
        if self._result.fallback_used or self._gp is None:
            # Polynomial evaluation at new times
            t_ref = self._result.time
            t_norm_ref = (t_ref - t_ref.mean()) / (t_ref.ptp() + 1e-12)
            t_norm_new = (time - t_ref.mean()) / (t_ref.ptp() + 1e-12)
            # Re-use the stored prediction by interpolation
            from scipy.interpolate import interp1d

            f_interp = interp1d(
                t_norm_ref,
                self._result.gp_mean_prediction,
                kind="linear",
                fill_value="extrapolate",
            )
            mean = f_interp(t_norm_new)
            variance = np.full_like(mean, np.mean(self._result.gp_variance))
            return mean, variance

        # celerite2 prediction
        # The GP was fitted on OOT zero-meaned flux; need oot_mean
        oot_mean = float(
            np.median(
                self._result.time[
                    np.isfinite(self._result.gp_mean_prediction)
                ]
            )
        )
        mu, var = self._gp.predict(
            self._result.gp_mean_prediction - oot_mean,
            t=time,
            return_var=True,
        )
        return mu + oot_mean, np.abs(var)

    # ------------------------------------------------------------------

    def get_residuals(self) -> np.ndarray:
        """Return the residuals (flux - GP model).

        Returns
        -------
        np.ndarray
            Residual flux array.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if self._result is None:
            raise RuntimeError("Call fit() before get_residuals().")
        return self._result.residuals

    # ------------------------------------------------------------------

    def get_result(self) -> GPResult:
        """Return the full GPResult dataclass.

        Returns
        -------
        GPResult
            Full result container.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if self._result is None:
            raise RuntimeError("Call fit() before get_result().")
        return self._result

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"GPVariabilityModel(kernel_type='{self.kernel_type}', "
            f"log_sigma={self.log_sigma}, log_rho={self.log_rho}, "
            f"log_Q={self.log_Q})"
        )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_lightcurve_npy(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a light curve from a .npz file with keys time, flux, flux_err."""
    data = np.load(path, allow_pickle=False)
    return data["time"], data["flux"], data["flux_err"]


def _load_lightcurve_fits(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load time, flux, flux_err from a TESS-style FITS file."""
    from astropy.io import fits as af

    with af.open(path) as hdul:
        for hdu in hdul:
            if hasattr(hdu, "columns") and hdu.columns is not None and hdu.data is not None:
                data = hdu.data
                colnames = [c.name for c in hdu.columns]
                t_col = "TIME" if "TIME" in colnames else colnames[0]
                f_col = (
                    "PDCSAP_FLUX" if "PDCSAP_FLUX" in colnames
                    else ("FLUX" if "FLUX" in colnames else colnames[1])
                )
                e_col = (
                    "PDCSAP_FLUX_ERR" if "PDCSAP_FLUX_ERR" in colnames
                    else ("FLUX_ERR" if "FLUX_ERR" in colnames else colnames[2])
                )
                t = np.asarray(data[t_col], dtype=float)
                f = np.asarray(data[f_col], dtype=float)
                e = np.asarray(data[e_col], dtype=float)
                ok = np.isfinite(t) & np.isfinite(f)
                return t[ok], f[ok], np.where(np.isfinite(e[ok]), e[ok], np.nanmedian(e))
    raise ValueError(f"Could not read light curve from {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GP stellar variability modelling using celerite2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_file", type=str, help="Path to FITS or .npz light curve.")
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output .npz file (gp_mean, residuals, gp_variance, time). "
             "Defaults to <input>_gp.npz.",
    )
    parser.add_argument(
        "--kernel", type=str, default="sho", choices=["sho", "matern32"],
        help="celerite2 kernel type.",
    )
    parser.add_argument(
        "--transit-mask-npz", type=str, default=None,
        help=".npz file with a boolean 'transit_mask' array.",
    )
    parser.add_argument(
        "--log-sigma", type=float, default=-3.0,
        help="Initial log-amplitude.",
    )
    parser.add_argument(
        "--log-rho", type=float, default=1.0,
        help="Initial log-timescale.",
    )
    parser.add_argument(
        "--log-Q", type=float, default=1.0,
        help="Initial log-quality factor (SHO only).",
    )
    parser.add_argument(
        "--max-iter", type=int, default=500,
        help="Maximum optimisation iterations.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Pipeline config YAML path.",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: Optional[list] = None) -> None:
    """Entry point for the GP variability modelling CLI."""
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
        p = Path(args.input_file)
        output = str(p.parent / (p.stem + "_gp.npz"))

    # Load data
    if args.input_file.endswith(".npz"):
        time, flux, flux_err = _load_lightcurve_npy(args.input_file)
    else:
        time, flux, flux_err = _load_lightcurve_fits(args.input_file)

    # Load transit mask if provided
    transit_mask: Optional[np.ndarray] = None
    if args.transit_mask_npz:
        tm_data = np.load(args.transit_mask_npz, allow_pickle=False)
        transit_mask = tm_data["transit_mask"].astype(bool)

    # Normalise flux
    median_flux = np.nanmedian(flux)
    if median_flux != 0:
        flux_norm = flux / median_flux
        flux_err_norm = flux_err / median_flux
    else:
        flux_norm = flux
        flux_err_norm = flux_err

    model = GPVariabilityModel(
        kernel_type=args.kernel,
        log_sigma=args.log_sigma,
        log_rho=args.log_rho,
        log_Q=args.log_Q,
        max_iter=args.max_iter,
        config=config,
    )
    model.fit(time, flux_norm, flux_err_norm, transit_mask=transit_mask)
    result = model.get_result()

    np.savez(
        output,
        time=result.time,
        gp_mean=result.gp_mean_prediction,
        residuals=result.residuals,
        gp_variance=result.gp_variance,
    )
    logger.info("Saved GP result to %s", output)
    print(f"GP result saved to: {output}")
    print(f"  Converged   : {result.converged}")
    print(f"  Fallback    : {result.fallback_used}")
    print(f"  log_L       : {result.log_likelihood:.4f}")
    print(f"  Residual σ  : {float(np.nanstd(result.residuals)):.6f}")


if __name__ == "__main__":
    main()
