"""
src/models/transit_gp_model.py
================================
Joint Gaussian Process + transit light-curve model for Bayesian parameter
estimation of candidate exoplanet transits.

Uses ``batman`` for physical transit model computation and ``celerite2`` for
correlated-noise (GP) modelling of stellar variability and instrumental
systematics.

Parameters inferred (9-dimensional theta vector)
-------------------------------------------------
0. ``log_period``    : log(orbital period / days)
1. ``t0``            : mid-transit time (BJD or relative days)
2. ``log_rp_rs``     : log(planet-to-star radius ratio)
3. ``b``             : impact parameter ∈ [0, 1+Rp/Rs)
4. ``log_a_rs``      : log(semi-major axis / stellar radius)
5. ``u1``            : quadratic limb-darkening coefficient 1
6. ``u2``            : quadratic limb-darkening coefficient 2
7. ``log_gp_amp``    : log(GP amplitude)
8. ``log_gp_rho``    : log(GP length scale / days)

Class mapping::

    PLANET           = 0
    ECLIPSING_BINARY = 1
    BLEND            = 2
    NOISE            = 3

Classes
-------
TransitGPModel
    Encapsulates the joint GP + transit likelihood for use with emcee.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

try:
    import batman
    BATMAN_AVAILABLE = True
except ImportError:
    batman = None  # type: ignore[assignment]
    BATMAN_AVAILABLE = False
    warnings.warn(
        "batman-package not installed.  Transit model will return zeros.",
        ImportWarning, stacklevel=2,
    )

try:
    import celerite2
    from celerite2 import terms as celerite2_terms
    CELERITE2_AVAILABLE = True
except ImportError:
    celerite2 = None  # type: ignore[assignment]
    celerite2_terms = None  # type: ignore[assignment]
    CELERITE2_AVAILABLE = False
    warnings.warn(
        "celerite2 not installed.  GP model will fall back to diagonal likelihood.",
        ImportWarning, stacklevel=2,
    )

from utils.logger import get_logger

logger = get_logger(__name__)

# Parameter names and indices for reference
PARAM_NAMES = [
    "log_period",
    "t0",
    "log_rp_rs",
    "b",
    "log_a_rs",
    "u1",
    "u2",
    "log_gp_amp",
    "log_gp_rho",
]
N_PARAMS = len(PARAM_NAMES)


# ---------------------------------------------------------------------------
# TransitGPModel
# ---------------------------------------------------------------------------

class TransitGPModel:
    """Joint transit + Gaussian Process model for transit parameter estimation.

    Parameters
    ----------
    time : np.ndarray
        Observation timestamps (BJD or relative days).  Shape ``(N,)``.
    flux : np.ndarray
        Normalised flux values.  Shape ``(N,)``.
    flux_err : np.ndarray
        Flux uncertainties (1-sigma).  Shape ``(N,)``.

    Attributes
    ----------
    time : np.ndarray
    flux : np.ndarray
    flux_err : np.ndarray
    _batman_params : batman.TransitParams or None
    _batman_model  : batman.TransitModel or None
    _gp : celerite2.GaussianProcess or None

    Examples
    --------
    >>> model = TransitGPModel(time, flux, flux_err)
    >>> theta = [np.log(3.0), 1234.5, np.log(0.1), 0.2, np.log(10.0),
    ...          0.3, 0.1, np.log(1e-4), np.log(1.0)]
    >>> lp = model.log_posterior(theta)
    """

    def __init__(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
    ) -> None:
        self.time = np.asarray(time, dtype=np.float64)
        self.flux = np.asarray(flux, dtype=np.float64)
        self.flux_err = np.asarray(flux_err, dtype=np.float64)

        if not (len(self.time) == len(self.flux) == len(self.flux_err)):
            raise ValueError(
                "time, flux, flux_err must all have the same length."
            )

        self._batman_params: Optional[object] = None
        self._batman_model: Optional[object] = None
        self._gp: Optional[object] = None

        self._cached_theta: Optional[tuple] = None

        logger.debug(
            "TransitGPModel initialised: N=%d batman=%s celerite2=%s",
            len(self.time), BATMAN_AVAILABLE, CELERITE2_AVAILABLE,
        )

    # ------------------------------------------------------------------
    # Transit model
    # ------------------------------------------------------------------

    def get_transit_model(self, params_dict: dict) -> np.ndarray:
        """Compute the batman transit flux model.

        Parameters
        ----------
        params_dict : dict
            Transit parameters with keys:

            * ``period`` : orbital period (days)
            * ``t0``     : mid-transit time
            * ``rp_rs``  : planet-to-star radius ratio
            * ``b``      : impact parameter
            * ``a_rs``   : semi-major axis / stellar radius
            * ``u1``, ``u2`` : limb-darkening coefficients

        Returns
        -------
        np.ndarray
            Shape ``(N,)`` -- relative flux values (1.0 outside transit).
        """
        if not BATMAN_AVAILABLE:
            return np.ones_like(self.time)

        period = float(params_dict["period"])
        t0 = float(params_dict["t0"])
        rp_rs = float(params_dict["rp_rs"])
        b = float(params_dict["b"])
        a_rs = float(params_dict["a_rs"])
        u1 = float(params_dict["u1"])
        u2 = float(params_dict["u2"])

        # Validate
        if period <= 0 or rp_rs <= 0 or a_rs <= 0:
            return np.ones_like(self.time)

        # Compute inclination from impact parameter
        # b = (a/R_*) cos(i)  =>  i = arccos(b * R_* / a)
        cos_i = b / a_rs
        if abs(cos_i) > 1.0:
            return np.ones_like(self.time)
        inc = np.degrees(np.arccos(cos_i))

        # Initialise or update batman
        if self._batman_params is None:
            self._batman_params = batman.TransitParams()
        p = self._batman_params
        p.t0 = t0
        p.per = period
        p.rp = rp_rs
        p.a = a_rs
        p.inc = inc
        p.ecc = 0.0
        p.w = 90.0
        p.u = [u1, u2]
        p.limb_dark = "quadratic"

        if self._batman_model is None:
            self._batman_model = batman.TransitModel(p, self.time)
        else:
            self._batman_model = batman.TransitModel(p, self.time)

        return self._batman_model.light_curve(p)

    def _unpack_theta(self, theta: np.ndarray) -> dict:
        """Convert raw theta vector to a dict of physical parameters."""
        log_period, t0, log_rp_rs, b, log_a_rs, u1, u2, log_gp_amp, log_gp_rho = theta
        return {
            "period": np.exp(log_period),
            "t0": t0,
            "rp_rs": np.exp(log_rp_rs),
            "b": b,
            "a_rs": np.exp(log_a_rs),
            "u1": u1,
            "u2": u2,
            "gp_amp": np.exp(log_gp_amp),
            "gp_rho": np.exp(log_gp_rho),
        }

    # ------------------------------------------------------------------
    # Transit mask
    # ------------------------------------------------------------------

    def compute_transit_mask(
        self,
        period: float,
        t0: float,
        duration: float,
        n_durations: float = 1.5,
    ) -> np.ndarray:
        """Return boolean mask of in-transit cadences.

        Parameters
        ----------
        period : float
            Orbital period (days).
        t0 : float
            Mid-transit time.
        duration : float
            Transit duration (days).
        n_durations : float
            Half-window in transit duration units.  Default ``1.5``.

        Returns
        -------
        np.ndarray of bool
            ``True`` for in-transit cadences.
        """
        phase = ((self.time - t0 + period / 2) % period) - period / 2
        half_window = n_durations * duration / 2.0
        return np.abs(phase) < half_window

    # ------------------------------------------------------------------
    # Log-likelihood
    # ------------------------------------------------------------------

    def log_likelihood(self, theta: np.ndarray) -> float:
        """Compute the joint GP + transit log-likelihood.

        Parameters
        ----------
        theta : array-like of shape ``(9,)``
            Parameter vector ``[log_period, t0, log_rp_rs, b, log_a_rs,
            u1, u2, log_gp_amp, log_gp_rho]``.

        Returns
        -------
        float
            Log-likelihood value; ``-np.inf`` if parameters are invalid.
        """
        try:
            theta = np.asarray(theta, dtype=np.float64)
            if len(theta) != N_PARAMS:
                return -np.inf

            params = self._unpack_theta(theta)

            # Compute transit model
            transit_flux = self.get_transit_model(params)

            # Residuals
            residuals = self.flux - transit_flux

            if CELERITE2_AVAILABLE:
                return self._celerite_log_likelihood(
                    residuals,
                    gp_amp=params["gp_amp"],
                    gp_rho=params["gp_rho"],
                )
            else:
                return self._diagonal_log_likelihood(residuals)

        except Exception as exc:
            logger.debug("log_likelihood error: %s", exc)
            return -np.inf

    def _celerite_log_likelihood(
        self,
        residuals: np.ndarray,
        gp_amp: float,
        gp_rho: float,
    ) -> float:
        """Compute celerite2 GP log-likelihood for *residuals*."""
        try:
            # SHO term: overdamped oscillator (Matern-3/2-like)
            term = celerite2_terms.SHOTerm(
                sigma=gp_amp,
                rho=gp_rho,
                Q=1.0 / np.sqrt(2),
            )
            gp = celerite2.GaussianProcess(term)
            gp.compute(self.time, diag=self.flux_err**2 + 1e-12)
            return float(gp.log_likelihood(residuals))
        except Exception as exc:
            logger.debug("celerite2 GP error: %s", exc)
            return self._diagonal_log_likelihood(residuals)

    def _diagonal_log_likelihood(self, residuals: np.ndarray) -> float:
        """Fallback: diagonal Gaussian log-likelihood."""
        sigma2 = self.flux_err**2
        ll = -0.5 * np.sum(residuals**2 / sigma2 + np.log(2 * np.pi * sigma2))
        return float(ll)

    # ------------------------------------------------------------------
    # Log-prior
    # ------------------------------------------------------------------

    def log_prior(self, theta: np.ndarray) -> float:
        """Compute the log-prior probability.

        Priors used:
        * ``log_period``  : Uniform(-4, 4)  [days]
        * ``t0``          : Uniform(time.min, time.max)
        * ``log_rp_rs``   : Uniform(-5, 0)   [Rp/Rs]
        * ``b``           : Uniform(0, 1.1)
        * ``log_a_rs``    : Uniform(0, 5)
        * ``u1``          : Gaussian(0.3, 0.2) clipped to [-0.5, 1.5]
        * ``u2``          : Gaussian(0.1, 0.2) clipped to [-0.5, 1.5]
        * ``log_gp_amp``  : Uniform(-15, 0)
        * ``log_gp_rho``  : Uniform(-3, 3)

        Parameters
        ----------
        theta : array-like of shape ``(9,)``

        Returns
        -------
        float
            Log-prior; ``-np.inf`` outside bounds.
        """
        try:
            (log_period, t0, log_rp_rs, b,
             log_a_rs, u1, u2, log_gp_amp, log_gp_rho) = theta

            # Uniform priors (returns -inf if out of range)
            if not (-4.0 <= log_period <= 4.0):
                return -np.inf
            if not (self.time.min() - 1.0 <= t0 <= self.time.max() + 1.0):
                return -np.inf
            if not (-5.0 <= log_rp_rs <= 0.0):
                return -np.inf
            if not (0.0 <= b <= 1.1):
                return -np.inf
            if not (0.0 <= log_a_rs <= 5.0):
                return -np.inf
            if not (-15.0 <= log_gp_amp <= 0.0):
                return -np.inf
            if not (-3.0 <= log_gp_rho <= 3.0):
                return -np.inf

            # Limb-darkening: Gaussian priors
            lp_u1 = -0.5 * ((u1 - 0.3) / 0.2) ** 2
            lp_u2 = -0.5 * ((u2 - 0.1) / 0.2) ** 2

            # Physical constraint: u1 + u2 in [-0.5, 1.5]
            if not (-0.5 <= u1 <= 1.5) or not (-0.5 <= u2 <= 1.5):
                return -np.inf
            if not (-0.5 <= u1 + u2 <= 1.0):
                return -np.inf

            return float(lp_u1 + lp_u2)

        except Exception:
            return -np.inf

    # ------------------------------------------------------------------
    # Log-posterior
    # ------------------------------------------------------------------

    def log_posterior(self, theta: np.ndarray) -> float:
        """Compute the log-posterior = log_prior + log_likelihood.

        Parameters
        ----------
        theta : array-like of shape ``(9,)``

        Returns
        -------
        float
            Log-posterior.  Returns ``-np.inf`` if the prior is zero.
        """
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return float(lp + ll)

    # ------------------------------------------------------------------
    # Convenience: initial guess from BLS result
    # ------------------------------------------------------------------

    @staticmethod
    def theta_from_bls(
        period: float,
        t0: float,
        depth: float,
        duration: float,
    ) -> np.ndarray:
        """Build an initial theta vector from BLS estimates.

        Parameters
        ----------
        period : float
            BLS orbital period (days).
        t0 : float
            BLS transit epoch.
        depth : float
            BLS transit depth (fractional).
        duration : float
            BLS transit duration (days).

        Returns
        -------
        np.ndarray
            Shape ``(9,)`` -- initial parameter vector.
        """
        rp_rs = np.sqrt(max(depth, 1e-8))
        # Rough a/Rs estimate from period and duration (assuming i~90)
        # duration ~ (period/pi) * arcsin(Rs/a) -> a/Rs ~ period/(pi*duration) for small angles
        a_rs = max(period / (np.pi * max(duration, 1e-4)), 2.0)
        theta = np.array([
            np.log(period),       # log_period
            t0,                   # t0
            np.log(rp_rs),        # log_rp_rs
            0.2,                  # b
            np.log(a_rs),         # log_a_rs
            0.3,                  # u1
            0.1,                  # u2
            np.log(1e-4),         # log_gp_amp
            np.log(1.0),          # log_gp_rho
        ])
        return theta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke-test TransitGPModel log-posterior."
    )
    parser.add_argument("--n_points", type=int, default=500,
                        help="Number of synthetic time points.")
    parser.add_argument("--period", type=float, default=3.0)
    parser.add_argument("--depth", type=float, default=0.01)
    parser.add_argument("--duration", type=float, default=0.1)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    time = np.sort(rng.uniform(0, 30, args.n_points))
    flux = 1.0 + rng.normal(0, 1e-3, args.n_points)
    flux_err = np.full(args.n_points, 1e-3)

    model = TransitGPModel(time, flux, flux_err)
    theta0 = TransitGPModel.theta_from_bls(
        period=args.period,
        t0=float(time[len(time) // 4]),
        depth=args.depth,
        duration=args.duration,
    )

    print(f"theta0 = {theta0}")
    print(f"Parameter names: {PARAM_NAMES}")

    lp = model.log_prior(theta0)
    ll = model.log_likelihood(theta0)
    lpost = model.log_posterior(theta0)

    print(f"log_prior      = {lp:.4f}")
    print(f"log_likelihood = {ll:.4f}")
    print(f"log_posterior  = {lpost:.4f}")

    transit = model.get_transit_model(model._unpack_theta(theta0))
    print(f"Transit model: min={transit.min():.6f}, max={transit.max():.6f}")
