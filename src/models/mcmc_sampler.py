"""
src/models/mcmc_sampler.py
============================
emcee Ensemble MCMC sampler for Bayesian transit parameter estimation.

Wraps ``emcee.EnsembleSampler`` to run multi-walker MCMC sampling around a
BLS-derived initial guess, then extracts posterior summaries and saves
results to disk.

Default sampler settings (overridable via config)::

    n_walkers     = 64
    n_steps       = 2000
    n_burn        = 500

Parameter vector theta (9-dimensional)::

    [log_period, t0, log_rp_rs, b, log_a_rs, u1, u2, log_gp_amp, log_gp_rho]

Outputs saved per target::

    outputs/{tic_id}_mcmc_samples.npy      -- flat_samples array (N_flat, 9)
    outputs/{tic_id}_parameters.json       -- median and 1-sigma estimates

Class mapping::

    PLANET           = 0
    ECLIPSING_BINARY = 1
    BLEND            = 2
    NOISE            = 3

Functions
---------
run_mcmc(model, bls_result, config, tic_id) -> (flat_samples, param_dict)
    Main entry point.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import emcee
    EMCEE_AVAILABLE = True
except ImportError:
    emcee = None  # type: ignore[assignment]
    EMCEE_AVAILABLE = False
    warnings.warn("emcee not installed.  MCMC sampling unavailable.", ImportWarning)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    tqdm = None  # type: ignore[assignment]
    TQDM_AVAILABLE = False

from models.transit_gp_model import TransitGPModel, PARAM_NAMES, N_PARAMS
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Walker initialisation
# ---------------------------------------------------------------------------

def _initialise_walkers(
    theta0: np.ndarray,
    n_walkers: int,
    scatter: float = 1e-3,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Scatter walkers around an initial position.

    Parameters
    ----------
    theta0 : np.ndarray
        Shape ``(N_PARAMS,)`` -- initial parameter guess.
    n_walkers : int
        Number of walkers (must be even and >= 2 * N_PARAMS).
    scatter : float
        Scale of the Gaussian scatter.  Default ``1e-3``.
    rng : np.random.Generator, optional
        Random number generator.  Uses ``np.random.default_rng(0)`` if ``None``.

    Returns
    -------
    np.ndarray
        Shape ``(n_walkers, N_PARAMS)`` -- initial walker positions.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    pos = theta0[np.newaxis, :] + scatter * rng.standard_normal(
        (n_walkers, len(theta0))
    )
    # Ensure impact parameter b stays positive
    b_idx = PARAM_NAMES.index("b")
    pos[:, b_idx] = np.abs(pos[:, b_idx])
    return pos


# ---------------------------------------------------------------------------
# Posterior summary
# ---------------------------------------------------------------------------

def _summarise_posterior(flat_samples: np.ndarray) -> dict:
    """Extract median + 16th/84th percentile credible intervals.

    Parameters
    ----------
    flat_samples : np.ndarray
        Shape ``(N_flat, N_PARAMS)`` -- flat posterior samples.

    Returns
    -------
    dict
        Per-parameter summary with keys ``{name}``, ``{name}_err_lo``,
        ``{name}_err_hi``.  Additionally includes derived physical parameters:
        period, depth_ppm, duration_hrs, rp_rs.
    """
    summary: dict = {}
    for i, name in enumerate(PARAM_NAMES):
        samples_i = flat_samples[:, i]
        med = float(np.median(samples_i))
        lo = float(np.percentile(samples_i, 16))
        hi = float(np.percentile(samples_i, 84))
        summary[name] = med
        summary[f"{name}_err_lo"] = med - lo
        summary[f"{name}_err_hi"] = hi - med

    # Derived parameters
    log_period = flat_samples[:, PARAM_NAMES.index("log_period")]
    period_samples = np.exp(log_period)
    log_rp_rs = flat_samples[:, PARAM_NAMES.index("log_rp_rs")]
    rp_rs_samples = np.exp(log_rp_rs)

    # Period
    summary["period"] = float(np.median(period_samples))
    summary["period_err_lo"] = float(np.median(period_samples) - np.percentile(period_samples, 16))
    summary["period_err_hi"] = float(np.percentile(period_samples, 84) - np.median(period_samples))

    # Depth in ppm (Rp/Rs)^2
    depth_samples = rp_rs_samples**2 * 1e6
    summary["depth_ppm"] = float(np.median(depth_samples))
    summary["depth_ppm_err_lo"] = float(np.median(depth_samples) - np.percentile(depth_samples, 16))
    summary["depth_ppm_err_hi"] = float(np.percentile(depth_samples, 84) - np.median(depth_samples))

    # Rp/Rs
    summary["rp_rs"] = float(np.median(rp_rs_samples))
    summary["rp_rs_err"] = float(
        (np.percentile(rp_rs_samples, 84) - np.percentile(rp_rs_samples, 16)) / 2
    )

    # Transit duration: approximate formula
    # T_dur = (P/pi) * arcsin( sqrt( (1+rp/rs)^2 - b^2 ) / (a/rs) )
    log_a_rs = flat_samples[:, PARAM_NAMES.index("log_a_rs")]
    a_rs_samples = np.exp(log_a_rs)
    b_samples = flat_samples[:, PARAM_NAMES.index("b")]

    arg = np.sqrt(
        np.clip((1 + rp_rs_samples)**2 - b_samples**2, 0, None)
    ) / np.clip(a_rs_samples, 1e-3, None)
    arg = np.clip(arg, -1, 1)
    dur_days = (period_samples / np.pi) * np.arcsin(arg)
    dur_hrs = dur_days * 24.0

    summary["duration_hrs"] = float(np.nanmedian(dur_hrs))
    summary["duration_hrs_err_lo"] = float(
        np.nanmedian(dur_hrs) - np.nanpercentile(dur_hrs, 16)
    )
    summary["duration_hrs_err_hi"] = float(
        np.nanpercentile(dur_hrs, 84) - np.nanmedian(dur_hrs)
    )

    # t0
    t0_samples = flat_samples[:, PARAM_NAMES.index("t0")]
    summary["t0"] = float(np.median(t0_samples))
    summary["t0_err"] = float(
        (np.percentile(t0_samples, 84) - np.percentile(t0_samples, 16)) / 2
    )

    return summary


# ---------------------------------------------------------------------------
# Main MCMC function
# ---------------------------------------------------------------------------

def run_mcmc(
    model: TransitGPModel,
    bls_result: dict,
    config: Optional[dict] = None,
    tic_id: str = "unknown",
) -> tuple[np.ndarray, dict]:
    """Run emcee MCMC sampling for transit parameter estimation.

    Parameters
    ----------
    model : TransitGPModel
        The joint GP + transit model (data already attached).
    bls_result : dict
        BLS periodogram results.  Expected keys:
        ``period``, ``t0``, ``depth``, ``duration``.
    config : dict, optional
        Pipeline configuration dict.  MCMC settings read from
        ``config['mcmc']``.  Defaults used if ``None`` or keys missing.
    tic_id : str, optional
        Target identifier used for output file naming.  Default ``'unknown'``.

    Returns
    -------
    flat_samples : np.ndarray
        Shape ``(N_flat, N_PARAMS)`` -- flattened posterior samples.
    param_dict : dict
        Summary statistics: medians, 16th/84th percentile errors, and derived
        physical parameters.

    Raises
    ------
    RuntimeError
        If emcee is not installed.
    """
    if not EMCEE_AVAILABLE:
        raise RuntimeError(
            "emcee is not installed.  Install with: pip install emcee"
        )

    config = config or {}
    mcmc_cfg = config.get("mcmc", {})
    n_walkers = int(mcmc_cfg.get("n_walkers", 64))
    n_steps = int(mcmc_cfg.get("n_steps", 2000))
    n_burn = int(mcmc_cfg.get("n_burn", 500))
    output_dir = Path(mcmc_cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure n_walkers is even and >= 2*N_PARAMS
    n_walkers = max(n_walkers, 2 * N_PARAMS + 2)
    if n_walkers % 2 != 0:
        n_walkers += 1

    logger.info(
        "Running MCMC for TIC %s: n_walkers=%d n_steps=%d n_burn=%d",
        tic_id, n_walkers, n_steps, n_burn,
    )

    # ------------------------------------------------------------------
    # Build initial walker positions
    # ------------------------------------------------------------------
    theta0 = TransitGPModel.theta_from_bls(
        period=float(bls_result.get("period", 1.0)),
        t0=float(bls_result.get("t0", model.time.mean())),
        depth=float(bls_result.get("depth", 1e-3)),
        duration=float(bls_result.get("duration", 0.1)),
    )
    p0 = _initialise_walkers(theta0, n_walkers)

    # ------------------------------------------------------------------
    # Sampler
    # ------------------------------------------------------------------
    sampler = emcee.EnsembleSampler(
        n_walkers,
        N_PARAMS,
        model.log_posterior,
    )

    # ------------------------------------------------------------------
    # Burn-in phase
    # ------------------------------------------------------------------
    logger.info("Starting burn-in (%d steps)...", n_burn)
    try:
        if TQDM_AVAILABLE:
            with tqdm(total=n_burn, desc=f"TIC {tic_id} burn-in") as pbar:
                for sample in sampler.sample(p0, iterations=n_burn, progress=False):
                    pbar.update(1)
        else:
            sampler.run_mcmc(p0, n_burn, progress=False)
    except Exception as exc:
        logger.error("Burn-in failed: %s", exc)
        raise

    p_burned = sampler.get_last_sample().coords
    sampler.reset()

    # ------------------------------------------------------------------
    # Production run
    # ------------------------------------------------------------------
    logger.info("Starting production run (%d steps)...", n_steps)
    try:
        if TQDM_AVAILABLE:
            with tqdm(total=n_steps, desc=f"TIC {tic_id} MCMC") as pbar:
                for sample in sampler.sample(p_burned, iterations=n_steps, progress=False):
                    pbar.update(1)
        else:
            sampler.run_mcmc(p_burned, n_steps, progress=False)
    except Exception as exc:
        logger.error("MCMC production run failed: %s", exc)
        raise

    # ------------------------------------------------------------------
    # Extract flat samples (discard 20% as additional burn-in)
    # ------------------------------------------------------------------
    thin = max(1, int(n_steps // 100))
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        thin = max(1, int(np.nanmin(tau) / 2))
        logger.info("Autocorrelation times: %s (thinning by %d)", tau, thin)
    except Exception as exc:
        logger.warning("Could not compute autocorrelation time: %s", exc)

    discard = n_steps // 5
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    logger.info("Flat samples shape: %s", flat_samples.shape)

    # ------------------------------------------------------------------
    # Save samples
    # ------------------------------------------------------------------
    samples_path = output_dir / f"{tic_id}_mcmc_samples.npy"
    np.save(samples_path, flat_samples)
    logger.info("MCMC samples saved to %s", samples_path)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    param_dict = _summarise_posterior(flat_samples)
    param_dict["tic_id"] = str(tic_id)
    param_dict["n_flat_samples"] = int(flat_samples.shape[0])
    param_dict["n_walkers"] = n_walkers
    param_dict["n_steps"] = n_steps
    param_dict["n_burn"] = n_burn

    params_path = output_dir / f"{tic_id}_parameters.json"
    with open(params_path, "w") as fh:
        json.dump(param_dict, fh, indent=2, default=float)
    logger.info("Parameter estimates saved to %s", params_path)

    logger.info(
        "MCMC complete for TIC %s: period=%.4f +%.4f/-%.4f days, "
        "depth=%.0f +%.0f/-%.0f ppm",
        tic_id,
        param_dict["period"],
        param_dict["period_err_hi"],
        param_dict["period_err_lo"],
        param_dict["depth_ppm"],
        param_dict["depth_ppm_err_hi"],
        param_dict["depth_ppm_err_lo"],
    )

    return flat_samples, param_dict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run emcee MCMC transit parameter estimation."
    )
    parser.add_argument("--tic_id", type=str, default="test_tic",
                        help="TIC identifier for output file naming.")
    parser.add_argument("--period", type=float, default=3.0,
                        help="BLS period estimate (days).")
    parser.add_argument("--t0", type=float, default=None,
                        help="BLS t0 estimate (days). Default: centre of time array.")
    parser.add_argument("--depth", type=float, default=0.01,
                        help="BLS depth estimate (fractional).")
    parser.add_argument("--duration", type=float, default=0.1,
                        help="BLS duration estimate (days).")
    parser.add_argument("--n_walkers", type=int, default=32,
                        help="Number of MCMC walkers (for quick test).")
    parser.add_argument("--n_steps", type=int, default=200,
                        help="Number of MCMC steps (for quick test).")
    parser.add_argument("--n_burn", type=int, default=50,
                        help="Number of burn-in steps.")
    parser.add_argument("--n_points", type=int, default=300,
                        help="Number of synthetic data points to generate.")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory.")
    args = parser.parse_args()

    # Generate synthetic data
    rng = np.random.default_rng(42)
    time = np.sort(rng.uniform(0, 30, args.n_points))
    flux = 1.0 + rng.normal(0, 1e-3, args.n_points)
    flux_err = np.full(args.n_points, 1e-3)

    model = TransitGPModel(time, flux, flux_err)
    t0 = args.t0 if args.t0 is not None else float(time[len(time) // 4])

    bls_result = {
        "period": args.period,
        "t0": t0,
        "depth": args.depth,
        "duration": args.duration,
    }

    config = {
        "mcmc": {
            "n_walkers": args.n_walkers,
            "n_steps": args.n_steps,
            "n_burn": args.n_burn,
            "output_dir": args.output_dir,
        }
    }

    flat_samples, param_dict = run_mcmc(
        model=model,
        bls_result=bls_result,
        config=config,
        tic_id=args.tic_id,
    )

    print(f"\n=== Parameter Estimates for TIC {args.tic_id} ===")
    for key in ["period", "depth_ppm", "duration_hrs", "rp_rs", "t0"]:
        val = param_dict.get(key, float("nan"))
        err_lo = param_dict.get(f"{key}_err_lo", param_dict.get(f"{key}_err", float("nan")))
        err_hi = param_dict.get(f"{key}_err_hi", err_lo)
        print(f"  {key:20s} = {val:.4f}  +{err_hi:.4f} / -{err_lo:.4f}")
