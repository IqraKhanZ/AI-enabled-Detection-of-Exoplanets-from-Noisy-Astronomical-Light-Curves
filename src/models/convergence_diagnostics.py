"""
src/models/convergence_diagnostics.py
========================================
MCMC convergence diagnostics for transit parameter MCMC runs.

Computes and visualises:

* Autocorrelation time ``tau`` per parameter (via emcee's built-in method).
* Gelman-Rubin R-hat statistic (implemented manually).
* Effective sample size = N / tau.
* Trace plots (parameter value vs. step) for all walkers.
* Autocorrelation plots.
* Corner plot using ``corner.corner()``.

Outputs
-------
* ``reports/mcmc_diagnostics/{tic_id}_traces.png``  -- trace plots
* ``reports/mcmc_diagnostics/{tic_id}_autocorr.png`` -- autocorrelation
* ``reports/mcmc_diagnostics/{tic_id}_corner.png``  -- corner plot
* printed convergence summary (R-hat < 1.1 = converged)

Functions
---------
diagnose(sampler, param_names, tic_id, flat_samples) -> dict

Class mapping::

    PLANET           = 0
    ECLIPSING_BINARY = 1
    BLEND            = 2
    NOISE            = 3
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import corner
    CORNER_AVAILABLE = True
except ImportError:
    corner = None  # type: ignore[assignment]
    CORNER_AVAILABLE = False
    warnings.warn("corner package not installed.  Corner plots disabled.", ImportWarning)

from utils.logger import get_logger

logger = get_logger(__name__)

# Convergence threshold for R-hat
RHAT_THRESHOLD = 1.1


# ---------------------------------------------------------------------------
# Gelman-Rubin R-hat
# ---------------------------------------------------------------------------

def compute_gelman_rubin(chain: np.ndarray) -> np.ndarray:
    """Compute the Gelman-Rubin R-hat statistic for each parameter.

    Uses the standard split-chain approach on the chain array.

    Parameters
    ----------
    chain : np.ndarray
        Shape ``(n_steps, n_walkers, n_params)`` -- full chain from emcee.

    Returns
    -------
    np.ndarray
        Shape ``(n_params,)`` -- R-hat per parameter.
    """
    n_steps, n_walkers, n_params = chain.shape

    if n_steps < 4:
        return np.full(n_params, np.nan)

    # Split each walker chain in half to double effective number of chains
    half = n_steps // 2
    chains = np.concatenate(
        [chain[:half, :, :], chain[half: 2 * half, :, :]],
        axis=1,
    )   # (half, 2*n_walkers, n_params)

    m = chains.shape[1]   # total number of split chains
    n = chains.shape[0]   # length of each split chain

    # Within-chain variance W
    chain_means = chains.mean(axis=0)           # (m, n_params)
    W = chains.var(axis=0, ddof=1).mean(axis=0)  # (n_params,)

    # Between-chain variance B
    grand_mean = chain_means.mean(axis=0)       # (n_params,)
    B = n * np.var(chain_means, axis=0, ddof=1)  # (n_params,)

    # Marginal posterior variance estimate
    var_hat = (n - 1) / n * W + B / n

    # R-hat
    rhat = np.sqrt(var_hat / np.where(W == 0, 1e-10, W))
    return rhat


# ---------------------------------------------------------------------------
# Autocorrelation time
# ---------------------------------------------------------------------------

def compute_autocorr_times(
    sampler: object,
    param_names: list[str],
) -> dict[str, float]:
    """Compute integrated autocorrelation time per parameter.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        Sampler object after production run.
    param_names : list[str]
        Parameter names.

    Returns
    -------
    dict[str, float]
        Autocorrelation time per parameter name.
    """
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        return {name: float(tau[i]) for i, name in enumerate(param_names)}
    except Exception as exc:
        logger.warning("Autocorrelation time computation failed: %s", exc)
        return {name: float("nan") for name in param_names}


# ---------------------------------------------------------------------------
# Trace plots
# ---------------------------------------------------------------------------

def plot_traces(
    chain: np.ndarray,
    param_names: list[str],
    output_path: Path,
    tic_id: str = "unknown",
) -> None:
    """Plot per-parameter trace plots (all walkers overlaid).

    Parameters
    ----------
    chain : np.ndarray
        Shape ``(n_steps, n_walkers, n_params)``.
    param_names : list[str]
    output_path : Path
    tic_id : str
    """
    n_steps, n_walkers, n_params = chain.shape
    plt.style.use("dark_background")

    n_cols = 2
    n_rows = int(np.ceil(n_params / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 2.5 * n_rows))
    axes = axes.flatten()

    for i, name in enumerate(param_names):
        ax = axes[i]
        # Plot every walker
        for j in range(min(n_walkers, 64)):   # cap at 64 for readability
            ax.plot(chain[:, j, i], alpha=0.3, lw=0.5, color="#4ECDC4")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Step", fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide extra axes
    for ax in axes[n_params:]:
        ax.set_visible(False)

    fig.suptitle(f"MCMC Traces — TIC {tic_id}", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Trace plots saved to %s", output_path)


# ---------------------------------------------------------------------------
# Autocorrelation plots
# ---------------------------------------------------------------------------

def plot_autocorr(
    sampler: object,
    param_names: list[str],
    output_path: Path,
    tic_id: str = "unknown",
) -> None:
    """Plot integrated autocorrelation time estimates vs. step.

    Uses emcee's windowed autocorrelation estimator evaluated at multiple
    chain lengths to show convergence of tau estimates.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
    param_names : list[str]
    output_path : Path
    tic_id : str
    """
    try:
        import emcee
        chain = sampler.get_chain(flat=False)  # (n_steps, n_walkers, n_params)
        n_steps = chain.shape[0]

        step_indices = np.exp(
            np.linspace(np.log(10), np.log(n_steps), 30)
        ).astype(int)
        step_indices = np.unique(np.clip(step_indices, 10, n_steps))

        tau_estimates = np.zeros((len(step_indices), len(param_names)))
        for k, n in enumerate(step_indices):
            try:
                tau_estimates[k] = emcee.autocorr.integrated_time(
                    chain[:n].mean(axis=1), tol=0, quiet=True
                )
            except Exception:
                tau_estimates[k] = np.nan

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(param_names)))

        for i, (name, color) in enumerate(zip(param_names, colors)):
            ax.plot(step_indices, tau_estimates[:, i], label=name,
                    color=color, lw=1.5)

        ax.plot(step_indices, step_indices / 50, "w--", lw=1, label="N/50 reference")
        ax.set(
            xlabel="Chain length (steps)",
            ylabel="Integrated autocorr. time τ",
            title=f"Autocorrelation Time — TIC {tic_id}",
            xscale="log",
            yscale="log",
        )
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("Autocorrelation plot saved to %s", output_path)
    except Exception as exc:
        logger.warning("Could not generate autocorrelation plot: %s", exc)


# ---------------------------------------------------------------------------
# Corner plot
# ---------------------------------------------------------------------------

def plot_corner(
    flat_samples: np.ndarray,
    param_names: list[str],
    output_path: Path,
    tic_id: str = "unknown",
) -> None:
    """Generate a corner plot of the posterior samples.

    Parameters
    ----------
    flat_samples : np.ndarray
        Shape ``(N_flat, N_params)``.
    param_names : list[str]
    output_path : Path
    tic_id : str
    """
    if not CORNER_AVAILABLE:
        logger.warning("corner package not available; skipping corner plot.")
        return

    try:
        plt.style.use("dark_background")
        fig = corner.corner(
            flat_samples,
            labels=param_names,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 9},
            label_kwargs={"fontsize": 9},
            title_fmt=".3f",
            color="#4ECDC4",
            truth_color="#FF6B6B",
        )
        fig.suptitle(f"Posterior Corner Plot — TIC {tic_id}", y=1.01, fontsize=12)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        logger.info("Corner plot saved to %s", output_path)
    except Exception as exc:
        logger.warning("Corner plot failed: %s", exc)


# ---------------------------------------------------------------------------
# Main diagnose function
# ---------------------------------------------------------------------------

def diagnose(
    sampler: object,
    param_names: list[str],
    tic_id: str,
    flat_samples: np.ndarray,
    reports_dir: str | Path = "reports/mcmc_diagnostics",
) -> dict:
    """Run full MCMC convergence diagnostics.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        Sampler object after production run (``reset()`` *not* called).
    param_names : list[str]
        Parameter names (length must match N_PARAMS = 9).
    tic_id : str
        Target identifier used for output file names.
    flat_samples : np.ndarray
        Shape ``(N_flat, N_params)`` -- flat posterior samples.
    reports_dir : str or Path
        Directory to save diagnostic plots.

    Returns
    -------
    dict
        Convergence diagnostics with keys:

        ``tau``           : dict[str, float] -- autocorrelation times
        ``ess``           : dict[str, float] -- effective sample sizes
        ``rhat``          : dict[str, float] -- Gelman-Rubin R-hat
        ``converged``     : bool -- True if all R-hat < 1.1
        ``n_flat_samples``: int
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    n_flat = int(flat_samples.shape[0])

    # ------------------------------------------------------------------
    # Get full chain for diagnostic use
    # ------------------------------------------------------------------
    try:
        chain = sampler.get_chain(flat=False)   # (n_steps, n_walkers, n_params)
    except Exception as exc:
        logger.warning("Could not retrieve chain from sampler: %s", exc)
        chain = flat_samples[np.newaxis, :, :]   # degenerate fallback

    n_steps = chain.shape[0]

    # ------------------------------------------------------------------
    # 1. Autocorrelation times
    # ------------------------------------------------------------------
    tau_dict = compute_autocorr_times(sampler, param_names)

    # ------------------------------------------------------------------
    # 2. Effective sample size
    # ------------------------------------------------------------------
    ess_dict: dict[str, float] = {}
    for name in param_names:
        tau = tau_dict.get(name, float("nan"))
        ess = n_steps / tau if np.isfinite(tau) and tau > 0 else float("nan")
        ess_dict[name] = float(ess)

    # ------------------------------------------------------------------
    # 3. Gelman-Rubin R-hat
    # ------------------------------------------------------------------
    rhat_arr = compute_gelman_rubin(chain)
    rhat_dict = {name: float(rhat_arr[i]) for i, name in enumerate(param_names)}
    all_converged = all(
        np.isfinite(v) and v < RHAT_THRESHOLD
        for v in rhat_dict.values()
    )

    # ------------------------------------------------------------------
    # 4. Print summary
    # ------------------------------------------------------------------
    print(f"\n=== Convergence Diagnostics for TIC {tic_id} ===")
    print(f"{'Parameter':<20} {'tau':>8} {'ESS':>10} {'R-hat':>8} {'OK?':>6}")
    print("-" * 60)
    for name in param_names:
        tau = tau_dict[name]
        ess = ess_dict[name]
        rh = rhat_dict[name]
        ok = "YES" if (np.isfinite(rh) and rh < RHAT_THRESHOLD) else "WARN"
        tau_str = f"{tau:.1f}" if np.isfinite(tau) else "nan"
        ess_str = f"{ess:.0f}" if np.isfinite(ess) else "nan"
        rh_str = f"{rh:.4f}" if np.isfinite(rh) else "nan"
        print(f"  {name:<18} {tau_str:>8} {ess_str:>10} {rh_str:>8} {ok:>6}")
    print("-" * 60)
    status = "CONVERGED" if all_converged else "NOT CONVERGED (R-hat >= 1.1)"
    print(f"  Overall: {status}\n")

    if not all_converged:
        bad = [n for n, v in rhat_dict.items() if not np.isfinite(v) or v >= RHAT_THRESHOLD]
        logger.warning(
            "TIC %s: MCMC not converged for parameters: %s", tic_id, bad
        )

    # ------------------------------------------------------------------
    # 5. Plots
    # ------------------------------------------------------------------
    plot_traces(
        chain, param_names,
        reports_dir / f"{tic_id}_traces.png",
        tic_id=tic_id,
    )
    plot_autocorr(
        sampler, param_names,
        reports_dir / f"{tic_id}_autocorr.png",
        tic_id=tic_id,
    )
    plot_corner(
        flat_samples, param_names,
        reports_dir / f"{tic_id}_corner.png",
        tic_id=tic_id,
    )

    # ------------------------------------------------------------------
    # 6. Return summary dict
    # ------------------------------------------------------------------
    return {
        "tic_id": tic_id,
        "tau": tau_dict,
        "ess": ess_dict,
        "rhat": rhat_dict,
        "converged": bool(all_converged),
        "n_flat_samples": n_flat,
        "n_steps": n_steps,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MCMC convergence diagnostics on saved samples."
    )
    parser.add_argument("--samples_path", type=str, required=True,
                        help="Path to {tic_id}_mcmc_samples.npy file.")
    parser.add_argument("--tic_id", type=str, default="unknown",
                        help="TIC identifier.")
    parser.add_argument("--reports_dir", type=str,
                        default="reports/mcmc_diagnostics",
                        help="Output directory for diagnostic plots.")
    args = parser.parse_args()

    from models.transit_gp_model import PARAM_NAMES

    flat_samples = np.load(args.samples_path)
    print(f"Loaded flat_samples: shape={flat_samples.shape}")

    # Reconstruct a minimal diagnostic from flat samples only
    # (without a live sampler, we skip tau/ESS and produce corner plot only)
    n_flat, n_params = flat_samples.shape
    param_names = PARAM_NAMES[:n_params]

    # Generate corner plot directly
    plot_corner(
        flat_samples,
        param_names,
        Path(args.reports_dir) / f"{args.tic_id}_corner.png",
        tic_id=args.tic_id,
    )

    # Print basic percentile summary
    print(f"\n=== Posterior Summary for TIC {args.tic_id} ===")
    print(f"{'Parameter':<20} {'16%':>10} {'50%':>10} {'84%':>10}")
    print("-" * 54)
    for i, name in enumerate(param_names):
        lo, med, hi = np.percentile(flat_samples[:, i], [16, 50, 84])
        print(f"  {name:<18} {lo:>10.4f} {med:>10.4f} {hi:>10.4f}")
