"""
src/scoring/mcmc_sampler.py
===========================
Wrapper to expose emcee MCMC sampler from models.mcmc_sampler.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional
import numpy as np

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from models.transit_gp_model import TransitGPModel
from models.mcmc_sampler import run_mcmc

def run(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period_init: float,
    depth_init: float,
    duration_init: float,
    config_path: Optional[str] = None,
) -> dict[str, Any]:
    """Exposes emcee MCMC sampling as a run function for pipeline integration."""
    model = TransitGPModel(time, flux, flux_err)
    
    # Guess t0 at the flux minimum
    t0_init = float(time[np.argmin(flux)]) if len(time) > 0 else 0.0
    
    bls_result = {
        "period": period_init,
        "t0": t0_init,
        "depth": depth_init,
        "duration": duration_init,
    }
    
    # Load config dict
    import yaml
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
    else:
        # Defaults
        config = {
            "mcmc": {
                "nwalkers": 32,
                "nsteps": 200,
                "nburn": 50,
            }
        }
        
    flat_samples, param_dict = run_mcmc(
        model=model,
        bls_result=bls_result,
        config=config,
        tic_id="mcmc_temp",
    )
    
    # Align parameter names to what run_pipeline.py expects
    mapped_dict = {
        "period_days": param_dict.get("period", period_init),
        "period_err": 0.5 * (param_dict.get("period_err_hi", 0.0) + param_dict.get("period_err_lo", 0.0)),
        "depth_ppm": param_dict.get("depth_ppm", depth_init * 1e6),
        "depth_err": 0.5 * (param_dict.get("depth_ppm_err_hi", 0.0) + param_dict.get("depth_ppm_err_lo", 0.0)),
        "duration_hrs": param_dict.get("duration_hrs", duration_init * 24.0),
        "duration_err": 0.5 * (param_dict.get("duration_hrs_err_hi", 0.0) + param_dict.get("duration_hrs_err_lo", 0.0)),
    }
    
    # Keep original keys as well
    mapped_dict.update(param_dict)
    return mapped_dict
