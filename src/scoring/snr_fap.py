"""
src/scoring/snr_fap.py
======================
SNR and False Alarm Probability calculation module.
"""

from __future__ import annotations

import logging
import numpy as np
from scipy.stats import norm

from utils.logger import get_logger

logger = get_logger(__name__)

def compute(
    flux_detrended: np.ndarray,
    period_days: float,
    depth_ppm: float,
    duration_hrs: float
) -> dict[str, float]:
    """Compute the signal-to-noise ratio and false alarm probability of a transit signal.
    
    Parameters
    ----------
    flux_detrended : np.ndarray
        Detrended, normalized flux array.
    period_days : float
        Best-fit orbital period.
    depth_ppm : float
        Best-fit depth in ppm.
    duration_hrs : float
        Best-fit duration in hours.
        
    Returns
    -------
    dict
        Dictionary containing 'snr' and 'fap'.
    """
    flux = np.asarray(flux_detrended)
    
    # Defaults
    snr = 0.0
    fap = 1.0
    
    if len(flux) == 0 or not np.isfinite(depth_ppm) or depth_ppm <= 0:
        return {"snr": snr, "fap": fap}
        
    try:
        # Estimate noise level in ppm
        noise_std_ppm = float(np.nanstd(flux) * 1e6)
        
        # Simple point estimate of SNR
        if noise_std_ppm > 0:
            snr = float(depth_ppm / noise_std_ppm)
            
            # Analytical FAP based on Gaussian tail probability
            # We treat the transit depth as a statistical outlier from the noise distribution
            # and compute the probability of observing such a dip in a random normal sample.
            # Z = snr
            fap = float(2.0 * (1.0 - norm.cdf(snr)))
            # Keep FAP between 0 and 1
            fap = max(min(fap, 1.0), 0.0)
            
    except Exception as exc:
        logger.debug("SNR/FAP calculation failed: %s", exc)
        
    return {"snr": snr, "fap": fap}
