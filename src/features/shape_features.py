"""
src/features/shape_features.py
==============================
Topological/shape-based transit dip feature extraction.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional
import numpy as np
from scipy.optimize import curve_fit

from utils.logger import get_logger

logger = get_logger(__name__)

def _trapezoid_model(x: np.ndarray, depth: float, duration: float, ingress: float, t0: float) -> np.ndarray:
    """Simple trapezoidal transit model.
    
    Parameters
    ----------
    x : np.ndarray
        Phases (or times) of the observations.
    depth : float
        Transit depth (fractional or normalized).
    duration : float
        Total transit duration (width at top of trapezoid).
    ingress : float
        Ingress/egress duration (slanted part width).
    t0 : float
        Mid-transit epoch.
    """
    y = np.ones_like(x)
    half_dur = duration / 2.0
    
    # Left flat out of transit
    # Left ingress: from t0 - half_dur to t0 - half_dur + ingress
    # Flat bottom: from t0 - half_dur + ingress to t0 + half_dur - ingress
    # Right egress: from t0 + half_dur - ingress to t0 + half_dur
    # Right flat out of transit
    
    # Phase distance from center
    dist = np.abs(x - t0)
    
    # Out of transit
    y[dist >= half_dur] = 1.0
    
    # Ingress/Egress region
    in_eg_mask = (dist < half_dur) & (dist >= (half_dur - ingress))
    if ingress > 0:
        y[in_eg_mask] = 1.0 - depth * (half_dur - dist[in_eg_mask]) / ingress
        
    # Flat bottom
    y[dist < (half_dur - ingress)] = 1.0 - depth
    
    return y

def extract_shape_features(
    phase_global: np.ndarray,
    phase_local: Optional[np.ndarray] = None,
    period_days: float = 1.0,
    depth_ppm: float = 1000.0,
    duration_hrs: float = 1.0
) -> np.ndarray:
    """Extract shape-based and topological features from the phase-folded light curve views.
    
    Returns a 1D numpy array of shape features.
    """
    if phase_local is None:
        # 1-argument signature fallback: phase_global is actually phase_local
        phase_local = np.asarray(phase_global, dtype=np.float32)
        phase_global = phase_local
    else:
        phase_global = np.asarray(phase_global, dtype=np.float32)
        phase_local = np.asarray(phase_local, dtype=np.float32)
    
    # Default values for safety
    v_shape_index = 0.0
    transit_asymmetry = 0.0
    sharpness_ratio = 1.0
    depth_uniformity = 0.0
    secondary_peak_ratio = 0.0
    
    n_bins_local = len(phase_local)
    
    # Ensure local view is valid
    if n_bins_local > 4 and np.isfinite(phase_local).all():
        try:
            # 1. Transit asymmetry: compare left half and right half of the local view
            mid = n_bins_local // 2
            left_half = phase_local[:mid]
            right_half = phase_local[mid:]
            left_sum = np.sum(1.0 - left_half)
            right_sum = np.sum(1.0 - right_half)
            total_sum = left_sum + right_sum
            if total_sum > 0:
                transit_asymmetry = float((left_sum - right_sum) / total_sum)
        except Exception as exc:
            logger.debug("Asymmetry computation failed: %s", exc)

        try:
            # 2. Fit U-shape (trapezoid) vs V-shape (triangle)
            # Create a simple phase grid for local view [-0.5, 0.5] relative to mid-transit
            x_local = np.linspace(-0.5, 0.5, n_bins_local)
            y_local = phase_local.copy()
            
            # Normalize y_local to baseline 1.0 and depth matching depth_ppm
            depth_frac = (depth_ppm / 1e6) if np.isfinite(depth_ppm) else (1.0 - np.min(y_local))
            
            # Fit V-shape (ingress = duration/2)
            # fit params: [depth, duration, ingress, t0]
            # V-shape: ingress is equal to half-duration
            p0_v = [depth_frac, 0.5, 0.25, 0.0]
            bounds_v = ([0.0, 0.0, 0.249, -0.1], [1.0, 1.0, 0.251, 0.1]) # force ingress ~ 0.25
            
            # U-shape: ingress can be small
            p0_u = [depth_frac, 0.5, 0.05, 0.0]
            bounds_u = ([0.0, 0.0, 0.0, -0.1], [1.0, 1.0, 0.25, 0.1])
            
            try:
                popt_v, _ = curve_fit(_trapezoid_model, x_local, y_local, p0=p0_v, bounds=bounds_v, maxfev=500)
                fit_v = _trapezoid_model(x_local, *popt_v)
                rmse_v = np.sqrt(np.mean((y_local - fit_v)**2))
            except Exception:
                rmse_v = 1.0
                
            try:
                popt_u, _ = curve_fit(_trapezoid_model, x_local, y_local, p0=p0_u, bounds=bounds_u, maxfev=500)
                fit_u = _trapezoid_model(x_local, *popt_u)
                rmse_u = np.sqrt(np.mean((y_local - fit_u)**2))
                
                # Ingress ratio
                ingress_fit = popt_u[2]
                duration_fit = popt_u[1]
                if duration_fit > 0:
                    sharpness_ratio = float(ingress_fit / duration_fit)
            except Exception:
                rmse_u = 1.0
            
            # v_shape_index: lower U-shape error relative to V-shape error implies U-shape (closer to 0)
            # V-shape index = rmse_u / (rmse_v + 1e-6)
            if rmse_v > 0:
                v_shape_index = float(rmse_u / rmse_v)
                
        except Exception as exc:
            logger.debug("Trapezoid fitting failed: %s", exc)

        try:
            # 3. Depth uniformity: variance inside the transit bottom
            # Bottom is defined as the central 20% of the local view
            start_idx = int(n_bins_local * 0.4)
            end_idx = int(n_bins_local * 0.6)
            bottom_flux = phase_local[start_idx:end_idx]
            if len(bottom_flux) > 0:
                depth_uniformity = float(np.var(bottom_flux))
        except Exception as exc:
            logger.debug("Uniformity computation failed: %s", exc)
            
    # 4. Secondary peak ratio: look for the secondary eclipse in global view
    # Exclude the primary transit region (phases -0.1 to 0.1)
    n_bins_global = len(phase_global)
    if n_bins_global > 20:
        try:
            # Mask out the primary transit
            mid_g = n_bins_global // 2
            exclude_width = int(n_bins_global * 0.1)
            mask = np.ones(n_bins_global, dtype=bool)
            mask[max(0, mid_g - exclude_width):min(n_bins_global, mid_g + exclude_width)] = False
            
            non_transit_flux = phase_global[mask]
            if len(non_transit_flux) > 0:
                # Find the deepest dip in the non-transit region (around phase 0.5)
                # Max depth is 1.0 - min_flux
                max_dip = float(1.0 - np.min(non_transit_flux))
                primary_depth = (depth_ppm / 1e6) if np.isfinite(depth_ppm) and depth_ppm > 0 else (1.0 - np.min(phase_local))
                if primary_depth > 0:
                    secondary_peak_ratio = float(max_dip / primary_depth)
        except Exception as exc:
            logger.debug("Secondary peak ratio failed: %s", exc)

    features = np.array([
        v_shape_index,
        transit_asymmetry,
        sharpness_ratio,
        depth_uniformity,
        secondary_peak_ratio,
        float(np.nanmean(phase_local)),
        float(np.nanstd(phase_local))
    ], dtype=np.float32)
    
    # Fill any NaNs with 0.0
    features = np.where(np.isfinite(features), features, 0.0)
    return features

# Alias to match run_pipeline.py
extract = extract_shape_features
