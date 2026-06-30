"""
src/scoring/confidence.py
=========================
Confidence score integration module.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional
import numpy as np

from utils.config import get
from utils.logger import get_logger

logger = get_logger(__name__)

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def compute_confidence(
    class_probs: np.ndarray | list[float],
    snr: float,
    fap: float,
    config_path: Optional[str] = None
) -> dict[str, Any]:
    """Combine softmax class probabilities, SNR, and FAP into a single confidence score.
    
    Parameters
    ----------
    class_probs : np.ndarray or list of float
        Classifier probabilities for the 4 classes: [PLANET, EB, BLEND, NOISE].
    snr : float
        Signal-to-noise ratio.
    fap : float
        False alarm probability.
    config_path : str, optional
        Path to config file.
        
    Returns
    -------
    dict
        Confidence summary dictionary.
    """
    probs = np.asarray(class_probs)
    planet_prob = float(probs[0]) if len(probs) > 0 else 0.0
    
    # Load weights from configuration
    w_softmax = get("scoring.confidence_weights.softmax_planet_prob", 0.5, config_path)
    w_snr = get("scoring.confidence_weights.snr_score", 0.3, config_path)
    w_fap = get("scoring.confidence_weights.fap_score", 0.2, config_path)
    
    # Normalize weights
    total_w = w_softmax + w_snr + w_fap
    if total_w > 0:
        w_softmax /= total_w
        w_snr /= total_w
        w_fap /= total_w
        
    # 1. Softmax probability component
    softmax_comp = planet_prob
    
    # 2. SNR component: sigmoid centered at SNR=7.0 with width parameter 3.0
    snr_comp = float(_sigmoid((snr - 7.0) / 3.0))
    
    # 3. FAP component: inverted FAP (1 - FAP)
    fap_comp = float(1.0 - min(max(fap, 0.0), 1.0))
    
    # Combine components
    pipeline_confidence = float(
        w_softmax * softmax_comp +
        w_snr * snr_comp +
        w_fap * fap_comp
    )
    
    # Determine confidence level
    if pipeline_confidence >= 0.7:
        level = "HIGH"
    elif pipeline_confidence >= 0.4:
        level = "MEDIUM"
    else:
        level = "LOW"
        
    return {
        "pipeline_confidence": pipeline_confidence,
        "softmax_component": softmax_comp,
        "snr_component": snr_comp,
        "fap_component": fap_comp,
        "confidence_level": level
    }
