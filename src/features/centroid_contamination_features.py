"""
src/features/centroid_contamination_features.py
==============================================
Consolidates centroid shift and Gaia DR3 contamination features into a single dict/array.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

def extract_centroid_contamination_features(
    centroid_result: Optional[Any] = None,
    contamination_result: Optional[Any] = None
) -> dict[str, Any]:
    """Consolidate centroid and contamination metrics into a feature dictionary."""
    features = {
        "centroid_shift_arcsec": 0.0,
        "centroid_shift_pixels": 0.0,
        "has_centroid_data": False,
        "contamination_ratio": 0.0,
        "neighbor_count": 0,
        "is_contaminated": False,
        "has_gaia_data": False
    }
    
    if centroid_result is not None:
        try:
            if isinstance(centroid_result, dict):
                features["centroid_shift_arcsec"] = float(centroid_result.get("centroid_shift_arcsec", 0.0))
                features["centroid_shift_pixels"] = float(centroid_result.get("centroid_shift_pixels", 0.0))
                features["has_centroid_data"] = True
            else:
                features["centroid_shift_arcsec"] = float(getattr(centroid_result, "centroid_shift_arcsec", 0.0))
                features["centroid_shift_pixels"] = float(getattr(centroid_result, "centroid_shift_pixels", 0.0))
                features["has_centroid_data"] = True
        except Exception as exc:
            logger.debug("Failed to extract centroid features: %s", exc)
            
    if contamination_result is not None:
        try:
            if isinstance(contamination_result, dict):
                features["contamination_ratio"] = float(contamination_result.get("contamination_ratio", 0.0))
                features["neighbor_count"] = int(contamination_result.get("neighbor_count", 0))
                features["is_contaminated"] = bool(contamination_result.get("is_contaminated", False))
                features["has_gaia_data"] = True
            else:
                features["contamination_ratio"] = float(getattr(contamination_result, "contamination_ratio", 0.0))
                features["neighbor_count"] = int(getattr(contamination_result, "neighbor_count", 0))
                features["is_contaminated"] = bool(getattr(contamination_result, "is_contaminated", False))
                features["has_gaia_data"] = True
        except Exception as exc:
            logger.debug("Failed to extract contamination features: %s", exc)
            
    return features

# Alias
extract = extract_centroid_contamination_features
