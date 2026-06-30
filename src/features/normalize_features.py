"""
src/features/normalize_features.py
==================================
Handles scaling and normalization of exoplanet pipeline feature vectors.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.config import get, project_root
from utils.logger import get_logger

logger = get_logger(__name__)

def fit_scaler(
    train_features: list[dict] | np.ndarray,
    scaler_type: str = "robust",
    output_path: str | Path | None = None
) -> tuple[Any, list[str] | None]:
    """Fit a feature scaler on training features and save to disk."""
    if isinstance(train_features, list):
        df = pd.DataFrame(train_features)
        feature_names = list(df.columns)
        X = df.values
    else:
        X = train_features
        feature_names = None
        
    # Handle NaNs / Infinite values
    X = np.where(np.isfinite(X), X, np.nan)
    col_medians = np.nanmedian(X, axis=0)
    # Fill remaining NaNs with 0.0
    col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)
    for col_idx in range(X.shape[1]):
        nan_mask = np.isnan(X[:, col_idx])
        X[nan_mask, col_idx] = col_medians[col_idx]
        
    if scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
        
    scaler.fit(X)
    scaler.medians_ = col_medians # attach medians for inference imputation
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as fh:
            pickle.dump({"scaler": scaler, "feature_names": feature_names, "medians": col_medians}, fh)
        logger.info("Saved scaler to %s", output_path)
        
    return scaler, feature_names

def transform_features(
    features: list[dict] | np.ndarray,
    scaler: Any,
    feature_names: list[str] | None = None
) -> np.ndarray:
    """Transform features using a pre-fitted scaler."""
    if isinstance(features, list):
        df = pd.DataFrame(features)
        if feature_names:
            # Ensure columns are aligned
            df = df.reindex(columns=feature_names, fill_value=0.0)
        X = df.values
    else:
        X = np.asarray(features, dtype=np.float32)
        
    # Impute missing values using scaler's saved medians
    medians = getattr(scaler, "medians_", np.zeros(X.shape[1]))
    X = np.where(np.isfinite(X), X, np.nan)
    for col_idx in range(X.shape[1]):
        nan_mask = np.isnan(X[:, col_idx])
        med_val = medians[col_idx] if col_idx < len(medians) else 0.0
        X[nan_mask, col_idx] = med_val
        
    return scaler.transform(X)

def load_scaler(scaler_path: str | Path) -> tuple[Any, list[str] | None]:
    """Load pre-fitted scaler from disk."""
    scaler_path = Path(scaler_path)
    with open(scaler_path, "rb") as fh:
        data = pickle.load(fh)
    return data["scaler"], data.get("feature_names")

def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize features.")
    parser.add_argument("--input-csv", type=str, required=True, help="CSV containing training features.")
    parser.add_argument("--scaler-type", type=str, default="robust", choices=["robust", "standard", "minmax"])
    parser.add_argument("--output-path", type=str, default=None, help="Output path for serialized scaler.")
    args = parser.parse_args()
    
    root = project_root()
    output_path = args.output_path or root / get("paths.checkpoints", "checkpoints") / "feature_scaler.pkl"
    
    df = pd.read_csv(args.input_csv)
    # drop non-feature columns
    features_cols = [c for c in df.columns if c not in ["tic_id", "label", "split"]]
    train_feats = df[features_cols].values
    
    fit_scaler(train_feats, scaler_type=args.scaler_type, output_path=output_path)

if __name__ == "__main__":
    main()
