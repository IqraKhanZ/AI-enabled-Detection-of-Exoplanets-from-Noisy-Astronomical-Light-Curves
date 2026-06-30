"""
src/features/validate_features.py
=================================
Validates feature distributions and checks for data leakage and outliers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.config import get, project_root
from utils.logger import get_logger

logger = get_logger(__name__)

def validate_features_data(
    features_csv: str | Path,
    output_dir: str | Path | None = None
) -> None:
    """Validate extracted features by plotting distributions and checking for data leakage."""
    features_csv = Path(features_csv)
    output_dir = Path(output_dir) if output_dir else project_root() / "reports/feature_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not features_csv.exists():
        logger.error("Features CSV file not found: %s", features_csv)
        sys.exit(1)
        
    df = pd.read_csv(features_csv)
    
    if "split" not in df.columns:
        logger.warning("No 'split' column found. Data leakage checks between train/test cannot be performed.")
        return
        
    # Split into train and test groups
    df_train = df[df["split"] == "train"]
    df_test = df[df["split"] == "test"]
    
    # Feature columns are those that are numeric and not metadata columns
    metadata_cols = ["tic_id", "label", "split"]
    feature_cols = [c for c in df.columns if c not in metadata_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    if not feature_cols:
        logger.error("No numeric feature columns found to validate.")
        return
        
    logger.info("Validating %d feature distributions...", len(feature_cols))
    
    plt.style.use("dark_background")
    leakage_warnings = []
    outlier_warnings = []
    
    # 1. KS Test for data leakage check
    for col in feature_cols:
        train_vals = df_train[col].dropna().values
        test_vals = df_test[col].dropna().values
        
        if len(train_vals) > 0 and len(test_vals) > 0:
            stat, p_val = ks_2samp(train_vals, test_vals)
            # If p-value is extremely low (<0.01), it means train and test distributions differ significantly
            # which could indicate a dataset shift or a leakage problem
            if p_val < 0.01:
                leakage_warnings.append(f"{col} (p-val={p_val:.2e})")
                
        # 2. Outlier detection (>5-sigma deviation)
        all_vals = df[col].dropna().values
        if len(all_vals) > 0:
            med = np.median(all_vals)
            mad = np.median(np.abs(all_vals - med))
            # Safe MAD division
            mad = max(mad, 1e-6)
            outliers = np.abs(all_vals - med) / mad > 5.0
            outlier_pct = float(np.mean(outliers)) * 100
            if outlier_pct > 5.0:
                outlier_warnings.append(f"{col} ({outlier_pct:.2f}% outliers)")
                
    # Log warnings
    if leakage_warnings:
        logger.warning("Potential dataset shift or leakage in features: %s", ", ".join(leakage_warnings))
    else:
        logger.info("No distribution shift / leakage detected between train and test splits.")
        
    if outlier_warnings:
        logger.warning("Features with high outlier fraction (>5% beyond 5-MAD): %s", ", ".join(outlier_warnings))
        
    # 3. Create pairwise correlation heatmap
    try:
        corr_matrix = df[feature_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set_xticks(np.arange(len(feature_cols)))
        ax.set_yticks(np.arange(len(feature_cols)))
        ax.set_xticklabels(feature_cols, rotation=90, fontsize=8)
        ax.set_yticklabels(feature_cols, fontsize=8)
        ax.set_title("Feature Pairwise Correlation Heatmap")
        plt.tight_layout()
        
        heatmap_path = output_dir / "feature_correlation_heatmap.png"
        fig.savefig(heatmap_path, dpi=150)
        plt.close(fig)
        logger.info("Saved correlation heatmap plot to %s", heatmap_path)
    except Exception as exc:
        logger.error("Failed to generate correlation heatmap: %s", exc)

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate feature distributions.")
    parser.add_argument("--features-csv", type=str, required=True, help="Path to features CSV.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save validation plots.")
    args = parser.parse_args()
    
    validate_features_data(args.features_csv, args.output_dir)

if __name__ == "__main__":
    main()
