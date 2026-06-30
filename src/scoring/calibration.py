"""
src/scoring/calibration.py
==========================
Calibrates confidence scores using known true and false positive targets.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.config import get, project_root
from utils.logger import get_logger

logger = get_logger(__name__)

def calibrate_scores(
    predictions_csv: str | Path,
    output_path: str | Path | None = None
) -> IsotonicRegression:
    """Fit isotonic regression model on test/val set planet predictions and save it."""
    predictions_csv = Path(predictions_csv)
    output_path = Path(output_path) if output_path else project_root() / "checkpoints/calibration_model.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not predictions_csv.exists():
        logger.error("Predictions CSV not found: %s", predictions_csv)
        sys.exit(1)
        
    df = pd.read_csv(predictions_csv)
    
    # We calibrate based on predicted planet probabilities (planet_prob) vs actual labels
    # True positive: label == 0 (PLANET)
    y_true_col = "label" if "label" in df.columns else "true_label"
    if y_true_col not in df.columns:
        logger.error("CSV must contain a column for true labels ('label' or 'true_label')")
        sys.exit(1)
        
    y_prob_col = "planet_prob"
    if y_prob_col not in df.columns:
        logger.error("CSV must contain a column for planet probability ('planet_prob')")
        sys.exit(1)
        
    # True positives: 1 if PLANET, 0 if EB, BLEND or NOISE
    y_true = (df[y_true_col].values == 0).astype(int)
    y_prob = df[y_prob_col].values
    
    valid = np.isfinite(y_prob) & np.isfinite(y_true)
    y_true = y_true[valid]
    y_prob = y_prob[valid]
    
    if len(y_true) == 0:
        logger.error("No valid samples for calibration.")
        sys.exit(1)
        
    logger.info("Fitting Isotonic Regression calibration model on %d samples...", len(y_true))
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(y_prob, y_true)
    
    # Save the calibration model
    with open(output_path, "wb") as fh:
        pickle.dump(ir, fh)
    logger.info("Saved calibration model to %s", output_path)
    
    # Plot calibration curve
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Bin predictions to plot calibration curve
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    
    # Calibrated probabilities
    y_calibrated = ir.predict(y_prob)
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, y_calibrated, n_bins=10)
    
    ax.plot(prob_pred, prob_true, "s-", color="red", label="Uncalibrated")
    ax.plot(prob_pred_cal, prob_true_cal, "o-", color="green", label="Calibrated")
    ax.plot([0, 1], [0, 1], "w--", label="Perfect Calibration")
    
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    
    plot_path = output_path.parent / "calibration_curve.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info("Saved calibration reliability plot to %s", plot_path)
    
    return ir

def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate pipeline confidence scores.")
    parser.add_argument("--predictions-csv", type=str, required=True, help="Predictions file path.")
    parser.add_argument("--output-path", type=str, default=None, help="Output path for pickle file.")
    args = parser.parse_args()
    
    calibrate_scores(args.predictions_csv, args.output_path)

if __name__ == "__main__":
    main()
