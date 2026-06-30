"""
tests/classification_metrics.py
================================
Computes and visualises classification metrics (precision, recall, F1, confusion matrix)
from test predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, matthews_corrcoef

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_SRC_PKG_DIR = _SRC_DIR / "src"
if str(_SRC_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_PKG_DIR))

from utils.config import get, project_root
from utils.logger import get_logger

logger = get_logger(__name__)

CLASS_NAMES = ["PLANET", "ECLIPSING_BINARY", "BLEND", "NOISE"]

def compute_metrics(
    predictions_csv: str | Path,
    output_dir: str | Path | None = None
) -> None:
    """Compute classification statistics and generate plots from predictions."""
    predictions_csv = Path(predictions_csv)
    output_dir = Path(output_dir) if output_dir else predictions_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not predictions_csv.exists():
        logger.error("Predictions file not found: %s", predictions_csv)
        sys.exit(1)
        
    df = pd.read_csv(predictions_csv)
    
    # Check that both actual labels and predicted labels are available
    y_true_col = "label" if "label" in df.columns else "true_label"
    if y_true_col not in df.columns:
        logger.error("CSV must contain a column for true labels ('label' or 'true_label')")
        sys.exit(1)
        
    y_pred_col = "predicted_label"
    if y_pred_col not in df.columns:
        logger.error("CSV must contain a column for predictions ('predicted_label')")
        sys.exit(1)
        
    y_true = df[y_true_col].astype(int).values
    y_pred = df[y_pred_col].astype(int).values
    
    # Filter to valid classes [0, 1, 2, 3]
    valid_mask = np.isin(y_true, [0, 1, 2, 3]) & np.isin(y_pred, [0, 1, 2, 3])
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    if len(y_true) == 0:
        logger.error("No valid classification samples found.")
        sys.exit(1)
        
    # Standard classification report
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES[:len(np.unique(y_true))], output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Kappa & MCC
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    summary = {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "cohen_kappa": float(kappa),
        "matthews_corrcoef": float(mcc)
    }
    
    report_json_path = output_dir / "final_classification_report.json"
    with open(report_json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        
    logger.info("Saved metrics JSON to %s", report_json_path)
    
    # Plot Confusion Matrix
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Normalize CM
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm) # handle divisions by zero
    
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=CLASS_NAMES[:cm.shape[1]],
        yticklabels=CLASS_NAMES[:cm.shape[0]],
        title="Normalized Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label"
    )
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    fmt = ".2f"
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm_norm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black"
            )
            
    fig.tight_layout()
    plot_path = output_dir / "confusion_matrix.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix plot to %s", plot_path)
    
    # Print print-friendly report
    print("\n=== Classification Performance Report ===")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES[:len(np.unique(y_true))]))
    print(f"Cohen's Kappa          : {kappa:.4f}")
    print(f"Matthews Corr Coeff    : {mcc:.4f}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate classification performance metrics.")
    parser.add_argument("--predictions-csv", type=str, default=None, help="Path to test_predictions.csv.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save metrics reports and plots.")
    args = parser.parse_args()
    
    root = project_root()
    predictions_csv = args.predictions_csv or root / get("paths.outputs", "outputs") / "test_predictions.csv"
    output_dir = args.output_dir or root / get("paths.reports", "reports")
    
    compute_metrics(predictions_csv, output_dir)

if __name__ == "__main__":
    main()
