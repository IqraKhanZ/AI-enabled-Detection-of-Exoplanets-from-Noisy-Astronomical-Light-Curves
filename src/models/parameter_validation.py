"""
src/models/parameter_validation.py
==================================
Validates estimated parameters (period, depth, duration) against reference values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.config import get, project_root
from utils.logger import get_logger

logger = get_logger(__name__)

def run_validation(
    results_csv: str | Path,
    reference_csv: str | Path | None = None,
    output_dir: str | Path | None = None
) -> dict:
    """Compare pipeline results against known confirmed planetary parameters."""
    results_csv = Path(results_csv)
    output_dir = Path(output_dir) if output_dir else results_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not results_csv.exists():
        logger.error("Pipeline results CSV not found: %s", results_csv)
        return {}
        
    df_res = pd.read_csv(results_csv)
    
    # We filter to PLANET predictions
    df_planets = df_res[df_res["predicted_label_name"] == "PLANET"].copy()
    if df_planets.empty:
        logger.warning("No PLANET predictions found in pipeline results.")
        return {}
        
    # Load reference parameters (TOI table or NASA Exoplanet Archive)
    if reference_csv and Path(reference_csv).exists():
        df_ref = pd.read_csv(reference_csv)
    else:
        # Try to load from raw labels as default fallback
        ref_path = project_root() / "data/raw/labels/toi_labels.csv"
        if ref_path.exists():
            df_ref = pd.read_csv(ref_path)
        else:
            logger.error("No reference/labels catalog found to validate parameters.")
            return {}

    # Merge on tic_id
    # Reference columns expected: period_days, depth_ppm, duration_hrs
    # Result columns: period_days, depth_ppm, duration_hrs
    df_merged = pd.merge(
        df_planets,
        df_ref,
        on="tic_id",
        suffixes=("_est", "_ref")
    )
    
    if df_merged.empty:
        logger.warning("No overlapping targets found between results and reference catalog.")
        return {}

    metrics = {}
    plt.style.use("dark_background")
    
    for param, unit in [("period_days", "days"), ("depth_ppm", "ppm"), ("duration_hrs", "hours")]:
        ref_col = f"{param}_ref" if f"{param}_ref" in df_merged.columns else param
        # Check if column name in reference table has different name
        if ref_col not in df_merged.columns:
            # try finding direct name mapping
            if param in df_ref.columns:
                df_merged[ref_col] = df_merged["tic_id"].map(dict(zip(df_ref["tic_id"], df_ref[param])))
                
        if ref_col not in df_merged.columns:
            continue
            
        est_vals = df_merged[f"{param}_est"].values
        ref_vals = df_merged[ref_col].values
        
        # Filter NaNs
        valid = np.isfinite(est_vals) & np.isfinite(ref_vals)
        if not valid.any():
            continue
            
        est_vals = est_vals[valid]
        ref_vals = ref_vals[valid]
        
        errors = est_vals - ref_vals
        abs_errors = np.abs(errors)
        frac_errors = abs_errors / np.maximum(ref_vals, 1e-6)
        
        mae = float(np.median(abs_errors))
        rmse = float(np.sqrt(np.mean(errors**2)))
        med_frac_err = float(np.median(frac_errors))
        
        # 1-sigma recovery rate (within 10% error)
        recovery_1s = float(np.mean(frac_errors <= 0.10))
        
        metrics[param] = {
            "MAE": mae,
            "RMSE": rmse,
            "Median_Fractional_Error": med_frac_err,
            "Recovery_Rate_10pct": recovery_1s,
            "Count": int(len(est_vals))
        }
        
        # Plot Scatter
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(ref_vals, est_vals, alpha=0.7, color="#1f77b4", edgecolors="w")
        
        # Diagonal reference line
        lims = [
            min(np.min(ref_vals), np.min(est_vals)),
            max(np.max(ref_vals), np.max(est_vals))
        ]
        ax.plot(lims, lims, "r--", alpha=0.7, label="y=x")
        
        ax.set_xlabel(f"Reference {param} ({unit})")
        ax.set_ylabel(f"Estimated {param} ({unit})")
        ax.set_title(f"Parameter Validation: {param}")
        ax.legend()
        plt.tight_layout()
        
        plot_path = output_dir / f"validation_scatter_{param}.png"
        fig.savefig(plot_path)
        plt.close(fig)
        
    metrics_path = output_dir / "parameter_validation_report.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
        
    logger.info("Parameter validation complete. Report saved to %s", metrics_path)
    print(json.dumps(metrics, indent=2))
    
    return metrics

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate estimated transit parameters.")
    parser.add_argument("--results-csv", type=str, default=None, help="Path to pipeline_results.csv.")
    parser.add_argument("--reference-csv", type=str, default=None, help="Path to reference parameters CSV.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save validation plots.")
    args = parser.parse_args()
    
    root = project_root()
    results_csv = args.results_csv or root / get("paths.outputs", "outputs") / "pipeline_results.csv"
    reference_csv = args.reference_csv
    output_dir = args.output_dir or root / get("paths.reports", "reports")
    
    run_validation(results_csv, reference_csv, output_dir)

if __name__ == "__main__":
    main()
