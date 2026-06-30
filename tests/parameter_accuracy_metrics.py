"""
tests/parameter_accuracy_metrics.py
===================================
Wrapper script to calculate and visualize final parameter estimation accuracy metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_SRC_PKG_DIR = _SRC_DIR / "src"
if str(_SRC_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_PKG_DIR))

from models.parameter_validation import run_validation
from utils.config import get, project_root

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate final parameter estimation accuracy metrics.")
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
