"""
tests/export_cases.py
=====================
Exports high-confidence and low-confidence classification cases for manual validation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_SRC_PKG_DIR = _SRC_DIR / "src"
if str(_SRC_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_PKG_DIR))

from utils.config import get, project_root
from utils.logger import get_logger

logger = get_logger(__name__)

def export_cases(
    pipeline_results_csv: str | Path,
    output_dir: str | Path | None = None,
    n_cases: int = 20
) -> None:
    """Filter and export the top and bottom classification confidence targets."""
    pipeline_results_csv = Path(pipeline_results_csv)
    output_dir = Path(output_dir) if output_dir else pipeline_results_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not pipeline_results_csv.exists():
        logger.error("Pipeline results file not found: %s", pipeline_results_csv)
        sys.exit(1)
        
    df = pd.read_csv(pipeline_results_csv)
    
    if df.empty:
        logger.warning("Empty pipeline results. No cases to export.")
        return
        
    # Sort by confidence
    df_sorted = df.sort_values(by="pipeline_confidence", ascending=False)
    
    # High confidence cases: top n_cases classified as PLANET
    df_planets = df_sorted[df_sorted["predicted_label_name"] == "PLANET"]
    df_high = df_planets.head(n_cases)
    
    # Low confidence cases: bottom n_cases (closest to decision boundary or most uncertain)
    # We can also sort by absolute value of distance from 0.5 or lowest pipeline_confidence
    df_low = df_sorted.tail(n_cases)
    
    high_path = output_dir / "high_confidence_cases.csv"
    low_path = output_dir / "low_confidence_cases.csv"
    
    df_high.to_csv(high_path, index=False)
    df_low.to_csv(low_path, index=False)
    
    logger.info("Exported %d high-confidence cases to %s", len(df_high), high_path)
    logger.info("Exported %d low-confidence cases to %s", len(df_low), low_path)

def main() -> None:
    parser = argparse.ArgumentParser(description="Export high and low confidence pipeline cases.")
    parser.add_argument("--results-csv", type=str, default=None, help="Path to pipeline_results.csv.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save exported case CSVs.")
    parser.add_argument("--n-cases", type=int, default=20, help="Number of cases to export.")
    args = parser.parse_args()
    
    root = project_root()
    results_csv = args.results_csv or root / get("paths.outputs", "outputs") / "pipeline_results.csv"
    output_dir = args.output_dir or root / get("paths.outputs", "outputs")
    
    export_cases(results_csv, output_dir, args.n_cases)

if __name__ == "__main__":
    main()
