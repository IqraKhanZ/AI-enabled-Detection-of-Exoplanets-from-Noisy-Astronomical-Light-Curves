"""
src/preparation/organize_storage.py
===================================
Organizes light curves into train, val, and test splits under data/processed/
and creates a master_index.csv of all targets.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.config import get, project_root
from utils.logger import get_logger

logger = get_logger(__name__)

def organize_files(
    lc_dir: str | Path,
    tpf_dir: str | Path,
    splits_csv: str | Path,
    qc_csv: str | Path,
    output_dir: str | Path,
    gaia_csv: str | Path | None = None
) -> None:
    """Organize light curves into split subdirectories and create master index."""
    lc_dir = Path(lc_dir)
    tpf_dir = Path(tpf_dir)
    splits_csv = Path(splits_csv)
    qc_csv = Path(qc_csv)
    output_dir = Path(output_dir)
    
    if not splits_csv.exists():
        logger.error("Splits file not found: %s", splits_csv)
        sys.exit(1)
    if not qc_csv.exists():
        logger.error("QC flags file not found: %s", qc_csv)
        sys.exit(1)
        
    df_splits = pd.read_csv(splits_csv)
    df_qc = pd.read_csv(qc_csv)
    
    # Merge splits and QC info
    df_master = pd.merge(df_splits, df_qc, on="tic_id", how="outer")
    df_master["pass_qc"] = df_master["pass_qc"].fillna(False)
    
    # Check Gaia crossmatch status
    gaia_set = set()
    if gaia_csv:
        gaia_csv = Path(gaia_csv)
        if gaia_csv.exists():
            try:
                df_gaia = pd.read_csv(gaia_csv)
                if "tic_id" in df_gaia.columns:
                    gaia_set = set(df_gaia["tic_id"].unique())
            except Exception as exc:
                logger.warning("Could not read Gaia crossmatch file: %s", exc)

    # Resolve light curve paths and TPF paths
    fits_files = sorted(lc_dir.glob("**/*.fits")) + sorted(lc_dir.glob("**/*.fit"))
    lc_map = {}
    for fp in fits_files:
        stem = fp.name
        # extract TIC ID from typical TESS file name
        parts = fp.stem.split("-")
        for part in parts:
            digits = "".join(c for c in part if c.isdigit())
            if 6 <= len(digits) <= 20:
                lc_map[int(digits)] = fp
                break

    tpf_files = sorted(tpf_dir.glob("**/*.fits")) + sorted(tpf_dir.glob("**/*.fit"))
    tpf_map = {}
    for fp in tpf_files:
        parts = fp.stem.split("-")
        for part in parts:
            digits = "".join(c for c in part if c.isdigit())
            if 6 <= len(digits) <= 20:
                tpf_map[int(digits)] = fp
                break

    records = []
    
    # Create target directories
    for split_name in ["train", "val", "test"]:
        (output_dir / split_name).mkdir(parents=True, exist_ok=True)
        
    for _, row in df_master.iterrows():
        tic_id = int(row["tic_id"])
        split = str(row["split"]) if pd.notna(row["split"]) else "none"
        label = int(row["label"]) if pd.notna(row["label"]) else -1
        qc_pass = bool(row["pass_qc"])
        
        orig_lc_path = lc_map.get(tic_id)
        orig_tpf_path = tpf_map.get(tic_id)
        has_gaia = tic_id in gaia_set
        
        dest_lc_path = ""
        dest_tpf_path = ""
        
        if qc_pass and split in ["train", "val", "test"] and orig_lc_path:
            # Copy file to partitioned folder
            dest_file = output_dir / split / orig_lc_path.name
            try:
                shutil.copy2(orig_lc_path, dest_file)
                dest_lc_path = str(dest_file.relative_to(project_root()))
            except Exception as exc:
                logger.error("Failed to copy light curve for TIC %d: %s", tic_id, exc)
                
            # If TPF is available, copy it too
            if orig_tpf_path:
                tpf_dest_file = output_dir / split / orig_tpf_path.name
                try:
                    shutil.copy2(orig_tpf_path, tpf_dest_file)
                    dest_tpf_path = str(tpf_dest_file.relative_to(project_root()))
                except Exception as exc:
                    logger.error("Failed to copy TPF for TIC %d: %s", tic_id, exc)
        else:
            if orig_lc_path:
                dest_lc_path = str(orig_lc_path.relative_to(project_root()))
            if orig_tpf_path:
                dest_tpf_path = str(orig_tpf_path.relative_to(project_root()))
                
        records.append({
            "tic_id": tic_id,
            "split": split,
            "label": label,
            "lc_path": dest_lc_path,
            "tpf_path": dest_tpf_path,
            "has_gaia_match": has_gaia,
            "qc_pass": qc_pass
        })
        
    df_index = pd.DataFrame(records)
    master_index_path = output_dir / "master_index.csv"
    df_index.to_csv(master_index_path, index=False)
    
    logger.info("Storage organization completed. Master index saved to %s", master_index_path)
    
    # Print summary
    counts = df_index[df_index["qc_pass"]]["split"].value_counts()
    logger.info("Passing QC targets organized by split: Train: %d, Val: %d, Test: %d", 
                counts.get("train", 0), counts.get("val", 0), counts.get("test", 0))

def main() -> None:
    parser = argparse.ArgumentParser(description="Organize light curve storage into split subdirectories.")
    parser.add_argument("--lc-dir", type=str, default=None, help="Path to raw light curves directory.")
    parser.add_argument("--tpf-dir", type=str, default=None, help="Path to raw TPF directory.")
    parser.add_argument("--splits-csv", type=str, default=None, help="Path to splits CSV.")
    parser.add_argument("--qc-csv", type=str, default=None, help="Path to QC flags CSV.")
    parser.add_argument("--gaia-csv", type=str, default=None, help="Path to Gaia matches CSV.")
    parser.add_argument("--output-dir", type=str, default=None, help="Path to output processed directory.")
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    args = parser.parse_args()
    
    root = project_root()
    lc_dir = args.lc_dir or root / get("paths.raw_lc", "data/raw/lightcurves", args.config)
    tpf_dir = args.tpf_dir or root / get("paths.raw_tpf", "data/raw/tpf", args.config)
    splits_csv = args.splits_csv or root / get("paths.processed", "data/processed", args.config) / "splits.csv"
    qc_csv = args.qc_csv or root / get("paths.processed", "data/processed", args.config) / "qc_flags.csv"
    gaia_csv = args.gaia_csv or root / get("paths.interim", "data/interim", args.config) / "gaia_matches.csv"
    output_dir = args.output_dir or root / get("paths.processed", "data/processed", args.config)
    
    organize_files(
        lc_dir=lc_dir,
        tpf_dir=tpf_dir,
        splits_csv=splits_csv,
        qc_csv=qc_csv,
        gaia_csv=gaia_csv,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()
