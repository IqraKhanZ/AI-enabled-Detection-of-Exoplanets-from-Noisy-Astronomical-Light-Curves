"""
src/preparation/train_val_test_split.py
=======================================
Splits the exoplanet dataset into train, validation, and test sets
stratified by target class with no leakage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.config import get, project_root
from utils.logger import get_logger

logger = get_logger(__name__)

def split_dataset(
    labels_csv: str | Path,
    output_csv: str | Path,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    stratify: bool = True
) -> None:
    """Split targets into train, val, and test partitions.
    
    Ensures that each TIC ID is assigned to exactly one partition.
    """
    labels_csv = Path(labels_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    if not labels_csv.exists():
        logger.error("Labels file not found: %s", labels_csv)
        sys.exit(1)
        
    df = pd.read_csv(labels_csv)
    if "tic_id" not in df.columns:
        logger.error("labels CSV must contain 'tic_id' column")
        sys.exit(1)
        
    # Group by tic_id to get unique targets and their labels
    # If a target has multiple rows, use the first label or majority
    target_groups = df.groupby("tic_id").first().reset_index()
    
    tic_ids = target_groups["tic_id"].values
    labels = target_groups["label"].values if "label" in target_groups.columns else None
    
    # Calculate relative fractions
    total_frac = train_frac + val_frac + test_frac
    train_frac /= total_frac
    val_frac /= total_frac
    test_frac /= total_frac
    
    # First split: train vs val+test
    val_test_size = val_frac + test_frac
    
    strat_labels = labels if (stratify and labels is not None) else None
    
    train_ids, val_test_ids, train_labels, val_test_labels = train_test_split(
        tic_ids,
        labels if labels is not None else tic_ids,
        test_size=val_test_size,
        random_state=seed,
        stratify=strat_labels
    )
    
    # Second split: val vs test
    test_rel_size = test_frac / val_test_size
    strat_labels_val_test = val_test_labels if (stratify and labels is not None) else None
    
    val_ids, test_ids = train_test_split(
        val_test_ids,
        test_size=test_rel_size,
        random_state=seed,
        stratify=strat_labels_val_test
    )
    
    # Map back to a splits dataframe
    splits_dict = {}
    for tid in train_ids:
        splits_dict[tid] = "train"
    for tid in val_ids:
        splits_dict[tid] = "val"
    for tid in test_ids:
        splits_dict[tid] = "test"
        
    df_splits = pd.DataFrame(list(splits_dict.items()), columns=["tic_id", "split"])
    
    # Merge label information back
    if labels is not None:
        label_map = dict(zip(target_groups["tic_id"], target_groups["label"]))
        df_splits["label"] = df_splits["tic_id"].map(label_map)
        
    df_splits.to_csv(output_csv, index=False)
    logger.info("Dataset split completed. Saved to %s", output_csv)
    
    # Print statistics
    counts = df_splits["split"].value_counts()
    logger.info("Split counts: Train: %d, Val: %d, Test: %d", 
                counts.get("train", 0), counts.get("val", 0), counts.get("test", 0))
    
    if "label" in df_splits.columns:
        distribution = pd.crosstab(df_splits["label"], df_splits["split"], normalize="columns") * 100
        logger.info("Class distribution per split (%%):\n%s", distribution.to_string())

def main() -> None:
    parser = argparse.ArgumentParser(description="Split exoplanet dataset into train/val/test splits.")
    parser.add_argument("--labels-csv", type=str, default=None, help="Path to input labels CSV.")
    parser.add_argument("--output-csv", type=str, default=None, help="Path to output splits CSV.")
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    args = parser.parse_args()
    
    root = project_root()
    labels_csv = args.labels_csv or root / get("paths.raw_labels", "data/raw/labels") / "toi_labels.csv"
    output_csv = args.output_csv or root / get("paths.processed", "data/processed") / "splits.csv"
    
    seed = get("split.random_seed", 42, args.config)
    train_frac = get("split.train_fraction", 0.70, args.config)
    val_frac = get("split.val_fraction", 0.15, args.config)
    test_frac = get("split.test_fraction", 0.15, args.config)
    stratify = get("split.stratify", True, args.config)
    
    split_dataset(
        labels_csv=labels_csv,
        output_csv=output_csv,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
        stratify=stratify
    )

if __name__ == "__main__":
    main()
