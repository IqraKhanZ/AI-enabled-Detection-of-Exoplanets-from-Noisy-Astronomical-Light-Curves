"""
src/models/train_pipeline.py
==============================
Dataset and DataLoader infrastructure for the exoplanet detection pipeline.

Provides:

* ``ExoplanetDataset``  -- ``torch.utils.data.Dataset`` that loads global-view
  flux arrays, river-plot images, and hand-crafted feature vectors from disk.
* ``get_class_weights`` -- computes inverse-frequency class weights for
  ``nn.CrossEntropyLoss``.
* ``create_dataloaders`` -- factory that builds train / val / test
  ``DataLoader`` objects from a configuration dictionary.

Class mapping::

    PLANET           = 0
    ECLIPSING_BINARY = 1
    BLEND            = 2
    NOISE            = 3

File layout expected on disk
----------------------------
``data_dir/{tic_id}_global_view.npy``   -- float32 array (200,)
``phase_results_dir/{tic_id}_river.npy`` -- float32 array (n_cycles, 200)

The index DataFrame (``index_df``) must contain at minimum:
    tic_id  : str or int
    label   : int  (0-3)

It may optionally contain pre-computed scalar feature columns; if not, the
feature vector is loaded from a separate ``{tic_id}_features.npy`` file.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from utils.logger import get_logger
from utils.config import load_config, get

logger = get_logger(__name__)

# Class index constants
PLANET = 0
ECLIPSING_BINARY = 1
BLEND = 2
NOISE = 3

# Maximum number of river-plot rows (transit cycles) to use
MAX_RIVER_HEIGHT = 20
# Standard global-view length
GLOBAL_VIEW_LEN = 200


# ---------------------------------------------------------------------------
# ExoplanetDataset
# ---------------------------------------------------------------------------

class ExoplanetDataset(Dataset):
    """PyTorch Dataset for exoplanet classification.

    Loads per-target NumPy arrays from disk for global-view flux, river-plot
    images, and hand-crafted feature vectors.  Optionally applies
    on-the-fly data augmentation.

    Parameters
    ----------
    index_df : pd.DataFrame
        Index table with at least ``tic_id`` (str/int) and ``label`` (int 0-3)
        columns.
    data_dir : str or Path
        Directory containing ``{tic_id}_global_view.npy`` files.
    phase_results_dir : str or Path
        Directory containing ``{tic_id}_river.npy`` files.
    feature_scaler_path : str or Path or None
        Path to a ``sklearn`` scaler ``.pkl`` file.  If provided, the hand-
        crafted feature vector is scaled before returning.  Pass ``None`` to
        skip scaling.
    augment : bool, optional
        Whether to apply stochastic data augmentation.  Default ``False``.
        Augmentations applied when ``True``:

        * Gaussian noise injection: ``flux += N(0, 0.001)``
        * Random phase shift: circular roll of ±10% of sequence length
        * Random flux scaling: multiply by ``U(0.98, 1.02)``

    Attributes
    ----------
    index_df : pd.DataFrame
    data_dir : Path
    phase_results_dir : Path
    feature_scaler : object or None
    augment : bool
    """

    def __init__(
        self,
        index_df: pd.DataFrame,
        data_dir: str | Path,
        phase_results_dir: str | Path,
        feature_scaler_path: str | Path | None,
        augment: bool = False,
    ) -> None:
        self.index_df = index_df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.phase_results_dir = Path(phase_results_dir)
        self.augment = augment

        # Load scaler if provided
        self.feature_scaler = None
        if feature_scaler_path is not None:
            scaler_path = Path(feature_scaler_path)
            if scaler_path.exists():
                with open(scaler_path, "rb") as fh:
                    self.feature_scaler = pickle.load(fh)
                logger.debug("Feature scaler loaded from %s", scaler_path)
            else:
                logger.warning(
                    "feature_scaler_path %s does not exist; skipping scaler.",
                    scaler_path,
                )

        # Identify feature columns (any column that is not tic_id / label /
        # known metadata columns)
        meta_cols = {
            "tic_id", "label", "split", "sector", "ra", "dec",
            "tmag", "teff", "logg", "radius",
        }
        self._feature_cols = [
            c for c in self.index_df.columns if c not in meta_cols
        ]
        logger.debug(
            "ExoplanetDataset: %d samples, %d feature columns, augment=%s",
            len(self.index_df), len(self._feature_cols), augment,
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:  # noqa: D105
        return len(self.index_df)

    def __getitem__(self, idx: int) -> dict:
        """Return a single training sample.

        Parameters
        ----------
        idx : int
            Row index into ``self.index_df``.

        Returns
        -------
        dict with keys:

        ``global_view``  : torch.Tensor of shape ``(200,)``  -- float32
        ``river_plot``   : torch.Tensor of shape ``(1, 20, 200)``  -- float32
        ``feature_vec``  : torch.Tensor of shape ``(n_features,)``  -- float32
        ``label``        : torch.Tensor scalar -- int64
        ``tic_id``       : str
        """
        row = self.index_df.iloc[idx]
        tic_id = str(row["tic_id"])
        label = int(row["label"])

        # ------------------------------------------------------------------
        # 1. Global view
        # ------------------------------------------------------------------
        global_view = self._load_global_view(tic_id)

        # ------------------------------------------------------------------
        # 2. River plot
        # ------------------------------------------------------------------
        river_plot = self._load_river_plot(tic_id)

        # ------------------------------------------------------------------
        # 3. Feature vector
        # ------------------------------------------------------------------
        feature_vec = self._load_feature_vec(row, tic_id)

        # ------------------------------------------------------------------
        # 4. Data augmentation (training only)
        # ------------------------------------------------------------------
        if self.augment:
            global_view, river_plot = self._augment(global_view, river_plot)

        return {
            "global_view": global_view,
            "river_plot": river_plot,
            "feature_vec": feature_vec,
            "label": torch.tensor(label, dtype=torch.int64),
            "tic_id": tic_id,
        }

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _load_global_view(self, tic_id: str) -> torch.Tensor:
        """Load and validate the global-view flux array."""
        path = self.data_dir / f"{tic_id}_global_view.npy"
        if path.exists():
            arr = np.load(path).astype(np.float32)
        else:
            logger.warning(
                "global_view not found for TIC %s at %s; using zeros.", tic_id, path
            )
            arr = np.zeros(GLOBAL_VIEW_LEN, dtype=np.float32)

        # Ensure correct length
        if len(arr) < GLOBAL_VIEW_LEN:
            arr = np.pad(arr, (0, GLOBAL_VIEW_LEN - len(arr)), constant_values=0.0)
        elif len(arr) > GLOBAL_VIEW_LEN:
            arr = arr[:GLOBAL_VIEW_LEN]

        # Clip extreme outliers (> 5-sigma)
        med = float(np.nanmedian(arr))
        std = float(np.nanstd(arr)) + 1e-9
        arr = np.clip(arr, med - 5 * std, med + 5 * std)

        return torch.tensor(arr, dtype=torch.float32)

    def _load_river_plot(self, tic_id: str) -> torch.Tensor:
        """Load, pad/crop, and return river-plot image tensor."""
        path = self.phase_results_dir / f"{tic_id}_river.npy"
        if path.exists():
            arr = np.load(path).astype(np.float32)
        else:
            logger.warning(
                "river_plot not found for TIC %s at %s; using zeros.", tic_id, path
            )
            arr = np.zeros((1, GLOBAL_VIEW_LEN), dtype=np.float32)

        # Ensure arr is 2D (n_cycles, n_bins)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]

        n_cycles, n_bins = arr.shape

        # Crop/pad phase axis to GLOBAL_VIEW_LEN
        if n_bins < GLOBAL_VIEW_LEN:
            arr = np.pad(arr, ((0, 0), (0, GLOBAL_VIEW_LEN - n_bins)))
        elif n_bins > GLOBAL_VIEW_LEN:
            arr = arr[:, :GLOBAL_VIEW_LEN]

        # Crop/pad cycle axis to MAX_RIVER_HEIGHT
        if n_cycles < MAX_RIVER_HEIGHT:
            arr = np.pad(arr, ((0, MAX_RIVER_HEIGHT - n_cycles), (0, 0)))
        elif n_cycles > MAX_RIVER_HEIGHT:
            arr = arr[:MAX_RIVER_HEIGHT, :]

        # Add channel dimension: (1, H, W)
        arr = arr[np.newaxis, ...]
        return torch.tensor(arr, dtype=torch.float32)

    def _load_feature_vec(self, row: pd.Series, tic_id: str) -> torch.Tensor:
        """Return hand-crafted feature vector for this sample.

        First tries feature columns in the DataFrame row, then falls back to
        a per-target ``.npy`` file, then returns zeros.
        """
        if self._feature_cols:
            vec = row[self._feature_cols].values.astype(np.float32)
            # Replace NaN with 0
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            # Try per-target file
            feat_path = self.data_dir / f"{tic_id}_features.npy"
            if feat_path.exists():
                vec = np.load(feat_path).astype(np.float32)
                vec = np.nan_to_num(vec, nan=0.0)
            else:
                vec = np.zeros(0, dtype=np.float32)

        if self.feature_scaler is not None and len(vec) > 0:
            try:
                vec = self.feature_scaler.transform(vec.reshape(1, -1)).ravel().astype(np.float32)
            except Exception as exc:
                logger.warning("Feature scaling failed for TIC %s: %s", tic_id, exc)

        return torch.tensor(vec, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def _augment(
        self,
        global_view: torch.Tensor,
        river_plot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply stochastic augmentations to flux arrays.

        Augmentations applied:
        * Gaussian noise injection to global_view.
        * Random phase shift (circular roll ±10%).
        * Random flux scaling (±2%).

        Parameters
        ----------
        global_view : torch.Tensor
            Shape ``(200,)``.
        river_plot : torch.Tensor
            Shape ``(1, 20, 200)``.

        Returns
        -------
        global_view, river_plot : torch.Tensor, torch.Tensor
            Augmented tensors.
        """
        # 1. Gaussian noise
        noise_scale = 0.001
        global_view = global_view + torch.randn_like(global_view) * noise_scale

        # 2. Random phase shift (circular roll along phase axis)
        max_shift = GLOBAL_VIEW_LEN // 10
        shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
        global_view = torch.roll(global_view, shifts=shift, dims=0)
        river_plot = torch.roll(river_plot, shifts=shift, dims=2)   # roll along W

        # 3. Random flux scaling
        scale = 1.0 + (torch.rand(1).item() * 0.04 - 0.02)   # U(0.98, 1.02)
        global_view = global_view * scale
        river_plot = river_plot * scale

        return global_view, river_plot

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def labels(self) -> np.ndarray:
        """Return integer label array for all samples."""
        return self.index_df["label"].values.astype(np.int64)

    @property
    def n_features(self) -> int:
        """Return the number of hand-crafted scalar features."""
        if self._feature_cols:
            return len(self._feature_cols)
        # Peek at a file
        tic_id = str(self.index_df.iloc[0]["tic_id"])
        feat_path = self.data_dir / f"{tic_id}_features.npy"
        if feat_path.exists():
            return int(np.load(feat_path).shape[0])
        return 0


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def get_class_weights(dataset: ExoplanetDataset) -> torch.Tensor:
    """Compute inverse-frequency class weights for weighted CrossEntropyLoss.

    Parameters
    ----------
    dataset : ExoplanetDataset
        The dataset from which to count class occurrences.

    Returns
    -------
    torch.Tensor
        Shape ``(num_classes,)`` -- weight for each class.
        Larger weight for under-represented classes.

    Examples
    --------
    >>> weights = get_class_weights(train_dataset)
    >>> loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    """
    labels = dataset.labels
    num_classes = 4
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)   # avoid division by zero
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # normalise so mean weight == 1
    logger.info("Class weights: %s", dict(enumerate(weights.round(4))))
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloaders(
    config: dict,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test DataLoaders from a config dict.

    Expected config keys::

        data.index_csv           : path to master index CSV
        data.data_dir            : directory with global-view .npy files
        data.phase_results_dir   : directory with river-plot .npy files
        data.feature_scaler_path : path to sklearn scaler .pkl (optional)
        training.batch_size      : int (default 32)
        training.num_workers     : int (default 4)
        training.use_weighted_sampler : bool (default True)

    The index CSV must have columns: tic_id, label, split
    where split in {'train', 'val', 'test'}.

    Parameters
    ----------
    config : dict
        Pipeline configuration dictionary (as returned by
        ``utils.config.load_config``).

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader, DataLoader, DataLoader

    Raises
    ------
    FileNotFoundError
        If the index CSV does not exist.
    ValueError
        If the required columns are missing from the index CSV.
    """
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})

    index_csv = Path(data_cfg.get("index_csv", "data/index.csv"))
    data_dir = Path(data_cfg.get("data_dir", "data/processed"))
    phase_results_dir = Path(data_cfg.get("phase_results_dir", "data/phase_results"))
    feature_scaler_path = data_cfg.get("feature_scaler_path", None)

    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 4))
    use_weighted_sampler = bool(train_cfg.get("use_weighted_sampler", True))
    pin_memory = torch.cuda.is_available()

    if not index_csv.exists():
        raise FileNotFoundError(f"Index CSV not found: {index_csv}")

    index_df = pd.read_csv(index_csv)
    required_cols = {"tic_id", "label", "split"}
    missing = required_cols - set(index_df.columns)
    if missing:
        raise ValueError(f"Index CSV missing required columns: {missing}")

    train_df = index_df[index_df["split"] == "train"].copy()
    val_df = index_df[index_df["split"] == "val"].copy()
    test_df = index_df[index_df["split"] == "test"].copy()

    logger.info(
        "Dataset split sizes: train=%d val=%d test=%d",
        len(train_df), len(val_df), len(test_df),
    )

    train_ds = ExoplanetDataset(
        index_df=train_df,
        data_dir=data_dir,
        phase_results_dir=phase_results_dir,
        feature_scaler_path=feature_scaler_path,
        augment=True,
    )
    val_ds = ExoplanetDataset(
        index_df=val_df,
        data_dir=data_dir,
        phase_results_dir=phase_results_dir,
        feature_scaler_path=feature_scaler_path,
        augment=False,
    )
    test_ds = ExoplanetDataset(
        index_df=test_df,
        data_dir=data_dir,
        phase_results_dir=phase_results_dir,
        feature_scaler_path=feature_scaler_path,
        augment=False,
    )

    # Build train sampler (weighted or sequential)
    train_sampler = None
    shuffle = True
    if use_weighted_sampler and len(train_ds) > 0:
        class_weights = get_class_weights(train_ds)
        sample_weights = class_weights[train_ds.labels]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_ds),
            replacement=True,
        )
        shuffle = False   # mutually exclusive with sampler

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    logger.info(
        "DataLoaders created: train=%d batches, val=%d batches, test=%d batches",
        len(train_loader), len(val_loader), len(test_loader),
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect and validate the ExoplanetDataset."
    )
    parser.add_argument(
        "--config", type=str, default="config/pipeline_config.yaml",
        help="Path to pipeline config YAML.",
    )
    parser.add_argument(
        "--index_csv", type=str, default="",
        help="Override index CSV path.",
    )
    parser.add_argument(
        "--data_dir", type=str, default="",
        help="Override data directory.",
    )
    parser.add_argument(
        "--phase_dir", type=str, default="",
        help="Override phase results directory.",
    )
    parser.add_argument("--n_samples", type=int, default=3)
    args = parser.parse_args()

    # Build a minimal config if the YAML does not exist
    try:
        config = load_config(args.config)
    except Exception:
        config = {}

    if args.index_csv:
        config.setdefault("data", {})["index_csv"] = args.index_csv
    if args.data_dir:
        config.setdefault("data", {})["data_dir"] = args.data_dir
    if args.phase_dir:
        config.setdefault("data", {})["phase_results_dir"] = args.phase_dir

    index_csv_path = Path(config.get("data", {}).get("index_csv", "data/index.csv"))
    if not index_csv_path.exists():
        print(f"Index CSV not found at {index_csv_path}. "
              "Please specify --index_csv.")
    else:
        train_loader, val_loader, test_loader = create_dataloaders(config)
        batch = next(iter(train_loader))
        print("=== First train batch ===")
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
            else:
                print(f"  {key}: {val[:args.n_samples]}")
