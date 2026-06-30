"""
src/models/train_loop.py
==========================
Complete training loop with checkpointing, early stopping, and rich progress
display for the exoplanet classification pipeline.

Features
--------
* Adam optimiser with configurable weight decay.
* CosineAnnealingLR learning-rate scheduler.
* Weighted CrossEntropyLoss to handle class imbalance.
* Per-epoch logging of train loss, val loss, val macro-F1.
* Best-model checkpoint saved to ``checkpoints/best_model.pt``.
* Last-model checkpoint saved to ``checkpoints/last_model.pt``.
* Training history (JSON) saved to ``checkpoints/training_history.json``.
* Early stopping with configurable patience.
* Rich progress bars for batch iteration.

Class mapping::

    PLANET           = 0
    ECLIPSING_BINARY = 1
    BLEND            = 2
    NOISE            = 3

Entry point
-----------
Run directly::

    python -m models.train_loop --config config/pipeline_config.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from models.exoplanet_classifier import ExoplanetClassifier
from models.train_pipeline import create_dataloaders, get_class_weights
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_progress() -> "Progress | None":
    """Build a Rich Progress bar if Rich is available."""
    if not RICH_AVAILABLE:
        return None
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


def _to_device(
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Move batch tensors to *device* and return (global_view, river_plot, feature_vec, labels)."""
    global_view = batch["global_view"].to(device, non_blocking=True)
    river_plot = batch["river_plot"].to(device, non_blocking=True)
    feature_vec = batch.get("feature_vec")
    if feature_vec is not None and feature_vec.numel() > 0:
        feature_vec = feature_vec.to(device, non_blocking=True)
    else:
        feature_vec = None
    labels = batch["label"].to(device, non_blocking=True)
    return global_view, river_plot, feature_vec, labels


# ---------------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------------

def _train_epoch(
    model: ExoplanetClassifier,
    loader: "DataLoader",
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    progress: "Progress | None" = None,
    epoch: int = 0,
) -> float:
    """Run one training epoch.

    Parameters
    ----------
    model : ExoplanetClassifier
    loader : DataLoader
    optimizer : torch.optim.Optimizer
    loss_fn : nn.Module
    device : torch.device
    progress : rich.progress.Progress or None
    epoch : int

    Returns
    -------
    float
        Mean training loss over all batches.
    """
    model.train()
    total_loss = 0.0
    n_batches = len(loader)

    task_id = None
    if progress is not None:
        task_id = progress.add_task(
            f"[cyan]Epoch {epoch:3d} train", total=n_batches
        )

    for batch in loader:
        global_view, river_plot, feature_vec, labels = _to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(global_view, river_plot, feature_vec)
        loss = loss_fn(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        if progress is not None and task_id is not None:
            progress.advance(task_id)

    if progress is not None and task_id is not None:
        progress.remove_task(task_id)

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

@torch.no_grad()
def _validate(
    model: ExoplanetClassifier,
    loader: "DataLoader",
    loss_fn: nn.Module,
    device: torch.device,
    progress: "Progress | None" = None,
    epoch: int = 0,
) -> tuple[float, float]:
    """Run validation and compute loss + macro-F1.

    Parameters
    ----------
    model : ExoplanetClassifier
    loader : DataLoader
    loss_fn : nn.Module
    device : torch.device
    progress : rich.progress.Progress or None
    epoch : int

    Returns
    -------
    val_loss : float
    val_f1 : float
        Macro-averaged F1 score across all classes.
    """
    model.eval()
    total_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    n_batches = len(loader)

    task_id = None
    if progress is not None:
        task_id = progress.add_task(
            f"[green]Epoch {epoch:3d} val  ", total=n_batches
        )

    for batch in loader:
        global_view, river_plot, feature_vec, labels = _to_device(batch, device)
        logits, _ = model(global_view, river_plot, feature_vec)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

        if progress is not None and task_id is not None:
            progress.advance(task_id)

    if progress is not None and task_id is not None:
        progress.remove_task(task_id)

    all_preds_arr = np.concatenate(all_preds)
    all_labels_arr = np.concatenate(all_labels)

    val_f1 = float(
        f1_score(all_labels_arr, all_preds_arr, average="macro", zero_division=0)
    )
    return total_loss / max(n_batches, 1), val_f1


# ---------------------------------------------------------------------------
# Main train function
# ---------------------------------------------------------------------------

def train(config_path: str | Path | None = None) -> dict:
    """Run the full training loop.

    Parameters
    ----------
    config_path : str or Path or None
        Path to the pipeline YAML config.  If ``None``, tries
        ``config/pipeline_config.yaml``.

    Returns
    -------
    dict
        Training history: list of per-epoch dicts with keys
        ``epoch, train_loss, val_loss, val_f1, lr``.
    """
    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    if config_path is None:
        config_path = "config/pipeline_config.yaml"
    config = load_config(str(config_path))

    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    n_epochs = int(train_cfg.get("epochs", 50))
    learning_rate = float(train_cfg.get("learning_rate", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-5))
    patience = int(train_cfg.get("patience", 10))
    checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = checkpoint_dir / "best_model.pt"
    last_model_path = checkpoint_dir / "last_model.pt"
    history_path = checkpoint_dir / "training_history.json"

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, val_loader, _ = create_dataloaders(config)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    # Infer n_features from dataset
    n_features = train_loader.dataset.n_features
    config.setdefault("model", {})["n_features"] = n_features

    model = ExoplanetClassifier(config_dict=config)
    model.to(device)
    logger.info("Model: %s", model)

    # ------------------------------------------------------------------
    # Loss (weighted)
    # ------------------------------------------------------------------
    class_weights = get_class_weights(train_loader.dataset).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=learning_rate * 1e-2
    )

    # ------------------------------------------------------------------
    # Rich console
    # ------------------------------------------------------------------
    console = Console() if RICH_AVAILABLE else None

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history: list[dict] = []
    best_val_f1 = -1.0
    epochs_no_improve = 0

    progress = _make_progress()

    try:
        if progress is not None:
            progress.start()

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()

            train_loss = _train_epoch(
                model, train_loader, optimizer, loss_fn, device,
                progress=progress, epoch=epoch,
            )
            val_loss, val_f1 = _validate(
                model, val_loader, loss_fn, device,
                progress=progress, epoch=epoch,
            )
            scheduler.step()

            current_lr = float(optimizer.param_groups[0]["lr"])
            elapsed = time.time() - t0

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_f1": round(val_f1, 6),
                "lr": current_lr,
                "elapsed_s": round(elapsed, 2),
            }
            history.append(epoch_record)

            # Log epoch
            logger.info(
                "Epoch %03d/%03d | train_loss=%.4f val_loss=%.4f "
                "val_f1=%.4f lr=%.2e [%.1fs]",
                epoch, n_epochs, train_loss, val_loss, val_f1, current_lr, elapsed,
            )

            # Rich table row
            if console is not None:
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Epoch")
                table.add_column("Train Loss")
                table.add_column("Val Loss")
                table.add_column("Val F1")
                table.add_column("LR")
                table.add_row(
                    str(epoch),
                    f"{train_loss:.4f}",
                    f"{val_loss:.4f}",
                    f"{val_f1:.4f}",
                    f"{current_lr:.2e}",
                )
                console.print(table)

            # Save last checkpoint
            model.save(last_model_path)

            # Best checkpoint
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                epochs_no_improve = 0
                model.save(best_model_path)
                logger.info(
                    "New best model saved (val_f1=%.4f) to %s",
                    best_val_f1, best_model_path,
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(
                        "Early stopping triggered at epoch %d "
                        "(patience=%d, best val_f1=%.4f)",
                        epoch, patience, best_val_f1,
                    )
                    break

        # Save history
        with open(history_path, "w") as fh:
            json.dump(history, fh, indent=2)
        logger.info("Training history saved to %s", history_path)

    finally:
        if progress is not None:
            progress.stop()

    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the ExoplanetClassifier."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline_config.yaml",
        help="Path to the pipeline YAML configuration file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs from config.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate from config.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size from config.",
    )
    args = parser.parse_args()

    # Load and optionally patch config
    config = load_config(args.config)
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.lr is not None:
        config.setdefault("training", {})["learning_rate"] = args.lr
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size

    import tempfile, yaml
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as tmp:
        yaml.dump(config, tmp)
        tmp_path = tmp.name

    history = train(config_path=tmp_path)
    print(f"Training complete.  Best val_f1: {max(h['val_f1'] for h in history):.4f}")
