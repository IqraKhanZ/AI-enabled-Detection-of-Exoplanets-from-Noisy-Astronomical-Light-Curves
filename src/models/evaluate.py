"""
src/models/evaluate.py
========================
Comprehensive model evaluation for the exoplanet classification pipeline.

Loads the best model checkpoint and runs inference on the specified data
split (test by default), then computes and saves:

Metrics
-------
* Overall accuracy
* Per-class precision, recall, F1
* Macro and weighted averages
* ROC-AUC (one-vs-rest)
* Confusion matrix

Reports
-------
* ``reports/confusion_matrix.png``          -- heatmap (matplotlib dark style)
* ``reports/roc_curves.png``                -- ROC curve per class
* ``reports/pr_curves.png``                 -- Precision-recall curve per class
* ``reports/classification_metrics.json``   -- all numerical metrics

CLI
---
::

    python -m models.evaluate \\
        --checkpoint checkpoints/best_model.pt \\
        --config     config/pipeline_config.yaml \\
        --split      test

Class mapping::

    PLANET           = 0
    ECLIPSING_BINARY = 1
    BLEND            = 2
    NOISE            = 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from models.exoplanet_classifier import ExoplanetClassifier
from models.train_pipeline import create_dataloaders
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger(__name__)

CLASS_NAMES = ["PLANET", "ECLIPSING_BINARY", "BLEND", "NOISE"]
NUM_CLASSES = 4


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: ExoplanetClassifier,
    loader: "DataLoader",
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model over *loader* and collect predictions.

    Parameters
    ----------
    model : ExoplanetClassifier
    loader : DataLoader
    device : torch.device

    Returns
    -------
    all_labels : np.ndarray  shape (N,)
    all_preds  : np.ndarray  shape (N,)
    all_probs  : np.ndarray  shape (N, num_classes)
    """
    model.eval()
    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    for batch in loader:
        global_view = batch["global_view"].to(device)
        river_plot = batch["river_plot"].to(device)
        feature_vec = batch.get("feature_vec")
        if feature_vec is not None and feature_vec.numel() > 0:
            feature_vec = feature_vec.to(device)
        else:
            feature_vec = None
        labels = batch["label"].numpy()

        logits, probs = model(global_view, river_plot, feature_vec)
        preds = logits.argmax(dim=-1).cpu().numpy()
        probs_np = probs.cpu().numpy()

        all_labels.append(labels)
        all_preds.append(preds)
        all_probs.append(probs_np)

    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs, axis=0),
    )


# ---------------------------------------------------------------------------
# Plot: Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: Path,
    normalize: bool = True,
) -> None:
    """Save a confusion-matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Shape ``(num_classes, num_classes)`` -- raw count matrix.
    output_path : Path
        Destination PNG file.
    normalize : bool
        If ``True``, rows are normalised to sum to 1.
    """
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.where(row_sums == 0, 0, cm / row_sums)
        fmt_str = ".2f"
        title = "Confusion Matrix (row-normalised)"
    else:
        cm_plot = cm
        fmt_str = "d"
        title = "Confusion Matrix (counts)"

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues", vmin=0, vmax=1 if normalize else None)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(NUM_CLASSES),
        yticks=np.arange(NUM_CLASSES),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    ax.tick_params(axis="x", rotation=30)

    thresh = cm_plot.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = cm_plot[i, j]
            text = f"{val:{fmt_str}}" if fmt_str == ".2f" else f"{int(val)}"
            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if val < thresh else "black",
                fontsize=11,
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", output_path)


# ---------------------------------------------------------------------------
# Plot: ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    output_path: Path,
) -> dict:
    """Plot per-class ROC curves and return AUC scores.

    Parameters
    ----------
    all_labels : np.ndarray  (N,)
    all_probs  : np.ndarray  (N, num_classes)
    output_path : Path

    Returns
    -------
    dict
        Keys are class names; values are AUC floats.
    """
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#95E1D3"]
    aucs: dict[str, float] = {}

    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        aucs[name] = roc_auc
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "w--", lw=1, label="Random")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curves (One-vs-Rest)",
        xlim=[0, 1],
        ylim=[0, 1.02],
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curves saved to %s", output_path)
    return aucs


# ---------------------------------------------------------------------------
# Plot: PR curves
# ---------------------------------------------------------------------------

def plot_pr_curves(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    output_path: Path,
) -> None:
    """Plot per-class Precision-Recall curves.

    Parameters
    ----------
    all_labels : np.ndarray  (N,)
    all_probs  : np.ndarray  (N, num_classes)
    output_path : Path
    """
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#95E1D3"]

    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], all_probs[:, i])
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, color=color, lw=2,
                label=f"{name} (AUC={pr_auc:.3f})")

    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-Recall Curves",
        xlim=[0, 1],
        ylim=[0, 1.02],
    )
    ax.legend(loc="lower left")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("PR curves saved to %s", output_path)


# ---------------------------------------------------------------------------
# Rich summary table
# ---------------------------------------------------------------------------

def print_rich_table(metrics: dict) -> None:
    """Print a formatted summary table to the console using Rich."""
    if not RICH_AVAILABLE:
        print(json.dumps(metrics, indent=2))
        return

    console = Console()
    table = Table(title="Classification Metrics", show_header=True,
                  header_style="bold cyan")
    table.add_column("Class", style="bold")
    table.add_column("Precision")
    table.add_column("Recall")
    table.add_column("F1")
    table.add_column("Support")

    report = metrics.get("per_class", {})
    for name in CLASS_NAMES:
        row = report.get(name, {})
        table.add_row(
            name,
            f"{row.get('precision', 0):.4f}",
            f"{row.get('recall', 0):.4f}",
            f"{row.get('f1', 0):.4f}",
            str(int(row.get('support', 0))),
        )

    for avg in ("macro avg", "weighted avg"):
        row = report.get(avg, {})
        table.add_row(
            f"[italic]{avg}[/italic]",
            f"{row.get('precision', 0):.4f}",
            f"{row.get('recall', 0):.4f}",
            f"{row.get('f1', 0):.4f}",
            "-",
            style="dim",
        )

    console.print(table)
    console.print(f"[bold]Overall Accuracy:[/bold] {metrics.get('accuracy', 0):.4f}")
    roc_aucs = metrics.get("roc_auc_ovr", {})
    if roc_aucs:
        console.print("[bold]ROC-AUC (OVR):[/bold]")
        for cls, auc_val in roc_aucs.items():
            console.print(f"  {cls}: {auc_val:.4f}")


# ---------------------------------------------------------------------------
# Main evaluate function
# ---------------------------------------------------------------------------

def evaluate(
    checkpoint_path: str | Path = "checkpoints/best_model.pt",
    config_path: str | Path = "config/pipeline_config.yaml",
    split: str = "test",
    reports_dir: str | Path = "reports",
) -> dict:
    """Load checkpoint, run evaluation, save all reports.

    Parameters
    ----------
    checkpoint_path : str or Path
    config_path : str or Path
    split : str
        Which data split to evaluate: ``'train'``, ``'val'``, or ``'test'``.
    reports_dir : str or Path
        Directory for output reports.

    Returns
    -------
    dict
        All computed metrics.
    """
    checkpoint_path = Path(checkpoint_path)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(str(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ExoplanetClassifier.load(checkpoint_path, device=device)
    model.eval()

    # Get appropriate loader
    train_loader, val_loader, test_loader = create_dataloaders(config)
    split_map = {"train": train_loader, "val": val_loader, "test": test_loader}
    if split not in split_map:
        raise ValueError(f"split must be one of {list(split_map.keys())}, got '{split}'")
    loader = split_map[split]

    logger.info("Running inference on '%s' split (%d batches)...", split, len(loader))
    all_labels, all_preds, all_probs = run_inference(model, loader, device)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    accuracy = float(accuracy_score(all_labels, all_preds))

    cr = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))

    try:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))
        roc_auc_dict: dict[str, float] = {}
        for i, name in enumerate(CLASS_NAMES):
            fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
            roc_auc_dict[name] = float(auc(fpr, tpr))
        macro_roc_auc = float(
            roc_auc_score(y_bin, all_probs, multi_class="ovr", average="macro")
        )
    except Exception as exc:
        logger.warning("ROC-AUC computation failed: %s", exc)
        roc_auc_dict = {}
        macro_roc_auc = float("nan")

    # Build per-class metrics dict
    per_class = {}
    for cls_name in CLASS_NAMES:
        row = cr.get(cls_name, {})
        per_class[cls_name] = {
            "precision": float(row.get("precision", 0)),
            "recall": float(row.get("recall", 0)),
            "f1": float(row.get("f1-score", 0)),
            "support": int(row.get("support", 0)),
        }
    for avg in ("macro avg", "weighted avg"):
        row = cr.get(avg, {})
        per_class[avg] = {
            "precision": float(row.get("precision", 0)),
            "recall": float(row.get("recall", 0)),
            "f1": float(row.get("f1-score", 0)),
            "support": int(row.get("support", 0)),
        }

    metrics = {
        "split": split,
        "n_samples": int(len(all_labels)),
        "accuracy": accuracy,
        "per_class": per_class,
        "roc_auc_ovr": roc_auc_dict,
        "macro_roc_auc": macro_roc_auc,
        "confusion_matrix": cm.tolist(),
    }

    # ------------------------------------------------------------------
    # Save metrics JSON
    # ------------------------------------------------------------------
    metrics_path = reports_dir / "classification_metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # ------------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------------
    plot_confusion_matrix(cm, reports_dir / "confusion_matrix.png", normalize=True)
    plot_roc_curves(all_labels, all_probs, reports_dir / "roc_curves.png")
    plot_pr_curves(all_labels, all_probs, reports_dir / "pr_curves.png")

    # ------------------------------------------------------------------
    # Rich summary
    # ------------------------------------------------------------------
    print_rich_table(metrics)

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the ExoplanetClassifier on a data split."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint .pt file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline_config.yaml",
        help="Path to pipeline YAML config.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which data split to evaluate.",
    )
    parser.add_argument(
        "--reports_dir",
        type=str,
        default="reports",
        help="Directory to save report figures and JSON.",
    )
    args = parser.parse_args()

    metrics = evaluate(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        split=args.split,
        reports_dir=args.reports_dir,
    )
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Macro ROC-AUC: {metrics['macro_roc_auc']:.4f}")
