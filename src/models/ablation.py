"""
src/models/ablation.py
========================
Ablation study comparing four model configurations to assess the contribution
of each architectural component.

Configurations
--------------
1. **transformer_only** -- TransformerBranch + MLP head; no CNN branch.
2. **cnn_only**         -- CNNBranch + MLP head; no Transformer branch.
3. **features_only**    -- Plain MLP operating on hand-crafted features only.
4. **full_hybrid**      -- Complete ExoplanetClassifier (all components).

Each configuration is trained for a reduced number of epochs (default 10) on
the same dataset split, then evaluated on the validation set.  Results are
saved as:

* ``reports/ablation_results.csv``   -- comparison table
* ``reports/ablation_bar_chart.png`` -- bar chart of val F1 scores

Class mapping::

    PLANET           = 0
    ECLIPSING_BINARY = 1
    BLEND            = 2
    NOISE            = 3

CLI
---
::

    python -m models.ablation --config config/pipeline_config.yaml --epochs 10
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
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.exoplanet_classifier import ExoplanetClassifier
from models.train_pipeline import create_dataloaders, get_class_weights
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger(__name__)

CLASS_NAMES = ["PLANET", "ECLIPSING_BINARY", "BLEND", "NOISE"]
NUM_CLASSES = 4


# ---------------------------------------------------------------------------
# Ablation sub-models
# ---------------------------------------------------------------------------

class _TransformerOnlyModel(nn.Module):
    """TransformerBranch + MLP, no CNN."""

    def __init__(self, d_model: int = 128, seq_len: int = 200,
                 num_classes: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        from models.transformer_branch import TransformerBranch
        self.transformer = TransformerBranch(
            d_model=d_model, seq_len=seq_len, dropout=dropout
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, global_view: torch.Tensor, river_plot: torch.Tensor,
                feature_vec: Optional[torch.Tensor] = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.transformer(global_view)
        logits = self.head(emb)
        return logits, F.softmax(logits, dim=-1)


class _CNNOnlyModel(nn.Module):
    """CNNBranch + MLP, no Transformer."""

    def __init__(self, d_model: int = 128, num_classes: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        from models.cnn_branch import CNNBranch
        self.cnn = CNNBranch(embedding_dim=d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, global_view: torch.Tensor, river_plot: torch.Tensor,
                feature_vec: Optional[torch.Tensor] = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.cnn(river_plot)
        logits = self.head(emb)
        return logits, F.softmax(logits, dim=-1)


class _FeaturesOnlyModel(nn.Module):
    """Plain MLP operating on hand-crafted features only."""

    def __init__(self, n_features: int, num_classes: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.n_features = n_features
        if n_features == 0:
            n_features = 1   # guard against degenerate case
        self.head = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, global_view: torch.Tensor, river_plot: torch.Tensor,
                feature_vec: Optional[torch.Tensor] = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        if feature_vec is None or feature_vec.numel() == 0:
            # Fall back to zeros of the expected size
            b = global_view.shape[0]
            feature_vec = torch.zeros(b, max(self.n_features, 1),
                                      device=global_view.device)
        logits = self.head(feature_vec)
        return logits, F.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Train / eval helpers (shared)
# ---------------------------------------------------------------------------

def _to_device(batch: dict, device: torch.device):
    gv = batch["global_view"].to(device)
    rp = batch["river_plot"].to(device)
    fv = batch.get("feature_vec")
    if fv is not None and fv.numel() > 0:
        fv = fv.to(device)
    else:
        fv = None
    labels = batch["label"].to(device)
    return gv, rp, fv, labels


def _train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        gv, rp, fv, labels = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(gv, rp, fv)
        loss = loss_fn(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def _evaluate_f1(model, loader, device) -> float:
    model.eval()
    preds_all, labels_all = [], []
    for batch in loader:
        gv, rp, fv, labels = _to_device(batch, device)
        logits, _ = model(gv, rp, fv)
        preds_all.append(logits.argmax(-1).cpu().numpy())
        labels_all.append(labels.cpu().numpy())
    import numpy as np
    preds = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)
    return float(f1_score(labels, preds, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# Run one ablation configuration
# ---------------------------------------------------------------------------

def _run_config(
    name: str,
    model: nn.Module,
    train_loader,
    val_loader,
    class_weights: torch.Tensor,
    n_epochs: int,
    lr: float,
    device: torch.device,
) -> dict:
    """Train and evaluate a single ablation model.

    Returns
    -------
    dict
        Keys: name, best_val_f1, history (list of per-epoch val_f1).
    """
    logger.info("=== Ablation: %s ===", name)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    best_f1 = -1.0
    history = []

    for epoch in range(1, n_epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_f1 = _evaluate_f1(model, val_loader, device)
        scheduler.step()
        best_f1 = max(best_f1, val_f1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_f1": val_f1})
        logger.info(
            "  [%s] Epoch %02d/%02d | train_loss=%.4f val_f1=%.4f",
            name, epoch, n_epochs, train_loss, val_f1,
        )

    return {"name": name, "best_val_f1": best_f1, "history": history}


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------

def _plot_bar_chart(results: list[dict], output_path: Path) -> None:
    """Save a bar chart comparing best validation F1 across configurations."""
    names = [r["name"] for r in results]
    f1s = [r["best_val_f1"] for r in results]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4ECDC4", "#FF6B6B", "#FFE66D", "#95E1D3"]
    bars = ax.bar(names, f1s, color=colors[:len(names)], edgecolor="white", linewidth=0.5)

    for bar, f1 in zip(bars, f1s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{f1:.4f}",
            ha="center", va="bottom", fontsize=12, color="white",
        )

    ax.set(
        xlabel="Configuration",
        ylabel="Best Validation Macro-F1",
        title="Ablation Study: Component Contribution",
        ylim=[0, 1.05],
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Ablation bar chart saved to %s", output_path)


# ---------------------------------------------------------------------------
# Main ablation function
# ---------------------------------------------------------------------------

def run_ablation(
    config_path: str | Path = "config/pipeline_config.yaml",
    n_epochs: int = 10,
    lr: float = 1e-4,
    reports_dir: str | Path = "reports",
) -> pd.DataFrame:
    """Run all four ablation configurations and save comparison results.

    Parameters
    ----------
    config_path : str or Path
    n_epochs : int
        Number of training epochs per configuration.  Default ``10``.
    lr : float
        Learning rate.  Default ``1e-4``.
    reports_dir : str or Path
        Directory to save outputs.

    Returns
    -------
    pd.DataFrame
        Summary table with columns: name, best_val_f1.
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(str(config_path))
    model_cfg = config.get("model", {})

    d_model = int(model_cfg.get("d_model", 128))
    seq_len = int(model_cfg.get("seq_len", 200))
    dropout = float(model_cfg.get("dropout", 0.1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Ablation study on device: %s", device)

    train_loader, val_loader, _ = create_dataloaders(config)
    n_features = train_loader.dataset.n_features
    class_weights = get_class_weights(train_loader.dataset)

    # ------------------------------------------------------------------
    # Define ablation models
    # ------------------------------------------------------------------
    configs_to_run = [
        ("transformer_only", _TransformerOnlyModel(
            d_model=d_model, seq_len=seq_len, num_classes=NUM_CLASSES, dropout=dropout
        )),
        ("cnn_only", _CNNOnlyModel(
            d_model=d_model, num_classes=NUM_CLASSES, dropout=dropout
        )),
        ("features_only", _FeaturesOnlyModel(
            n_features=n_features, num_classes=NUM_CLASSES, dropout=dropout
        )),
        ("full_hybrid", ExoplanetClassifier(config_dict={
            **config,
            "model": {**model_cfg, "n_features": n_features, "num_classes": NUM_CLASSES},
        })),
    ]

    all_results: list[dict] = []

    for name, model in configs_to_run:
        result = _run_config(
            name=name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=class_weights,
            n_epochs=n_epochs,
            lr=lr,
            device=device,
        )
        all_results.append(result)

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    summary_rows = [
        {"configuration": r["name"], "best_val_f1": round(r["best_val_f1"], 6)}
        for r in all_results
    ]
    df = pd.DataFrame(summary_rows)
    csv_path = reports_dir / "ablation_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Ablation results saved to %s", csv_path)

    # Save full history as JSON
    json_path = reports_dir / "ablation_history.json"
    with open(json_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    logger.info("Ablation history saved to %s", json_path)

    # ------------------------------------------------------------------
    # Bar chart
    # ------------------------------------------------------------------
    _plot_bar_chart(all_results, reports_dir / "ablation_bar_chart.png")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n=== Ablation Study Results ===")
    print(df.to_string(index=False))

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ablation study comparing model configurations."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline_config.yaml",
        help="Path to pipeline YAML config.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs per configuration.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--reports_dir",
        type=str,
        default="reports",
        help="Output directory for results.",
    )
    args = parser.parse_args()

    df = run_ablation(
        config_path=args.config,
        n_epochs=args.epochs,
        lr=args.lr,
        reports_dir=args.reports_dir,
    )
    print("\nBest configuration:",
          df.loc[df["best_val_f1"].idxmax(), "configuration"],
          f"(F1={df['best_val_f1'].max():.4f})")
