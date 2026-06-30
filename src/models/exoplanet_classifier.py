"""
src/models/exoplanet_classifier.py
=====================================
Full exoplanet classifier that wraps the Transformer branch, CNN branch,
and cross-attention fusion module into a single end-to-end PyTorch nn.Module.

Architecture
------------
    global_view   (B, 200)      --> TransformerBranch --> (B, 128)  --|
    river_plot    (B, 1, H, W)  --> CNNBranch          --> (B, 128)  --|-> CrossAttentionFusion --> (B, 4)
    feature_vec   (B, n_feat)   ------------------------------------------|

The classifier exposes:
    * ``forward``  : returns (logits, probs)
    * ``predict``  : returns (predicted_class, class_probs) as numpy/CPU
    * ``save``     : saves state dict + config dict to a .pt file
    * ``load``     : class-method to restore from a .pt file
    * ``num_params``: count trainable parameters

Class mapping::

    PLANET           = 0
    ECLIPSING_BINARY = 1
    BLEND            = 2
    NOISE            = 3

Classes
-------
ExoplanetClassifier
    Combined nn.Module for exoplanet vs false-positive classification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer_branch import TransformerBranch
from models.cnn_branch import CNNBranch
from models.cross_attention_fusion import CrossAttentionFusion
from utils.logger import get_logger

logger = get_logger(__name__)

# Class index constants
PLANET = 0
ECLIPSING_BINARY = 1
BLEND = 2
NOISE = 3

CLASS_NAMES = {
    PLANET: "PLANET",
    ECLIPSING_BINARY: "ECLIPSING_BINARY",
    BLEND: "BLEND",
    NOISE: "NOISE",
}


# ---------------------------------------------------------------------------
# ExoplanetClassifier
# ---------------------------------------------------------------------------

class ExoplanetClassifier(nn.Module):
    """Full exoplanet classification model.

    Combines a 1-D Transformer encoder on the global-view flux sequence, a
    2-D CNN encoder on the river-plot image, and a cross-attention fusion
    head that produces per-class softmax probabilities.

    Parameters
    ----------
    config_dict : dict
        Configuration dictionary.  Recognised keys (all optional):

        ``model.d_model`` : int
            Transformer model dimension.  Default ``128``.
        ``model.nhead`` : int
            Number of Transformer attention heads.  Default ``8``.
        ``model.num_transformer_layers`` : int
            Number of Transformer encoder layers.  Default ``4``.
        ``model.dim_feedforward`` : int
            Feed-forward hidden size in each Transformer layer.  Default ``512``.
        ``model.dropout`` : float
            Dropout rate.  Default ``0.1``.
        ``model.seq_len`` : int
            Global-view sequence length.  Default ``200``.
        ``model.n_features`` : int
            Number of hand-crafted scalar features.  Default ``0``.
        ``model.num_classes`` : int
            Number of output classes.  Default ``4``.

    Attributes
    ----------
    transformer_branch : TransformerBranch
    cnn_branch : CNNBranch
    fusion : CrossAttentionFusion
    config : dict
        Copy of the configuration dictionary used to build the model.

    Examples
    --------
    >>> cfg = {"model": {"d_model": 128, "n_features": 15}}
    >>> clf = ExoplanetClassifier(cfg)
    >>> gv   = torch.randn(2, 200)
    >>> rp   = torch.randn(2, 1, 20, 200)
    >>> fv   = torch.randn(2, 15)
    >>> logits, probs = clf(gv, rp, fv)
    >>> probs.shape
    torch.Size([2, 4])
    """

    def __init__(self, config_dict: dict | None = None) -> None:
        super().__init__()

        config_dict = config_dict or {}
        model_cfg = config_dict.get("model", {})

        self.config = config_dict

        # Read sub-module hyper-parameters
        d_model = int(model_cfg.get("d_model", 128))
        nhead = int(model_cfg.get("nhead", 8))
        num_transformer_layers = int(model_cfg.get("num_transformer_layers", 4))
        dim_feedforward = int(model_cfg.get("dim_feedforward", 512))
        dropout = float(model_cfg.get("dropout", 0.1))
        seq_len = int(model_cfg.get("seq_len", 200))
        n_features = int(model_cfg.get("n_features", 0))
        num_classes = int(model_cfg.get("num_classes", 4))

        # ------------------------------------------------------------------
        # Sub-modules
        # ------------------------------------------------------------------
        self.transformer_branch = TransformerBranch(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_transformer_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            seq_len=seq_len,
        )

        self.cnn_branch = CNNBranch(
            in_channels=1,
            channel_sizes=(16, 32, 64, 128),
            embedding_dim=d_model,
        )

        self.fusion = CrossAttentionFusion(
            d_model=d_model,
            num_classes=num_classes,
            n_features=n_features,
            dropout=dropout,
            mlp_hidden=256,
        )

        logger.info(
            "ExoplanetClassifier built: d_model=%d seq_len=%d "
            "n_features=%d num_classes=%d total_params=%s",
            d_model, seq_len, n_features, num_classes,
            f"{self.num_params():,}",
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        global_view: torch.Tensor,
        river_plot: torch.Tensor,
        feature_vec: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass through all branches.

        Parameters
        ----------
        global_view : torch.Tensor
            Shape ``(batch, seq_len)`` -- normalised global-view flux.
        river_plot : torch.Tensor
            Shape ``(batch, 1, H, W)`` -- river-plot image (H up to 20 cycles,
            W = 200 phase bins).
        feature_vec : torch.Tensor, optional
            Shape ``(batch, n_features)`` -- hand-crafted features.

        Returns
        -------
        logits : torch.Tensor
            Shape ``(batch, num_classes)`` -- raw classification logits.
        probs : torch.Tensor
            Shape ``(batch, num_classes)`` -- softmax probabilities.
        """
        # Branch encodings
        t_emb = self.transformer_branch(global_view)   # (B, d_model)
        c_emb = self.cnn_branch(river_plot)            # (B, d_model)

        # Fusion -> probabilities
        probs = self.fusion(t_emb, c_emb, feature_vec)     # (B, num_classes)
        logits = self.fusion.forward_logits(t_emb, c_emb, feature_vec)

        return logits, probs

    # ------------------------------------------------------------------
    # Prediction helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        global_view: torch.Tensor,
        river_plot: torch.Tensor,
        feature_vec: Optional[torch.Tensor] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict class labels and probabilities (no gradient computation).

        Parameters
        ----------
        global_view : torch.Tensor
            Shape ``(batch, seq_len)``.
        river_plot : torch.Tensor
            Shape ``(batch, 1, H, W)``.
        feature_vec : torch.Tensor, optional
            Shape ``(batch, n_features)``.

        Returns
        -------
        predicted_class : np.ndarray
            Shape ``(batch,)`` -- integer class indices.
        class_probs : np.ndarray
            Shape ``(batch, num_classes)`` -- softmax probabilities.
        """
        was_training = self.training
        self.eval()
        try:
            _, probs = self.forward(global_view, river_plot, feature_vec)
            predicted_class = probs.argmax(dim=-1).cpu().numpy()
            class_probs = probs.cpu().numpy()
        finally:
            if was_training:
                self.train()

        return predicted_class, class_probs

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save model state dict and configuration to a ``.pt`` file.

        Parameters
        ----------
        path : str or Path
            Destination file path (e.g. ``checkpoints/best_model.pt``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.config,
            },
            path,
        )
        logger.info("Model saved to %s", path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
    ) -> "ExoplanetClassifier":
        """Load a saved model from a ``.pt`` file.

        Parameters
        ----------
        path : str or Path
            Path to the saved ``.pt`` file.
        device : str or torch.device, optional
            Target device.  Default ``'cpu'``.

        Returns
        -------
        ExoplanetClassifier
            Model loaded with saved weights.
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint.get("config", {})
        model = cls(config_dict=config)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        logger.info("Model loaded from %s (device=%s)", path, device)
        return model

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def num_params(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_breakdown(self) -> dict[str, int]:
        """Return parameter counts per sub-module.

        Returns
        -------
        dict
            Keys: 'transformer_branch', 'cnn_branch', 'fusion', 'total'.
        """
        return {
            "transformer_branch": sum(
                p.numel() for p in self.transformer_branch.parameters()
                if p.requires_grad
            ),
            "cnn_branch": sum(
                p.numel() for p in self.cnn_branch.parameters()
                if p.requires_grad
            ),
            "fusion": sum(
                p.numel() for p in self.fusion.parameters()
                if p.requires_grad
            ),
            "total": self.num_params(),
        }

    def __repr__(self) -> str:  # noqa: D105
        breakdown = self.param_breakdown()
        return (
            f"ExoplanetClassifier(\n"
            f"  transformer_branch: {breakdown['transformer_branch']:,} params\n"
            f"  cnn_branch        : {breakdown['cnn_branch']:,} params\n"
            f"  fusion            : {breakdown['fusion']:,} params\n"
            f"  total             : {breakdown['total']:,} params\n"
            f")"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke-test ExoplanetClassifier forward pass."
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=200)
    parser.add_argument("--n_cycles", type=int, default=20)
    parser.add_argument("--n_features", type=int, default=15)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument(
        "--save", type=str, default="",
        help="If set, save model to this path and reload it.",
    )
    args = parser.parse_args()

    cfg = {
        "model": {
            "seq_len": args.seq_len,
            "n_features": args.n_features,
            "num_classes": args.num_classes,
        }
    }

    clf = ExoplanetClassifier(cfg)
    clf.eval()
    print(clf)

    gv = torch.randn(args.batch_size, args.seq_len)
    rp = torch.randn(args.batch_size, 1, args.n_cycles, 200)
    fv = torch.randn(args.batch_size, args.n_features)

    logits, probs = clf(gv, rp, fv)
    print(f"Logits shape : {logits.shape}")
    print(f"Probs  shape : {probs.shape}")
    print(f"Probs        : {probs}")

    pred_cls, pred_probs = clf.predict(gv, rp, fv)
    print(f"Predicted classes : {pred_cls}")
    for i, cls_idx in enumerate(pred_cls):
        print(f"  Sample {i}: {CLASS_NAMES[int(cls_idx)]} ({pred_probs[i, int(cls_idx)]:.3f})")

    if args.save:
        clf.save(args.save)
        clf2 = ExoplanetClassifier.load(args.save)
        _, probs2 = clf2(gv, rp, fv)
        max_diff = (probs - probs2).abs().max().item()
        print(f"Save/load round-trip max diff: {max_diff:.2e}")
