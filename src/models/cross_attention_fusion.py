"""
src/models/cross_attention_fusion.py
======================================
Cross-attention fusion module that combines the Transformer branch embedding
(from global-view flux) with the CNN branch embedding (from river-plot image)
and optionally hand-crafted feature vectors.

Architecture
------------
Given:
    - ``transformer_emb``: (batch, 128)  -- CLS token from TransformerBranch
    - ``cnn_emb``         : (batch, 128)  -- from CNNBranch
    - ``feature_vec``     : (batch, n_features)  -- optional hand-crafted features

Steps:
    1. Cross-attention with Q=transformer_emb, K=V=cnn_emb (single-head, d=128).
    2. Residual: fused = cross_attn_out + transformer_emb.
    3. If ``feature_vec`` provided: concat([fused, feature_vec]) -> project to 256.
       Else: Linear(128, 256).
    4. MLP head: Linear(256,256)->GELU->Dropout(0.1)->Linear(256,128)->GELU->Linear(128,n_classes).
    5. Softmax over class dimension.

Classes
-------
CrossAttentionFusion
    PyTorch nn.Module implementing the cross-attention fusion.

Notes
-----
Class mapping::

    PLANET           = 0
    ECLIPSING_BINARY = 1
    BLEND            = 2
    NOISE            = 3
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# CrossAttentionFusion
# ---------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion of Transformer and CNN branch embeddings.

    Fuses a pair of 128-dimensional branch embeddings via single-head
    cross-attention, optionally incorporates hand-crafted feature vectors,
    then applies an MLP classification head.

    Parameters
    ----------
    d_model : int, optional
        Embedding dimension expected from both branches.  Default ``128``.
    num_classes : int, optional
        Number of output classes.  Default ``4``
        (PLANET=0, ECLIPSING_BINARY=1, BLEND=2, NOISE=3).
    n_features : int, optional
        Length of the optional hand-crafted feature vector.  When ``0`` or
        ``None``, the feature pathway is disabled.  Default ``0``.
    dropout : float, optional
        Dropout rate inside the MLP head.  Default ``0.1``.
    mlp_hidden : int, optional
        Hidden size of the MLP head's first layer.  Default ``256``.

    Attributes
    ----------
    cross_attn : nn.MultiheadAttention
        Single-head cross-attention module.
    use_features : bool
        Whether hand-crafted features are incorporated.
    feature_proj : nn.Linear or None
        Projection from ``d_model + n_features`` to ``mlp_hidden``; only
        created when ``n_features > 0``.
    no_feature_proj : nn.Linear or None
        Projection from ``d_model`` to ``mlp_hidden``; created when
        ``n_features == 0``.
    mlp_head : nn.Sequential
        MLP classification head.

    Examples
    --------
    >>> fusion = CrossAttentionFusion(num_classes=4, n_features=15)
    >>> t_emb = torch.randn(4, 128)
    >>> c_emb = torch.randn(4, 128)
    >>> feat  = torch.randn(4, 15)
    >>> probs = fusion(t_emb, c_emb, feat)
    >>> probs.shape
    torch.Size([4, 4])
    >>> probs.sum(dim=-1)          # should be ~1.0
    tensor([1., 1., 1., 1.], grad_fn=<SumBackward1>)
    """

    def __init__(
        self,
        d_model: int = 128,
        num_classes: int = 4,
        n_features: int = 0,
        dropout: float = 0.1,
        mlp_hidden: int = 256,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_classes = num_classes
        self.n_features = n_features or 0
        self.use_features = self.n_features > 0
        self.mlp_hidden = mlp_hidden

        # ------------------------------------------------------------------
        # 1. Single-head cross-attention: Q = transformer_emb, K=V = cnn_emb
        # ------------------------------------------------------------------
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=1,
            dropout=0.0,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model)

        # ------------------------------------------------------------------
        # 2. Projection into MLP space
        # ------------------------------------------------------------------
        if self.use_features:
            self.feature_proj = nn.Linear(d_model + self.n_features, mlp_hidden)
            self.no_feature_proj = None
        else:
            self.feature_proj = None
            self.no_feature_proj = nn.Linear(d_model, mlp_hidden)

        # ------------------------------------------------------------------
        # 3. MLP classification head
        # ------------------------------------------------------------------
        self.mlp_head = nn.Sequential(
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.GELU(),
            nn.Linear(mlp_hidden // 2, num_classes),
        )

        self._init_weights()
        logger.debug(
            "CrossAttentionFusion initialised: d_model=%d n_classes=%d "
            "n_features=%d",
            d_model, num_classes, self.n_features,
        )

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Xavier uniform for Linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        transformer_emb: torch.Tensor,
        cnn_emb: torch.Tensor,
        feature_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse branch embeddings and classify.

        Parameters
        ----------
        transformer_emb : torch.Tensor
            Shape ``(batch, d_model)`` -- CLS embedding from TransformerBranch.
        cnn_emb : torch.Tensor
            Shape ``(batch, d_model)`` -- embedding from CNNBranch.
        feature_vec : torch.Tensor, optional
            Shape ``(batch, n_features)`` -- hand-crafted feature vector.
            Must be provided if the module was created with ``n_features > 0``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, num_classes)`` -- class probabilities (softmax).

        Raises
        ------
        ValueError
            If ``feature_vec`` is required but not provided, or vice versa.
        """
        # Validate
        if self.use_features and feature_vec is None:
            raise ValueError(
                "This fusion module expects feature_vec (n_features="
                f"{self.n_features}) but received None."
            )
        if not self.use_features and feature_vec is not None:
            logger.warning(
                "feature_vec provided but n_features=0; ignoring feature_vec."
            )

        # ------------------------------------------------------------------
        # Cross-attention
        # Q = transformer_emb: (B, 1, d_model)
        # K = V = cnn_emb     : (B, 1, d_model)
        # ------------------------------------------------------------------
        q = transformer_emb.unsqueeze(1)   # (B, 1, d)
        kv = cnn_emb.unsqueeze(1)          # (B, 1, d)

        attn_out, _ = self.cross_attn(q, kv, kv)   # (B, 1, d)
        attn_out = attn_out.squeeze(1)              # (B, d)

        # Residual connection + layer norm
        fused = self.attn_norm(attn_out + transformer_emb)   # (B, d)

        # ------------------------------------------------------------------
        # Projection into MLP space
        # ------------------------------------------------------------------
        if self.use_features and feature_vec is not None:
            combined = torch.cat([fused, feature_vec], dim=-1)  # (B, d+n_f)
            projected = self.feature_proj(combined)              # (B, mlp_hidden)
        else:
            projected = self.no_feature_proj(fused)              # (B, mlp_hidden)

        # ------------------------------------------------------------------
        # MLP head -> softmax
        # ------------------------------------------------------------------
        logits = self.mlp_head(projected)    # (B, num_classes)
        probs = F.softmax(logits, dim=-1)    # (B, num_classes)

        return probs

    def forward_logits(
        self,
        transformer_emb: torch.Tensor,
        cnn_emb: torch.Tensor,
        feature_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Like ``forward`` but returns raw logits (before softmax).

        Useful for use with ``nn.CrossEntropyLoss``.

        Parameters
        ----------
        transformer_emb : torch.Tensor
            Shape ``(batch, d_model)``.
        cnn_emb : torch.Tensor
            Shape ``(batch, d_model)``.
        feature_vec : torch.Tensor, optional
            Shape ``(batch, n_features)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, num_classes)`` -- raw logits.
        """
        if self.use_features and feature_vec is None:
            raise ValueError("feature_vec is required for this module.")

        q = transformer_emb.unsqueeze(1)
        kv = cnn_emb.unsqueeze(1)
        attn_out, _ = self.cross_attn(q, kv, kv)
        attn_out = attn_out.squeeze(1)
        fused = self.attn_norm(attn_out + transformer_emb)

        if self.use_features and feature_vec is not None:
            combined = torch.cat([fused, feature_vec], dim=-1)
            projected = self.feature_proj(combined)
        else:
            projected = self.no_feature_proj(fused)

        logits = self.mlp_head(projected)
        return logits

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def num_params(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"CrossAttentionFusion(d_model={self.d_model}, "
            f"num_classes={self.num_classes}, "
            f"n_features={self.n_features}, "
            f"params={self.num_params():,})"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke-test the CrossAttentionFusion module."
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--n_features", type=int, default=15)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    fusion = CrossAttentionFusion(
        d_model=args.d_model,
        num_classes=args.num_classes,
        n_features=args.n_features,
        dropout=args.dropout,
    )
    fusion.eval()

    t_emb = torch.randn(args.batch_size, args.d_model)
    c_emb = torch.randn(args.batch_size, args.d_model)
    feat = torch.randn(args.batch_size, args.n_features)

    probs = fusion(t_emb, c_emb, feat)
    print(f"Probs  shape : {probs.shape}")
    print(f"Row sums     : {probs.sum(dim=-1)}")
    print(f"Fusion       : {fusion}")

    # Test without features
    fusion_nf = CrossAttentionFusion(d_model=128, num_classes=4, n_features=0)
    probs_nf = fusion_nf(t_emb, c_emb)
    print(f"No-feature probs shape: {probs_nf.shape}")
