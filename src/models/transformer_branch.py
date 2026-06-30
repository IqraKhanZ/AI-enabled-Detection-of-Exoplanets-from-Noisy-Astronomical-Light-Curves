"""
src/models/transformer_branch.py
=================================
1D Self-Attention Transformer encoder branch for processing global-view
light-curve flux sequences.

The branch projects each time-step feature into a d_model-dimensional space,
prepends a learnable CLS token, adds learned positional encodings, then
passes the sequence through a stack of TransformerEncoderLayers.  The CLS
token output is returned as a fixed-size embedding for downstream fusion.

Classes
-------
TransformerBranch
    PyTorch nn.Module implementing the 1-D Transformer encoder.

Notes
-----
Class mapping used throughout the pipeline::

    PLANET           = 0
    ECLIPSING_BINARY = 1
    BLEND            = 2
    NOISE            = 3
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# TransformerBranch
# ---------------------------------------------------------------------------

class TransformerBranch(nn.Module):
    """1-D Self-Attention Transformer encoder branch.

    Accepts a batch of global-view flux sequences of shape ``(batch, seq_len)``
    and returns a ``(batch, d_model)`` embedding derived from a learnable CLS
    token prepended to the projected sequence.

    Parameters
    ----------
    d_model : int, optional
        Transformer model dimension.  Default ``128``.
    nhead : int, optional
        Number of attention heads.  Must evenly divide *d_model*.  Default ``8``.
    num_layers : int, optional
        Number of stacked ``TransformerEncoderLayer`` modules.  Default ``4``.
    dim_feedforward : int, optional
        Inner dimension of each layer's feed-forward block.  Default ``512``.
    dropout : float, optional
        Dropout probability applied inside each encoder layer and to the
        positional embedding.  Default ``0.1``.
    seq_len : int, optional
        Length of the input flux sequence (number of phase bins).  Default
        ``200``.

    Attributes
    ----------
    input_projection : nn.Linear
        Projects each scalar flux value to *d_model* dimensions.
    cls_token : nn.Parameter
        Learnable CLS token of shape ``(1, 1, d_model)``.
    pos_embedding : nn.Embedding
        Learned positional embeddings for positions ``0 ... seq_len`` (the ``+1``
        accounts for the prepended CLS token).
    encoder : nn.TransformerEncoder
        Stack of *num_layers* ``TransformerEncoderLayer`` modules.
    dropout : nn.Dropout
        Dropout applied to the embedded sequence before the encoder.
    _last_attn_weights : list[torch.Tensor] or None
        Stores attention weight tensors from the most recent forward pass
        when :py:meth:`get_attention_weights` hooks are active.

    Examples
    --------
    >>> branch = TransformerBranch()
    >>> x = torch.randn(4, 200)          # batch=4, seq_len=200
    >>> emb = branch(x)
    >>> emb.shape
    torch.Size([4, 128])
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        seq_len: int = 200,
    ) -> None:
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})."
            )

        self.d_model = d_model
        self.seq_len = seq_len

        # ------------------------------------------------------------------
        # 1. Linear projection: scalar flux -> d_model
        # ------------------------------------------------------------------
        self.input_projection = nn.Linear(1, d_model)

        # ------------------------------------------------------------------
        # 2. Learnable CLS token
        # ------------------------------------------------------------------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ------------------------------------------------------------------
        # 3. Learned positional encoding
        #    Position 0 -> CLS, positions 1...seq_len -> flux time steps
        # ------------------------------------------------------------------
        self.pos_embedding = nn.Embedding(seq_len + 1, d_model)

        # ------------------------------------------------------------------
        # 4. Transformer encoder layers
        # ------------------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,          # pre-layer-norm for training stability
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=encoder_norm,
        )

        self.dropout = nn.Dropout(p=dropout)

        # Attention-weight storage (populated by hooks when requested)
        self._last_attn_weights: list[torch.Tensor] = []
        self._hooks: list = []

        self._init_weights()
        logger.debug(
            "TransformerBranch initialised: d_model=%d nhead=%d "
            "num_layers=%d seq_len=%d",
            d_model, nhead, num_layers, seq_len,
        )

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Apply truncated-normal initialisation to linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of flux sequences and return CLS token embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, seq_len)`` -- global-view normalised flux values.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, d_model)`` -- CLS token embedding.

        Raises
        ------
        ValueError
            If the sequence length of *x* does not match ``self.seq_len``.
        """
        batch_size, seq_len = x.shape
        if seq_len != self.seq_len:
            raise ValueError(
                f"Expected seq_len={self.seq_len}, got {seq_len}."
            )

        device = x.device

        # (batch, seq_len, 1) -> (batch, seq_len, d_model)
        tokens = self.input_projection(x.unsqueeze(-1))

        # Prepend CLS token -> (batch, seq_len+1, d_model)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)          # (B, L+1, d_model)

        # Build position indices: [0, 1, 2, ..., seq_len]
        positions = torch.arange(seq_len + 1, device=device).unsqueeze(0)  # (1, L+1)
        tokens = tokens + self.pos_embedding(positions)

        tokens = self.dropout(tokens)

        # Transformer encoder
        encoded = self.encoder(tokens)                    # (B, L+1, d_model)

        # Extract CLS token (position 0)
        cls_output = encoded[:, 0, :]                     # (B, d_model)

        return cls_output

    # ------------------------------------------------------------------
    # Interpretability helpers
    # ------------------------------------------------------------------

    def register_attention_hooks(self) -> None:
        """Register forward hooks on all encoder layers to capture attention
        weight matrices during the next forward pass.

        Weights are stored in ``self._last_attn_weights`` as a list of
        tensors of shape ``(batch, nhead, seq_len+1, seq_len+1)``.

        Call :py:meth:`remove_attention_hooks` when finished to avoid
        memory leaks.
        """
        self._last_attn_weights = []
        self._hooks = []

        for layer in self.encoder.layers:
            hook = layer.self_attn.register_forward_hook(
                self._attn_hook_fn
            )
            self._hooks.append(hook)

    def remove_attention_hooks(self) -> None:
        """Remove all previously registered attention hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _attn_hook_fn(
        self,
        module: nn.Module,
        input: tuple,
        output: tuple,
    ) -> None:
        """Internal hook that captures attention weights."""
        # nn.MultiheadAttention returns (attn_output, attn_weights)
        # attn_weights may be None if need_weights=False
        _, attn_weights = output
        if attn_weights is not None:
            self._last_attn_weights.append(attn_weights.detach().cpu())

    def get_attention_weights(
        self,
        x: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Run a forward pass and return per-layer attention weight matrices.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, seq_len)``.

        Returns
        -------
        list[torch.Tensor]
            List of length *num_layers*, each tensor of shape
            ``(batch, nhead, seq_len+1, seq_len+1)``.
            Index 0 corresponds to the shallowest encoder layer.
        """
        self.register_attention_hooks()
        with torch.no_grad():
            self.forward(x)
        weights = list(self._last_attn_weights)  # copy
        self.remove_attention_hooks()
        return weights

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def num_params(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"TransformerBranch("
            f"d_model={self.d_model}, "
            f"seq_len={self.seq_len}, "
            f"params={self.num_params():,})"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke-test the TransformerBranch forward pass."
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=200)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    branch = TransformerBranch(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        seq_len=args.seq_len,
    )
    branch.eval()

    dummy = torch.randn(args.batch_size, args.seq_len)
    emb = branch(dummy)
    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {emb.shape}")
    print(f"Branch       : {branch}")

    attn = branch.get_attention_weights(dummy)
    print(f"Attention layers returned : {len(attn)}")
    print(f"Attention[0] shape        : {attn[0].shape}")
