"""
src/models/cnn_branch.py
=========================
2-D CNN branch for processing river-plot images of phase-folded light curves.

A river plot stacks successive transit windows as rows, producing a 2-D image
of shape ``(n_cycles, n_phase_bins)``.  This module encodes that image into a
fixed-size embedding using a hierarchical convolutional network.

Architecture overview::

    Input: (batch, 1, H, W)  — H = n_cycles (up to 20), W = 200 phase bins
    Block 1:  Conv2d(1 ->16,  k=3, pad=1) -> BN -> GELU -> MaxPool(2,2)
    Block 2:  Conv2d(16->32,  k=3, pad=1) -> BN -> GELU -> MaxPool(2,2)
    Block 3:  Conv2d(32->64,  k=3, pad=1) -> BN -> GELU -> MaxPool(2,2)
    Block 4:  Conv2d(64->128, k=3, pad=1) -> BN -> GELU -> MaxPool(2,2)
    AdaptiveAvgPool2d(1, 1) -> Flatten -> Linear(128, 128)
    Output: (batch, 128)

Classes
-------
CNNBranch
    PyTorch nn.Module implementing the 2-D CNN encoder.

Notes
-----
Class mapping::

    PLANET           = 0
    ECLIPSING_BINARY = 1
    BLEND            = 2
    NOISE            = 3
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper: single convolutional block
# ---------------------------------------------------------------------------

def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Return a single Conv -> BN -> GELU -> MaxPool block.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.

    Returns
    -------
    nn.Sequential
        The composed block.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.GELU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


# ---------------------------------------------------------------------------
# CNNBranch
# ---------------------------------------------------------------------------

class CNNBranch(nn.Module):
    """2-D Convolutional Neural Network branch for river-plot images.

    Accepts river-plot images of shape ``(batch, 1, H, W)`` where *H* is the
    number of stacked transit cycles (up to 20) and *W* is the number of phase
    bins (default 200).  Returns a ``(batch, 128)`` embedding vector.

    The branch handles variable-height inputs through a final
    ``AdaptiveAvgPool2d(1, 1)`` that collapses spatial dimensions regardless
    of input size.  This is particularly useful when different targets have
    different numbers of observed transits.

    Parameters
    ----------
    in_channels : int, optional
        Number of input image channels.  Default ``1`` (greyscale river plot).
    channel_sizes : Sequence[int], optional
        Channel widths after each convolutional block.
        Default ``(16, 32, 64, 128)``.
    embedding_dim : int, optional
        Dimensionality of the output embedding.  Default ``128``.

    Attributes
    ----------
    blocks : nn.ModuleList
        List of Conv-BN-GELU-MaxPool blocks.
    global_pool : nn.AdaptiveAvgPool2d
        Adaptive average pooling to ``(1, 1)``.
    head : nn.Sequential
        Final linear projection and activation.

    Examples
    --------
    >>> branch = CNNBranch()
    >>> img = torch.randn(4, 1, 20, 200)   # batch=4, H=20, W=200
    >>> emb = branch(img)
    >>> emb.shape
    torch.Size([4, 128])
    >>> img_small = torch.randn(4, 1, 5, 200)   # only 5 cycles
    >>> branch(img_small).shape
    torch.Size([4, 128])
    """

    def __init__(
        self,
        in_channels: int = 1,
        channel_sizes: Sequence[int] = (16, 32, 64, 128),
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim

        # ------------------------------------------------------------------
        # Build convolutional blocks
        # ------------------------------------------------------------------
        blocks = []
        prev_ch = in_channels
        for ch in channel_sizes:
            blocks.append(_conv_block(prev_ch, ch))
            prev_ch = ch
        self.blocks = nn.ModuleList(blocks)

        # ------------------------------------------------------------------
        # Global spatial pooling -> collapses (H', W') to (1, 1)
        # ------------------------------------------------------------------
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ------------------------------------------------------------------
        # Projection head: last channel -> embedding_dim
        # ------------------------------------------------------------------
        self.head = nn.Sequential(
            nn.Linear(prev_ch, embedding_dim),
            nn.GELU(),
        )

        self._init_weights()
        logger.debug(
            "CNNBranch initialised: channels=%s embedding_dim=%d",
            channel_sizes, embedding_dim,
        )

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Kaiming-uniform init for Conv layers; Xavier for Linear."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of river-plot images.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, 1, H, W)`` -- single-channel river-plot image.
            *H* may range from 1 up to ~20 cycles; *W* should be 200.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, embedding_dim)`` -- image embedding.

        Raises
        ------
        ValueError
            If the input does not have exactly 4 dimensions.
        """
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4-D input (batch, C, H, W), got shape {x.shape}."
            )

        # Pass through convolutional blocks
        out = x
        for block in self.blocks:
            out = block(out)           # (B, C_i, H_i, W_i)

        # Global average pool -> (B, C_last, 1, 1)
        out = self.global_pool(out)

        # Flatten -> (B, C_last)
        out = out.flatten(1)

        # Project -> (B, embedding_dim)
        out = self.head(out)

        return out

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def num_params(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def feature_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return intermediate feature maps after each block (for visualisation).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, 1, H, W)``.

        Returns
        -------
        list[torch.Tensor]
            Feature map tensors after each convolutional block.
        """
        maps: list[torch.Tensor] = []
        out = x
        for block in self.blocks:
            out = block(out)
            maps.append(out.detach())
        return maps

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"CNNBranch(embedding_dim={self.embedding_dim}, "
            f"params={self.num_params():,})"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke-test the CNNBranch forward pass."
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--height", type=int, default=20,
                        help="Number of transit cycles (rows) in river plot.")
    parser.add_argument("--width", type=int, default=200,
                        help="Number of phase bins (columns).")
    parser.add_argument("--embedding_dim", type=int, default=128)
    args = parser.parse_args()

    branch = CNNBranch(embedding_dim=args.embedding_dim)
    branch.eval()

    dummy = torch.randn(args.batch_size, 1, args.height, args.width)
    emb = branch(dummy)
    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {emb.shape}")
    print(f"Branch       : {branch}")

    # Test variable height
    for h in [5, 10, 15, 20]:
        img = torch.randn(2, 1, h, 200)
        out = branch(img)
        print(f"  H={h:2d} -> output {out.shape}")
