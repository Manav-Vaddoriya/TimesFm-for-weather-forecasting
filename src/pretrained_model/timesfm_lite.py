"""
TimesFM-Lite: Weather Forecasting Foundation Model

A decoder-only Transformer adapted from Google's TimesFM architecture
for efficient training on ERA5 temperature data.

Architecture:
    1. Patch Embedding  — projects a patch of size P to vector of size D
    2. Positional Encoding — learnable position embeddings
    3. Causal Transformer — stacked encoder layers with causal masking
    4. RMSNorm + Output Head — projects D back to patch size P
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm + self.eps)
        return self.scale * x_normed


class TimesFMLiteGPT(nn.Module):
    """
    Decoder-only (GPT-style) Transformer for time-series patch prediction.

    Uses PyTorch TransformerEncoder with causal masking — functionally
    equivalent to a decoder-only model.

    Parameters (via config dict):
        patch_len   : int   — length of each input patch (time steps)
        d_model     : int   — hidden dimension
        n_layers    : int   — number of Transformer blocks
        n_heads     : int   — number of attention heads
        d_ff        : int   — feed-forward dimension
        dropout     : float — dropout rate
        context_len : int   — max number of patches to look back
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = nn.Linear(config["patch_len"], config["d_model"])
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config["context_len"], config["d_model"])
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["d_model"],
            nhead=config["n_heads"],
            dim_feedforward=config["d_ff"],
            dropout=config["dropout"],
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            encoder_layer, num_layers=config["n_layers"]
        )

        self.norm_f = RMSNorm(config["d_model"])
        self.head = nn.Linear(config["d_model"], config["patch_len"])

        logger.info(
            "TimesFMLiteGPT initialized — d_model=%d, n_layers=%d, n_heads=%d, "
            "d_ff=%d, context_len=%d, patch_len=%d",
            config["d_model"], config["n_layers"], config["n_heads"],
            config["d_ff"], config["context_len"], config["patch_len"],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Seq_Len, Patch_Len]

        Returns:
            out: [Batch, Seq_Len, Patch_Len]  — predicted next-patches
        """
        B, T, P = x.shape

        # 1. Patch embedding
        h = self.patch_embed(x)  # [B, T, D]

        # 2. Add positional encoding
        h = h + self.pos_embed[:, :T, :]

        # 3. Causal mask — position t can only attend to 0…t
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)

        # 4. Transformer blocks
        h = self.blocks(h, mask=mask, is_causal=True)

        # 5. Final norm + output projection
        h = self.norm_f(h)
        out = self.head(h)
        return out

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
