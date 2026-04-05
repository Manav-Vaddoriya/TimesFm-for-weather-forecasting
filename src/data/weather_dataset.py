"""
PyTorch Dataset for weather patch data.

Loads .npy patch files and builds a sliding-window index so that
each sample is a (context, target) pair of consecutive patch sequences.

Example:
    If context_len = 32 and a file has 1000 patches, the dataset yields
    968 samples.  For sample i:
        x = patches[i : i + 32]      — input context
        y = patches[i + 1 : i + 33]  — target (shifted by 1)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class WeatherPatchDataset(Dataset):
    """
    Dataset that loads pre-computed .npy patch files and serves
    sliding-window (context → target) samples for autoregressive training.

    Args:
        file_list : list of paths to .npy files, each shaped (N_patches, patch_len).
        context_len : number of past patches used as input context.
    """

    def __init__(self, file_list: List[str | Path], context_len: int = 32) -> None:
        self.context_len = context_len
        self.data_chunks: list[torch.Tensor] = []
        self.samples: list[Tuple[int, int]] = []

        if not file_list:
            logger.warning("WeatherPatchDataset: No files provided — dataset is empty.")
            return

        logger.info("Loading %d patch file(s)…", len(file_list))

        for f in file_list:
            try:
                arr = np.load(f)  # shape (N_patches, patch_len)
                logger.info("  Loaded %s: shape=%s", os.path.basename(str(f)), arr.shape)

                if arr.ndim != 2:
                    logger.warning("  Skipping %s — expected 2D array, got shape %s", f, arr.shape)
                    continue

                self.data_chunks.append(torch.from_numpy(arr.astype(np.float32)))

            except Exception as e:
                logger.error("  Error loading %s: %s", f, e)

        # Build sample index: (chunk_idx, start_row)
        seq_len = context_len + 1  # context + 1 target patch

        for chunk_idx, chunk in enumerate(self.data_chunks):
            n = chunk.shape[0]
            if n < seq_len:
                logger.warning(
                    "  Chunk %d has only %d patches (need %d) — skipping.",
                    chunk_idx, n, seq_len,
                )
                continue

            for start in range(n - seq_len + 1):
                self.samples.append((chunk_idx, start))

        logger.info("Total samples: %d", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk_idx, start_row = self.samples[idx]
        chunk = self.data_chunks[chunk_idx]

        window = chunk[start_row : start_row + self.context_len + 1]
        x = window[:-1]  # context patches
        y = window[1:]   # target patches (shifted by 1)
        return x, y
