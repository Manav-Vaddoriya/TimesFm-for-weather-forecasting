"""
Converts raw time-series chunks into fixed-length patch arrays
suitable for transformer-based models.

Each series is split into non-overlapping windows of `patch_size`
consecutive time steps. The final window is zero-padded if the series
length is not an exact multiple of patch_size.

Outputs per input parquet file:
  - <stem>_patches.npy      — float32 array (num_patches, patch_size)
  - <stem>_metadata.parquet — DataFrame mapping patch_idx → series_id
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


# ── core patch logic ──────────────────────────────────────────────────────────

def _make_patches(series: np.ndarray, patch_size: int, pad_value: float) -> np.ndarray:
    """
    Split a 1-D array into a 2-D array of shape (n_patches, patch_size).
    The last patch is right-padded with `pad_value` if necessary.
    """
    n = len(series)
    if n == 0:
        return np.empty((0, patch_size), dtype=np.float32)

    n_patches = int(np.ceil(n / patch_size))
    pad_len = n_patches * patch_size - n

    if pad_len > 0:
        series = np.pad(series, (0, pad_len), mode="constant", constant_values=pad_value)

    return series.reshape(n_patches, patch_size)


# ── file-level processor ──────────────────────────────────────────────────────

def process_parquet_file(file_path: Path, output_dir: Path) -> int:
    """
    Read one parquet chunk, generate patches for every series_id,
    and save the patch array + metadata to `output_dir`.

    Returns the total number of patches saved.
    """
    patch_size = settings.patches.patch_size
    pad_value = settings.patches.pad_value

    df = pd.read_parquet(file_path)

    required = {"series_id", "temperature"}
    missing = required - set(df.columns)
    if missing:
        logger.warning("Skipping %s — missing columns: %s", file_path.name, missing)
        return 0

    if "time" in df.columns:
        df = df.sort_values(["series_id", "time"], kind="mergesort")
    else:
        logger.warning("'time' column missing in %s — sorting by series_id only.", file_path.name)
        df = df.sort_values("series_id", kind="mergesort")

    all_patches: list[np.ndarray] = []
    metadata_rows: list[dict] = []
    global_patch_idx = 0

    for sid, group in df.groupby("series_id", sort=False):
        ts = group["temperature"].to_numpy(dtype=np.float32)
        patches = _make_patches(ts, patch_size, pad_value)

        if patches.shape[0] == 0:
            continue

        all_patches.append(patches)
        for _ in range(patches.shape[0]):
            metadata_rows.append({"series_id": sid, "patch_idx": global_patch_idx})
            global_patch_idx += 1

    if not all_patches:
        logger.warning("No valid patches produced from %s.", file_path.name)
        return 0

    patch_array = np.vstack(all_patches)
    metadata_df = pd.DataFrame(metadata_rows)

    stem = file_path.stem
    np.save(output_dir / f"{stem}_patches.npy", patch_array)
    metadata_df.to_parquet(output_dir / f"{stem}_metadata.parquet", index=False)

    logger.info(
        "Patched %s → %d patches saved to %s", file_path.name, patch_array.shape[0], output_dir
    )
    return patch_array.shape[0]


# ── year-level orchestration ──────────────────────────────────────────────────

def process_year(year: int, staging_dir: Path) -> None:
    """
    Generate patches for all parquet chunks belonging to `year`.
    Output is written to <staging_dir>/<year>_weather_data/patches/.
    """
    year_dir = staging_dir / f"{year}_weather_data"
    output_dir = year_dir / "patches"
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(year_dir.glob("weather_part_*.parquet"))
    if not parquet_files:
        logger.warning("No parquet files found for year %d — skipping patch generation.", year)
        return

    total = 0
    for file_path in parquet_files:
        total += process_parquet_file(file_path, output_dir)

    logger.info("Year %d complete — %d total patches written.", year, total)