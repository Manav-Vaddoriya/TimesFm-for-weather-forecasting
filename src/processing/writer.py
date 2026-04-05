"""
Handles all local Parquet I/O for the raw weather data stage:
  - Write a year's DataFrame to N parquet chunk files
  - Build a series_id → filename lookup index
  - Check whether a year has already been processed (idempotency guard)
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────

def year_output_dir(year: int, staging_dir: Path) -> Path:
    return staging_dir / f"{year}_weather_data"


def is_year_processed(year: int, staging_dir: Path) -> bool:
    """Return True if parquet chunks for this year already exist."""
    out_dir = year_output_dir(year, staging_dir)
    if not out_dir.exists():
        return False
    return len(list(out_dir.glob("weather_part_*.parquet"))) > 0


# ── writers ──────────────────────────────────────────────────────────────────

def write_year_chunks(df: pd.DataFrame, year: int, staging_dir: Path) -> None:
    """
    Split the year's DataFrame evenly across N parquet files,
    grouped by series_id so each series stays in one file.

    Raises:
        OSError: If the output directory cannot be created or files cannot be written.
    """
    out_dir = year_output_dir(year, staging_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error("Cannot create output directory '%s': %s", out_dir, e)
        raise

    n_files = settings.raw_chunks_per_year
    series_ids = df["series_id"].unique()
    chunks = np.array_split(series_ids, n_files)

    for idx, chunk_ids in enumerate(chunks, start=1):
        try:
            chunk_df = df[df["series_id"].isin(chunk_ids)]
            out_path = out_dir / f"weather_part_{idx}.parquet"
            chunk_df.to_parquet(out_path, index=False)
            logger.info(
                "Saved chunk %d/%d → %s (%d rows)", idx, n_files, out_path, len(chunk_df)
            )
            del chunk_df
            gc.collect()
        except Exception as e:
            logger.error("Failed to write chunk %d for year %d: %s", idx, year, e, exc_info=True)
            raise


def build_series_id_index(year: int, staging_dir: Path) -> None:
    """
    Scan all chunk files for the given year and persist a
    series_id → file_name lookup table as a parquet index.
    """
    out_dir = year_output_dir(year, staging_dir)
    parquet_files = sorted(out_dir.glob("weather_part_*.parquet"))

    if not parquet_files:
        logger.warning("No parquet files found for year %d — skipping index build.", year)
        return

    records = []
    for file_path in parquet_files:
        try:
            logger.debug("Scanning %s for series IDs", file_path.name)
            ids = pd.read_parquet(file_path, columns=["series_id"])["series_id"].unique()
            records.extend({"series_id": sid, "file_name": file_path.name} for sid in ids)
        except Exception as e:
            logger.error("Failed to scan %s: %s", file_path.name, e, exc_info=True)
            continue

    if not records:
        logger.warning("No series IDs found for year %d.", year)
        return

    try:
        index_df = pd.DataFrame(records)
        index_path = out_dir / "series_id_index.parquet"
        index_df.to_parquet(index_path, index=False)
        logger.info("Series ID index saved → %s (%d entries)", index_path, len(index_df))
    except Exception as e:
        logger.error("Failed to save series ID index for year %d: %s", year, e, exc_info=True)
        raise


def lookup_series_file(year: int, series_id: str, staging_dir: Path) -> str | None:
    """
    Return the filename that contains the given series_id for a year,
    or None if not found.
    """
    index_path = year_output_dir(year, staging_dir) / "series_id_index.parquet"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Series ID index not found for year {year}. Run build_series_id_index first."
        )

    try:
        index_df = pd.read_parquet(index_path)
        match = index_df[index_df["series_id"] == series_id]
        if match.empty:
            logger.warning("series_id '%s' not found in year %d index.", series_id, year)
            return None
        return match.iloc[0]["file_name"]
    except Exception as e:
        logger.error("Error looking up series '%s' for year %d: %s", series_id, year, e, exc_info=True)
        raise