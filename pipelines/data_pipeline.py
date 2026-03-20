"""
End-to-end orchestration:
  Stage 1 — Raw data:  GRIB → clean parquet chunks + series_id index
  Stage 2 — Patches:   parquet chunks → patch .npy + metadata parquet

All outputs are saved to the configured output_dir.

Run with:
    python -m pipelines.data_pipeline
or import and call run_pipeline() programmatically.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

from config.settings import settings
from src.ingestion.grib_loader import open_grib, slice_year
from src.processing.transforming import assign_series_id, clean
from src.processing.writer import (
    build_series_id_index,
    is_year_processed,
    write_year_chunks,
)
from src.patches.patch_prep import process_year as generate_patches

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Stage 1: Raw weather data ────────────────────────────────────────────────

def run_raw_stage(output_dir: Path) -> None:
    """Load GRIB, process each year, write parquet chunks, build indexes."""
    logger.info("=== Stage 1: Raw data preparation ===")
    ds = open_grib(settings.grib_file)

    for year in settings.years:
        if is_year_processed(year, output_dir):
            logger.info("Year %d already processed — skipping.", year)
            continue

        logger.info("Processing year %d", year)
        df = slice_year(ds, year)
        df = clean(df)
        df = assign_series_id(df)

        write_year_chunks(df, year, output_dir)
        build_series_id_index(year, output_dir)

        del df
        gc.collect()
        logger.info("Year %d done.\n", year)

    logger.info("Stage 1 complete.\n")


# ── Stage 2: Patch generation ────────────────────────────────────────────────

def run_patch_stage(output_dir: Path) -> None:
    """Convert every year's parquet chunks into patch arrays."""
    logger.info("=== Stage 2: Patch generation ===")
    for year in settings.years:
        year_dir = output_dir / f"{year}_weather_data"
        if not year_dir.exists():
            logger.warning("Year %d folder missing — skipping patches.", year)
            continue
        generate_patches(year, output_dir)
    logger.info("Stage 2 complete.\n")


# ── Entry point ──────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    output_dir = settings.storage.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_raw_stage(output_dir)
    run_patch_stage(output_dir)

    logger.info("Pipeline complete. All outputs saved to: %s", output_dir)


if __name__ == "__main__":
    run_pipeline()