"""
End-to-end data preparation orchestration:
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
import sys
import time
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

    try:
        ds = open_grib(settings.grib_file)
    except Exception as e:
        logger.error("Failed to open GRIB file '%s': %s", settings.grib_file, e, exc_info=True)
        raise

    for year in settings.years:
        try:
            if is_year_processed(year, output_dir):
                logger.info("Year %d already processed — skipping.", year)
                continue

            year_start = time.time()
            logger.info("Processing year %d", year)

            df = slice_year(ds, year)
            df = clean(df)
            df = assign_series_id(df)

            write_year_chunks(df, year, output_dir)
            build_series_id_index(year, output_dir)

            del df
            gc.collect()

            elapsed = time.time() - year_start
            logger.info("Year %d done (%.1fs).\n", year, elapsed)

        except Exception as e:
            logger.error(
                "Failed to process year %d: %s — skipping to next year.",
                year, e, exc_info=True,
            )
            continue

    logger.info("Stage 1 complete.\n")


# ── Stage 2: Patch generation ────────────────────────────────────────────────

def run_patch_stage(output_dir: Path) -> None:
    """Convert every year's parquet chunks into patch arrays."""
    logger.info("=== Stage 2: Patch generation ===")

    for year in settings.years:
        try:
            year_dir = output_dir / f"{year}_weather_data"
            if not year_dir.exists():
                logger.warning("Year %d folder missing — skipping patches.", year)
                continue

            generate_patches(year, output_dir)

        except Exception as e:
            logger.error(
                "Failed to generate patches for year %d: %s — skipping.",
                year, e, exc_info=True,
            )
            continue

    logger.info("Stage 2 complete.\n")


# ── Entry point ──────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    """Run the complete data preparation pipeline."""
    pipeline_start = time.time()
    output_dir = settings.storage.output_dir

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error("Cannot create output directory '%s': %s", output_dir, e)
        sys.exit(1)

    try:
        run_raw_stage(output_dir)
    except Exception as e:
        logger.error("Raw data stage failed critically: %s", e, exc_info=True)
        sys.exit(1)

    run_patch_stage(output_dir)

    elapsed = time.time() - pipeline_start
    logger.info("Pipeline complete (%.1fs). All outputs saved to: %s", elapsed, output_dir)


if __name__ == "__main__":
    run_pipeline()