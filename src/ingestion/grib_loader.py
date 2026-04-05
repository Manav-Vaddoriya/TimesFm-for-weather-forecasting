"""
src/ingestion/grib_loader.py

Responsible only for opening a GRIB file and slicing it by year.
No transformation logic lives here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def open_grib(file_path: Path) -> xr.Dataset:
    """
    Open a GRIB file lazily as an xarray Dataset.

    Raises:
        FileNotFoundError: If the GRIB file does not exist.
        RuntimeError: If the file cannot be opened.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"GRIB file not found: {file_path}")

    logger.info("Opening GRIB file: %s", file_path)
    try:
        return xr.open_dataset(
            file_path,
            engine="cfgrib",
            backend_kwargs={"indexpath": ""},
        )
    except Exception as e:
        logger.error("Failed to open GRIB file '%s': %s", file_path, e, exc_info=True)
        raise RuntimeError(f"Cannot open GRIB file: {file_path}") from e


def slice_year(ds: xr.Dataset, year: int) -> pd.DataFrame:
    """
    Slice the dataset to a single calendar year and materialise as a DataFrame.

    Parameters
    ----------
    ds:   Full multi-year xarray Dataset.
    year: The year to extract.

    Returns
    -------
    Raw DataFrame with all original columns intact.

    Raises:
        ValueError: If the year yields an empty DataFrame.
    """
    logger.info("Extracting year %d from dataset", year)
    try:
        ds_year = ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
        df = ds_year.to_dataframe().reset_index()

        if df.empty:
            raise ValueError(f"No data found for year {year}")

        logger.info("Year %d extracted — %d rows", year, len(df))
        return df

    except Exception as e:
        logger.error("Failed to extract year %d: %s", year, e, exc_info=True)
        raise