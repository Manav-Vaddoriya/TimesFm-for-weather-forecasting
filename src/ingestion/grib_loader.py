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
    """Open a GRIB file lazily as an xarray Dataset."""
    logger.info("Opening GRIB file: %s", file_path)
    return xr.open_dataset(
        file_path,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""},
    )


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
    """
    logger.info("Extracting year %d from dataset", year)
    ds_year = ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
    return ds_year.to_dataframe().reset_index()