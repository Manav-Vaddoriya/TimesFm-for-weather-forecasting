"""
Pure DataFrame transformations:
  - Drop irrelevant ERA5 columns
  - Convert temperature from Kelvin to Celsius
  - Downcast dtypes to save memory
  - Assign a "<lat, lon>" series_id to every row
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)

_DROP_COLUMNS = ["number", "step", "surface", "valid_time"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop ERA5 metadata columns, rename t2m → temperature,
    convert Kelvin → Celsius, and downcast numeric dtypes.
    """
    df = df.drop(columns=_DROP_COLUMNS, errors="ignore")
    df = df.rename(columns={"t2m": "temperature"})

    df["temperature"] = (df["temperature"] - 273.15).astype(settings.temp_dtype)
    df["latitude"] = df["latitude"].astype(settings.coord_dtype)
    df["longitude"] = df["longitude"].astype(settings.coord_dtype)

    logger.debug("Cleaned DataFrame — shape: %s", df.shape)
    return df


def assign_series_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a '<lat, lon>' string identifier for each unique grid point.
    Processed in batches to keep peak memory bounded.
    """
    batch_size = settings.series_id_batch_size
    n = len(df)
    series_id = pd.array([None] * n, dtype=object)
    decimals = settings.coord_round_decimals

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        lat = df["latitude"].iloc[start:end].round(decimals).astype(str)
        lon = df["longitude"].iloc[start:end].round(decimals).astype(str)
        series_id[start:end] = ("<" + lat + ", " + lon + ">").values

    df = df.copy()
    df["series_id"] = series_id
    df = df.drop(columns=["latitude", "longitude"])

    logger.debug("Assigned series_id — unique IDs: %d", df["series_id"].nunique())
    return df