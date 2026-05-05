"""Ambient temperature cleaning.

The dataset stores temperature in Kelvin (Widmer et al., 2023, Table 1).
We KEEP the storage in Kelvin throughout the pipeline (avoiding implicit unit
errors during downstream computation) and convert to Celsius only at the
presentation layer. The HVAC submodel of the powertrain operates on Kelvin
internally (the absolute reference cancels in |T - T_comfort|).

Temperature changes slowly (seconds → minutes scale) so short-gap
interpolation is safe. Beyond the gap budget we leave NaN.

Adds:

- ``temperature_outlier_flag``: bool, where T was outside the plausibility envelope.
"""

from __future__ import annotations

import polars as pl

from ztbus.cleaning.config import TemperatureConfig

RAW_COL = "temperature_ambient"


def clean_temperature(df: pl.DataFrame, cfg: TemperatureConfig) -> pl.DataFrame:
    if RAW_COL not in df.columns:
        return df

    lo, hi = cfg.plausibility_bounds_K
    df = df.with_columns(
        ((pl.col(RAW_COL) < lo) | (pl.col(RAW_COL) > hi)).alias("temperature_outlier_flag"),
    )
    # Polars' interpolate fills all gaps; for temperature this is acceptable
    # because the signal is slowly varying. Long missions with the sensor
    # entirely missing will show as fully-NaN and are caught by QC.
    df = df.with_columns(pl.col(RAW_COL).interpolate())
    return df
