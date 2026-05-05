"""GNSS coordinate cleaning.

CRITICAL: latitude and longitude are NOT interpolated across long gaps.
The bus might have gone through the Schimmelstrasse tunnel on route 31, or
under the Hardbrücke; inventing a straight-line trajectory across that gap
fabricates a route that downstream map-matching will treat as ground truth.

Per the ZTBus paper:

* On bus 183, ``gnss_course`` is held constant by the GNSS sensor when
  stationary.
* On bus 208, ``gnss_course`` is set to zero when stationary.

Both encodings are preserved and a per-bus ``gnss_course_valid`` flag is
emitted so downstream code knows when to trust the heading.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from loguru import logger

from ztbus.cleaning.config import GNSSConfig

LAT_COL = "gnss_latitude"
LON_COL = "gnss_longitude"
COURSE_COL = "gnss_course"


def clean_gnss(df: pl.DataFrame, cfg: GNSSConfig, *, bus: int) -> pl.DataFrame:
    """Apply short-gap-only interpolation to lat/lon; flag course validity."""
    df = _interpolate_short_gaps(df, LAT_COL, cfg.short_gap_max_s)
    df = _interpolate_short_gaps(df, LON_COL, cfg.short_gap_max_s)
    df = _annotate_course_validity(df, cfg, bus=bus)
    return df


def _interpolate_short_gaps(df: pl.DataFrame, col: str, short_gap_max_s: float) -> pl.DataFrame:
    if col not in df.columns:
        return df

    raw = df[col]
    if raw.null_count() == 0:
        return df

    interp = df.select(pl.col(col).interpolate())[col].to_numpy()
    raw_np = raw.to_numpy()

    # Time vector in seconds
    if "time_unix" in df.columns:
        t = df["time_unix"].to_numpy().astype(float)
    else:
        t = np.arange(df.height, dtype=float)

    out = raw_np.copy()
    is_null = np.array([v is None or (isinstance(v, float) and np.isnan(v)) for v in raw_np.tolist()])

    in_run = False
    run_start = 0
    for i in range(df.height):
        if is_null[i] and not in_run:
            run_start = i
            in_run = True
        elif not is_null[i] and in_run:
            in_run = False
            duration = t[i - 1] - t[run_start] if i > run_start else 0
            if duration <= short_gap_max_s:
                out[run_start:i] = interp[run_start:i]
    if in_run:
        duration = t[-1] - t[run_start]
        if duration <= short_gap_max_s:
            out[run_start:] = interp[run_start:]

    return df.with_columns(pl.Series(col, out))


def _annotate_course_validity(df: pl.DataFrame, cfg: GNSSConfig, *, bus: int) -> pl.DataFrame:
    """Emit ``gnss_course_valid`` per the per-bus convention from [W23]."""
    if COURSE_COL not in df.columns:
        return df.with_columns(pl.lit(False).alias("gnss_course_valid"))

    speed_col = "speed_smoothed_mps" if "speed_smoothed_mps" in df.columns else "odometry_vehicleSpeed"
    if speed_col not in df.columns:
        return df.with_columns(pl.lit(True).alias("gnss_course_valid"))

    moving = pl.col(speed_col) > 0.5

    if bus == 183:
        # Course is held constant when stationary; valid only while moving
        valid = moving
    elif bus == 208:
        # Course set to zero when stationary; valid only while moving
        valid = moving & (pl.col(COURSE_COL) != 0.0)
    else:
        logger.warning("Unknown bus number {}; conservatively treating course as valid only while moving", bus)
        valid = moving

    return df.with_columns(valid.alias("gnss_course_valid"))
