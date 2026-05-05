"""Altitude cleaning.

GNSS altitude is the noisiest signal in the dataset and a critical input to
the road-grade calculation. The cleaning policy:

1. Linear-interpolate ONLY across short gaps (≤ ``short_gap_max_s``).
   Long gaps (tunnels, urban canyons) are left as NaN — inventing topology
   creates fake slopes downstream.
2. Apply a strong rolling median to suppress GNSS jitter without
   eliminating real elevation structure.

Adds:

- ``altitude_smoothed_m``: float, smoothed altitude (NaN where data is missing
  beyond the short-gap window).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ztbus.cleaning.config import AltitudeConfig

RAW_COL = "gnss_altitude"
SMOOTHED_COL = "altitude_smoothed_m"


def clean_altitude(df: pl.DataFrame, cfg: AltitudeConfig) -> pl.DataFrame:
    """Interpolate short gaps, smooth strongly. Long gaps stay NaN."""
    if RAW_COL not in df.columns:
        return df

    window = max(5, int(round(cfg.smoothing.window_seconds)))
    if window % 2 == 0:
        window += 1

    # Polars' `interpolate` fills ALL gaps; we restrict the fill to short gaps
    # by computing a per-row eligibility mask first.
    mask = _short_gap_mask(df, cfg.short_gap_max_s)

    interpolated = df.select(pl.col(RAW_COL).interpolate())[RAW_COL]
    raw = df[RAW_COL]
    # Use interpolated value only where mask is True; otherwise keep raw (NaN)
    short_gap_filled = pl.Series(
        SMOOTHED_COL + "_pre",
        np.where(mask.to_numpy(), interpolated.to_numpy(), raw.to_numpy()),
    )
    df = df.with_columns(short_gap_filled)

    # Rolling median for smoothing
    df = df.with_columns(
        pl.col(SMOOTHED_COL + "_pre")
            .rolling_median(window_size=window, min_samples=1, center=True)
            .alias(SMOOTHED_COL)
    ).drop(SMOOTHED_COL + "_pre")

    return df


def _short_gap_mask(df: pl.DataFrame, short_gap_max_s: float) -> pl.Series:
    """Return a boolean Series: True where this row is inside a short gap."""
    raw = df[RAW_COL]
    is_null = raw.is_null()
    if not is_null.any():
        return pl.Series("ok", [True] * df.height)

    # Use time_unix to measure actual gap length; falls back to row counting.
    if "time_unix" in df.columns and df["time_unix"].null_count() == 0:
        time_s = df["time_unix"].to_numpy().astype(float)
    else:
        time_s = np.arange(df.height, dtype=float)

    # For each null run, compute its duration. Mark rows in short runs as True.
    is_null_np = is_null.to_numpy()
    ok = np.ones(df.height, dtype=bool)

    # Identify contiguous null runs
    in_run = False
    run_start = 0
    for i, missing in enumerate(is_null_np):
        if missing and not in_run:
            run_start = i
            in_run = True
        elif not missing and in_run:
            in_run = False
            duration = time_s[i - 1] - time_s[run_start] if i > run_start else 0
            if duration > short_gap_max_s:
                ok[run_start:i] = False
    if in_run:  # tail run
        duration = time_s[-1] - time_s[run_start]
        if duration > short_gap_max_s:
            ok[run_start:] = False

    return pl.Series("ok", ok)
