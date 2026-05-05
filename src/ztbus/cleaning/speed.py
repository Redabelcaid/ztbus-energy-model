"""Speed signal cleaning.

The dataset paper notes that ``odometry_vehicleSpeed`` is slightly negative
near stops; this is a sensor artifact, not real motion. We clamp small
negatives to zero and FLAG large negatives or improbable highs rather than
silently overwriting them.

We also produce a smoothed speed signal for use as the basis of acceleration
(differentiating raw speed amplifies noise into the model).

Adds:

- ``speed_negative_flag``: bool, True where raw speed < small_negative_threshold.
- ``speed_outlier_flag``: bool, True where raw speed > upper_plausibility.
- ``speed_smoothed_mps``: float, raw → clamped → rolling-median → rolling-mean.
"""

from __future__ import annotations

import polars as pl

from ztbus.cleaning.config import SpeedConfig

RAW_COL = "odometry_vehicleSpeed"
SMOOTHED_COL = "speed_smoothed_mps"


def clean_speed(df: pl.DataFrame, cfg: SpeedConfig) -> pl.DataFrame:
    """Clamp small negatives, flag outliers, produce a smoothed speed."""
    if RAW_COL not in df.columns:
        return df

    # Window in samples ≈ window_seconds × sample rate. ZTBus is 1 Hz nominal,
    # so window_seconds ≈ samples. We keep an odd window for centered medians.
    window = max(3, round(cfg.smoothing.window_seconds))
    if window % 2 == 0:
        window += 1

    df = df.with_columns(
        # Flags first, on RAW data
        (pl.col(RAW_COL) < cfg.small_negative_threshold_mps).alias("speed_negative_flag"),
        (pl.col(RAW_COL) > cfg.upper_plausibility_mps).alias("speed_outlier_flag"),
    )

    # Clamp small negatives to zero, leave the rest alone
    df = df.with_columns(
        pl.when(pl.col(RAW_COL) < 0)
        .then(
            pl.when(pl.col(RAW_COL) >= cfg.small_negative_threshold_mps)
            .then(pl.lit(0.0))
            .otherwise(pl.col(RAW_COL))
        )  # keep the value, but it's flagged
        .otherwise(pl.col(RAW_COL))
        .alias("_speed_clamped"),
    )

    # Smooth: rolling median (kills spikes) → rolling mean (residual smoothing)
    df = df.with_columns(
        pl.col("_speed_clamped")
        .rolling_median(window_size=window, min_samples=1, center=True)
        .rolling_mean(window_size=window, min_samples=1, center=True)
        .alias(SMOOTHED_COL),
    ).drop("_speed_clamped")

    return df
