"""Timestamp-level cleaning.

Operations are conservative: we never invent timestamps, and we never silently
drop them. Real irregularities are kept and FLAGGED so that downstream code
(integration, differentiation) can decide how to react.

Adds these columns:

- ``dt_s``: time delta to the previous row (NaN for first row).
- ``time_gap_flag``: True where ``dt_s`` exceeds ``max_internal_gap_s``.
"""

from __future__ import annotations

import polars as pl
from loguru import logger

from ztbus.cleaning.config import TimestampConfig


class TimestampQualityError(ValueError):
    """Raised when a mission's timestamps are unrecoverable (e.g. non-monotonic)."""


def clean_timestamps(df: pl.DataFrame, cfg: TimestampConfig) -> pl.DataFrame:
    """Validate and annotate timestamps.

    Drops duplicate ``time_iso`` rows (keeping first), validates monotonicity,
    and adds ``dt_s`` and ``time_gap_flag`` columns.

    Raises
    ------
    TimestampQualityError
        If timestamps are non-monotonic and ``cfg.reject_on_non_monotonic_time``
        is True. This is a strong signal of corrupted data.
    """
    if df.is_empty():
        return df

    n_before = df.height
    df = df.unique(subset=["time_iso"], keep="first").sort("time_iso")
    n_after = df.height
    if n_before != n_after:
        logger.debug("Removed {} duplicate-timestamp rows", n_before - n_after)

    # Monotonicity check on time_unix (more reliable than ISO strings)
    diffs = df["time_unix"].diff().drop_nulls()
    if (diffs < 0).any():
        msg = f"Non-monotonic time_unix detected (min diff = {diffs.min()})"
        if cfg.reject_on_non_monotonic_time:
            raise TimestampQualityError(msg)
        logger.warning(msg)

    df = df.with_columns(
        pl.col("time_unix").diff().cast(pl.Float64).alias("dt_s"),
    ).with_columns(
        (pl.col("dt_s") > cfg.max_internal_gap_s).alias("time_gap_flag"),
    )
    return df
