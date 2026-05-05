"""Exploratory data analysis: mission-level and dataset-level statistics.

The output is a tidy Parquet table with one row per mission, suitable for
both global summary plots and quick "which mission is weird?" queries via
DuckDB. We deliberately compute summaries via Polars expressions on the
interim store rather than loading any single mission to RAM, so the same
function works on a laptop and on the cluster.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from loguru import logger


def profile_mission(lf: pl.LazyFrame, mission_id: str, bus: int) -> dict:
    """Compute a per-mission summary as a dict of scalars.

    Uses lazy evaluation so the file is read once. The output is a
    self-describing dict that downstream code can collect into a DataFrame.
    """
    aggs = (
        lf.select(
            # Row counts and time
            pl.len().alias("n_rows"),
            pl.col("time_unix").min().alias("t_start_unix"),
            pl.col("time_unix").max().alias("t_end_unix"),
            # Power
            pl.col("electric_powerDemand").mean().alias("power_mean_W"),
            pl.col("electric_powerDemand").min().alias("power_min_W"),
            pl.col("electric_powerDemand").max().alias("power_max_W"),
            pl.col("electric_powerDemand").std().alias("power_std_W"),
            (pl.col("electric_powerDemand") < 0).mean().alias("frac_regen"),
            # Speed
            pl.col("odometry_vehicleSpeed").mean().alias("speed_mean_mps"),
            pl.col("odometry_vehicleSpeed").max().alias("speed_max_mps"),
            (pl.col("odometry_vehicleSpeed") < 0).sum().alias("n_negative_speed"),
            (pl.col("odometry_vehicleSpeed") > 25).sum().alias("n_speed_over_25"),
            # Temperature
            pl.col("temperature_ambient").mean().alias("temperature_mean_K"),
            pl.col("temperature_ambient").null_count().alias("temperature_n_null"),
            # Passengers
            pl.col("itcs_numberOfPassengers").mean().alias("passengers_mean"),
            pl.col("itcs_numberOfPassengers").max().alias("passengers_max"),
            # GNSS
            pl.col("gnss_altitude").null_count().alias("altitude_n_null"),
            pl.col("gnss_latitude").null_count().alias("gnss_n_null"),
            # Routes
            pl.col("itcs_busRoute").drop_nulls().n_unique().alias("n_distinct_routes"),
        )
        .collect()
        .to_dicts()[0]
    )

    aggs["mission_id"] = mission_id
    aggs["bus"] = bus
    aggs["duration_h"] = (aggs["t_end_unix"] - aggs["t_start_unix"]) / 3600.0

    return aggs


def profile_corpus(interim_dir: Path, *, out_path: Path) -> pl.DataFrame:
    """Profile every mission in the interim Parquet store."""
    files = sorted(interim_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {interim_dir}")
    logger.info("Profiling {} missions", len(files))

    rows = []
    for f in files:
        # Bus number lives in the partition path: bus=183/year=...
        try:
            bus = int(next(part.split("=")[1] for part in f.parts if part.startswith("bus=")))
        except StopIteration:
            bus = -1
        rows.append(profile_mission(pl.scan_parquet(f), mission_id=f.stem, bus=bus))

    df = pl.DataFrame(rows).sort("t_start_unix")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    logger.success("Wrote dataset profile: {} missions → {}", df.height, out_path)
    return df
