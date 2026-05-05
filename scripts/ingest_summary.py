"""Aggregate per-mission stats from the interim Parquet store.

Produces a single Parquet report with one row per mission: row count, time
span, bus number, route(s) seen, simple value-range summaries. Used as a
dataset-level QC gate before cleaning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl
from loguru import logger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.interim_dir.rglob("*.parquet"))
    if not files:
        logger.error("No parquet files under {}", args.interim_dir)
        return 1

    logger.info("Aggregating {} mission parquets …", len(files))

    # Lazy scan so we never hold the full corpus in memory
    rows: list[dict] = []
    for f in files:
        lf = pl.scan_parquet(f)
        agg = lf.select(
            pl.lit(f.stem).alias("mission_id"),
            pl.col("time_iso").min().alias("t_start"),
            pl.col("time_iso").max().alias("t_end"),
            pl.col("time_iso").len().alias("n_rows"),
            pl.col("electric_powerDemand").mean().alias("power_mean_W"),
            pl.col("electric_powerDemand").min().alias("power_min_W"),
            pl.col("electric_powerDemand").max().alias("power_max_W"),
            pl.col("odometry_vehicleSpeed").mean().alias("speed_mean_mps"),
            pl.col("odometry_vehicleSpeed").max().alias("speed_max_mps"),
        ).collect()
        rows.append(agg.to_dicts()[0])

    summary = pl.DataFrame(rows).sort("t_start")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.write_parquet(args.out, compression="zstd")
    logger.success("Wrote ingest summary: {} rows → {}", summary.height, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
