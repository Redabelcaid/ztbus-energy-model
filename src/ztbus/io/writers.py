"""Writers for the interim and processed Parquet stores.

Storage layout (all under ``data/interim`` or ``data/processed``)::

    bus={bus}/year={yyyy}/month={mm}/{mission_id}.parquet

Hive-style partitioning lets DuckDB / Polars push down filters efficiently
(e.g., "all winter missions on bus 183") without scanning every file.

Compression: ZSTD level 3. On 10 GB of CSV the parquet store typically lands
around 1.0 - 1.5 GB.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from loguru import logger

PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 3
PARQUET_ROW_GROUP_SIZE = 100_000   # ~30s of 1Hz data fits in a row group nicely


def mission_partition_path(root: Path, *, bus: int, start_utc, mission_id: str) -> Path:
    """Compute the on-disk partitioned path for a single mission."""
    return (
        root
        / f"bus={bus}"
        / f"year={start_utc.year}"
        / f"month={start_utc.month:02d}"
        / f"{mission_id}.parquet"
    )


def write_mission_parquet(
    df: pl.DataFrame,
    *,
    root: Path,
    bus: int,
    start_utc,
    mission_id: str,
    overwrite: bool = False,
) -> Path:
    """Write a single mission to its partitioned Parquet path. Returns the path."""
    out = mission_partition_path(root, bus=bus, start_utc=start_utc, mission_id=mission_id)
    if out.exists() and not overwrite:
        logger.debug("Skipping existing parquet: {}", out)
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(
        out,
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_COMPRESSION_LEVEL,
        row_group_size=PARQUET_ROW_GROUP_SIZE,
        statistics=True,
    )
    logger.debug("Wrote {} rows → {}", df.height, out)
    return out
