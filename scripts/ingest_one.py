"""Ingest a single ZTBus CSV into the partitioned Parquet store.

Used as a standalone entry point so that Snakemake can call one task per
mission without going through the Typer CLI (which adds a small startup cost).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Path to one B*.csv")
    parser.add_argument("--interim-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from ztbus.io import (
        parse_mission_filename,
        read_mission_csv,
        write_mission_parquet,
    )

    meta = parse_mission_filename(args.csv)
    df = read_mission_csv(args.csv)
    out = write_mission_parquet(
        df,
        root=args.interim_dir,
        bus=meta["bus"],
        start_utc=meta["start_utc"],
        mission_id=meta["mission_id"],
        overwrite=args.overwrite,
    )
    logger.info("Ingested {} rows: {} → {}", df.height, args.csv.name, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
