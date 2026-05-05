"""Readers for the raw ZTBus CSV files.

The dataset ships as CSV with UTF-8 encoding (Widmer et al., 2023, § Usage Notes).
We read with Polars rather than pandas:

* Polars uses Apache Arrow as its in-memory format, so the conversion to
  Parquet (our chosen on-disk format) is zero-copy.
* The lazy/streaming API processes files larger than RAM without partitioning.
* Schema enforcement is first-class via :class:`polars.Schema`.

This module performs *only* I/O and minimal type coercion. Every cleaning
decision lives in :mod:`ztbus.cleaning` so that raw data and cleaning policy
can be swapped independently.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import polars as pl
from loguru import logger

# Filename pattern from the ZTBus dataset (e.g. B183_2019-10-16_02-52-43_2019-10-16_07-10-12.csv)
_FILENAME_RE: Final = re.compile(
    r"^B(?P<bus>\d+)_"
    r"(?P<start_date>\d{4}-\d{2}-\d{2})_(?P<start_time>\d{2}-\d{2}-\d{2})_"
    r"(?P<end_date>\d{4}-\d{2}-\d{2})_(?P<end_time>\d{2}-\d{2}-\d{2})\.csv$"
)


# ----------------------------------------------------------------------------
# Schema as defined in the ZTBus paper (Table 1)
# ----------------------------------------------------------------------------
ZTBUS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "time_iso": pl.Utf8,  # parsed to Datetime separately
    "time_unix": pl.Int64,
    "electric_powerDemand": pl.Float32,
    "gnss_altitude": pl.Float32,
    "gnss_course": pl.Float32,
    "gnss_latitude": pl.Float64,
    "gnss_longitude": pl.Float64,
    "itcs_busRoute": pl.Utf8,
    "itcs_numberOfPassengers": pl.Float32,
    "itcs_stopName": pl.Utf8,
    "odometry_articulationAngle": pl.Float32,
    "odometry_steeringAngle": pl.Float32,
    "odometry_vehicleSpeed": pl.Float32,
    "odometry_wheelSpeed_fl": pl.Float32,
    "odometry_wheelSpeed_fr": pl.Float32,
    "odometry_wheelSpeed_ml": pl.Float32,
    "odometry_wheelSpeed_mr": pl.Float32,
    "odometry_wheelSpeed_rl": pl.Float32,
    "odometry_wheelSpeed_rr": pl.Float32,
    "status_doorIsOpen": pl.Int8,
    "status_gridIsAvailable": pl.Int8,
    "temperature_ambient": pl.Float32,
    "traction_brakePressure": pl.Float32,
    "traction_tractionForce": pl.Float32,
}

REQUIRED_COLUMNS: Final = frozenset(
    {
        "time_iso",
        "time_unix",
        "electric_powerDemand",
        "odometry_vehicleSpeed",
    }
)


class MissionFileError(ValueError):
    """Raised when a mission file is malformed or fails schema validation."""


def parse_mission_filename(path: Path) -> dict[str, str | int | datetime]:
    """Extract mission metadata from a ZTBus filename.

    Returns a dict with keys ``bus``, ``start_utc``, ``end_utc``, ``mission_id``.
    """
    m = _FILENAME_RE.match(path.name)
    if not m:
        raise MissionFileError(f"Filename does not match ZTBus pattern: {path.name}")

    start_utc = datetime.strptime(
        f"{m['start_date']} {m['start_time'].replace('-', ':')}",
        "%Y-%m-%d %H:%M:%S",
    ).replace(tzinfo=UTC)
    end_utc = datetime.strptime(
        f"{m['end_date']} {m['end_time'].replace('-', ':')}",
        "%Y-%m-%d %H:%M:%S",
    ).replace(tzinfo=UTC)

    return {
        "bus": int(m["bus"]),
        "start_utc": start_utc,
        "end_utc": end_utc,
        "mission_id": path.stem,
    }


def read_mission_csv(path: Path, *, sniff_only: bool = False) -> pl.DataFrame:
    """Read one ZTBus mission CSV with schema enforcement.

    Parameters
    ----------
    path
        Path to a single ``B{bus}_{ts}_{ts}.csv``.
    sniff_only
        If True, return only the first 100 rows. Used for fast schema checks.

    Raises
    ------
    MissionFileError
        If the schema does not match :data:`ZTBUS_SCHEMA` in any required way.
    """
    n_rows = 100 if sniff_only else None

    try:
        df = pl.read_csv(
            path,
            schema_overrides=ZTBUS_SCHEMA,
            null_values=["-", ""],
            n_rows=n_rows,
            try_parse_dates=False,  # we parse time_iso explicitly below
            ignore_errors=False,  # surface bad rows; do not silently skip
        )
    except Exception as exc:
        raise MissionFileError(f"Failed to read {path.name}: {exc}") from exc

    # Required columns check
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise MissionFileError(f"{path.name} is missing required columns: {sorted(missing)}")

    # Parse the ISO timestamp once, into a real Datetime[us, UTC]
    df = df.with_columns(
        pl.col("time_iso").str.to_datetime(time_unit="us", time_zone="UTC", strict=True),
    )

    return df


def read_metadata_csv(path: Path) -> pl.DataFrame:
    """Read the dataset's ``metaData.csv`` if present (Widmer et al. 2023, Table 2)."""
    if not path.exists():
        logger.warning("Metadata file not found: {}", path)
        return pl.DataFrame()
    return pl.read_csv(path, null_values=["-", ""], try_parse_dates=True)


def discover_missions(raw_dir: Path) -> list[Path]:
    """List ZTBus mission CSVs in ``raw_dir`` matching the canonical filename pattern."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data dir does not exist: {raw_dir}")
    files = sorted(p for p in raw_dir.glob("B*.csv") if _FILENAME_RE.match(p.name))
    logger.info("Discovered {} mission files in {}", len(files), raw_dir)
    return files
