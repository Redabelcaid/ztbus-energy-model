"""Tests for the I/O readers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from ztbus.io import MissionFileError, parse_mission_filename, read_mission_csv


def test_parse_mission_filename_canonical() -> None:
    p = Path("B183_2019-10-16_02-52-43_2019-10-16_07-10-12.csv")
    meta = parse_mission_filename(p)
    assert meta["bus"] == 183
    assert meta["start_utc"] == datetime(2019, 10, 16, 2, 52, 43, tzinfo=timezone.utc)
    assert meta["end_utc"] == datetime(2019, 10, 16, 7, 10, 12, tzinfo=timezone.utc)
    assert meta["mission_id"] == p.stem


def test_parse_mission_filename_invalid_raises() -> None:
    with pytest.raises(MissionFileError):
        parse_mission_filename(Path("not_a_ztbus_file.csv"))


def test_read_mission_csv_minimal(tmp_path: Path) -> None:
    """Smoke test: a tiny CSV with the required columns reads cleanly."""
    csv = tmp_path / "B183_2020-01-01_00-00-00_2020-01-01_00-00-02.csv"
    csv.write_text(
        "time_iso,time_unix,electric_powerDemand,odometry_vehicleSpeed\n"
        "2020-01-01T00:00:00+00:00,1577836800,12345.0,3.0\n"
        "2020-01-01T00:00:01+00:00,1577836801,12500.0,3.1\n"
        "2020-01-01T00:00:02+00:00,1577836802,12700.0,3.2\n"
    )
    df = read_mission_csv(csv)
    assert df.height == 3
    assert df["time_iso"].dtype == pl.Datetime("us", "UTC")


def test_read_mission_csv_missing_required_column_raises(tmp_path: Path) -> None:
    csv = tmp_path / "B183_2020-01-01_00-00-00_2020-01-01_00-00-02.csv"
    csv.write_text(
        "time_iso,time_unix,electric_powerDemand\n"     # missing odometry_vehicleSpeed
        "2020-01-01T00:00:00+00:00,1577836800,12345.0\n"
    )
    with pytest.raises(MissionFileError, match="missing required columns"):
        read_mission_csv(csv)
