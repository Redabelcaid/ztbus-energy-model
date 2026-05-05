"""Cleaning pipeline orchestration.

Composes the per-signal cleaning steps in dependency order. Returns the
cleaned :class:`polars.DataFrame` together with a :class:`MissionQC` summary
that records what was found and what was done — so a reviewer (or your
supervisor) can audit a mission without re-running the pipeline.

Order matters because some steps depend on others:

1. timestamps — needed for ``dt_s`` and gap flags everywhere else.
2. speed — needed for the smoothed signal that feeds kinematics and grade.
3. power — independent.
4. altitude — independent.
5. temperature — independent.
6. GNSS coordinates — depends on speed (for course validity).
7. Kinematics features (acceleration, distance) — depend on speed.
8. Grade — depends on smoothed altitude AND distance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl
from loguru import logger

from ztbus.cleaning.altitude import clean_altitude
from ztbus.cleaning.config import CleaningConfig
from ztbus.cleaning.gnss import clean_gnss
from ztbus.cleaning.power import clean_power
from ztbus.cleaning.speed import clean_speed
from ztbus.cleaning.temperature import clean_temperature
from ztbus.cleaning.timestamps import TimestampQualityError, clean_timestamps


@dataclass
class MissionQC:
    """QC summary produced alongside a cleaned mission DataFrame."""

    mission_id: str
    bus: int
    n_rows_in: int
    n_rows_out: int
    duration_s: float
    rejected: bool = False
    rejection_reason: str | None = None
    flag_counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "bus": self.bus,
            "n_rows_in": self.n_rows_in,
            "n_rows_out": self.n_rows_out,
            "duration_s": self.duration_s,
            "rejected": self.rejected,
            "rejection_reason": self.rejection_reason,
            **{f"flag_count_{k}": v for k, v in self.flag_counts.items()},
            "notes": "; ".join(self.notes) if self.notes else None,
        }


def clean_mission(
    df: pl.DataFrame,
    *,
    mission_id: str,
    bus: int,
    cfg: CleaningConfig,
) -> tuple[pl.DataFrame, MissionQC]:
    """Run the full cleaning pipeline on one mission DataFrame.

    Returns the cleaned DataFrame and a :class:`MissionQC`. If the mission is
    rejected (e.g., non-monotonic timestamps), the returned DataFrame is the
    input unchanged and ``qc.rejected`` is True.

    Feature engineering (kinematics, distance, mass) is intentionally *outside*
    this function; it lives in :mod:`ztbus.features` and runs after cleaning.
    """
    qc = MissionQC(
        mission_id=mission_id,
        bus=bus,
        n_rows_in=df.height,
        n_rows_out=df.height,
        duration_s=0.0,
    )

    # 1. Timestamps -----------------------------------------------------------
    try:
        df = clean_timestamps(df, cfg.timestamps)
    except TimestampQualityError as exc:
        qc.rejected = True
        qc.rejection_reason = str(exc)
        logger.warning("[{}] rejected: {}", mission_id, exc)
        return df, qc

    if df.is_empty():
        qc.rejected = True
        qc.rejection_reason = "empty after timestamp cleaning"
        return df, qc

    qc.n_rows_out = df.height
    qc.duration_s = float(df["time_unix"].max() - df["time_unix"].min())

    # 2. Per-signal cleaning -------------------------------------------------
    df = clean_speed(df, cfg.speed)
    df = clean_power(df, cfg.power)
    df = clean_altitude(df, cfg.altitude)
    df = clean_temperature(df, cfg.temperature)
    df = clean_gnss(df, cfg.gnss_coordinates, bus=bus)

    # Note: distance + grade are added in the features stage because grade
    # requires distance. We just record what's pending.
    qc.notes.append("kinematics + grade pending in features stage")

    # 3. Tally flag columns --------------------------------------------------
    flag_cols = [c for c in df.columns if c.endswith("_flag")]
    qc.flag_counts = {c: int(df[c].sum() or 0) for c in flag_cols}

    return df, qc
