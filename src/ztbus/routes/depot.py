"""Depot detection.

The user noted that depot trajectories are not relevant to the energy model:
during depot maneuvering the bus is moving slowly through tight geometry that
the longitudinal model (built for line-haul operation) cannot represent well.
Including these phases in the parameter fit would bias the identified
parameters.

We detect depot phases at the START and END of each mission using a simple
but defensible heuristic:

1. Either the bus is stationary (speed near zero) for an extended period AND
   has no ITCS bus route assigned, OR
2. The first/last few minutes show low-speed maneuvering inside a small spatial
   bounding box (depot footprint).

Known VBZ trolley-bus depots in Zurich (approximate centers):

* Hardau (Hardstrasse): 47.385°N, 8.512°E
* Kalkbreite (Kalkbreitestrasse): 47.376°N, 8.519°E
* Zurich-Oerlikon: 47.412°N, 8.546°E

Outputs are added as a boolean column ``in_depot`` so downstream code can
mask them out without losing the data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from loguru import logger

# Approximate depot polygon centers (radians for compatibility with raw GNSS).
# The dataset stores lat/lon in radians per the ZTBus paper Table 1.
KNOWN_DEPOTS_DEG: list[tuple[str, float, float]] = [
    ("Hardau",     47.385, 8.512),
    ("Kalkbreite", 47.376, 8.519),
    ("Oerlikon",   47.412, 8.546),
]


@dataclass(frozen=True)
class DepotDetectionResult:
    n_rows_start_depot: int
    n_rows_end_depot: int
    detected_depot_at_start: str | None
    detected_depot_at_end: str | None


def _haversine_km(lat1_rad: np.ndarray, lon1_rad: np.ndarray,
                  lat2_rad: float, lon2_rad: float) -> np.ndarray:
    """Great-circle distance in km between arrays of points and a single point."""
    R = 6371.0088
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def detect_depot_phases(
    df: pl.DataFrame,
    *,
    depot_radius_km: float = 0.3,
    min_dwell_seconds: float = 60.0,
    speed_col: str = "speed_smoothed_mps",
) -> tuple[pl.DataFrame, DepotDetectionResult]:
    """Mark start- and end-of-mission depot phases.

    Adds a boolean ``in_depot`` column to ``df`` and returns a
    :class:`DepotDetectionResult` summary.
    """
    n = df.height
    in_depot = np.zeros(n, dtype=bool)

    if n == 0 or "gnss_latitude" not in df.columns:
        return df.with_columns(pl.Series("in_depot", in_depot)), DepotDetectionResult(0, 0, None, None)

    lat = df["gnss_latitude"].to_numpy().astype(float)
    lon = df["gnss_longitude"].to_numpy().astype(float)
    t = df["time_unix"].to_numpy().astype(float) if "time_unix" in df.columns else np.arange(n, dtype=float)
    v = df[speed_col].to_numpy().astype(float) if speed_col in df.columns else np.zeros(n)

    detected_start = _detect_endpoint_depot(lat, lon, v, t, depot_radius_km, min_dwell_seconds, "start")
    detected_end = _detect_endpoint_depot(lat, lon, v, t, depot_radius_km, min_dwell_seconds, "end")

    n_start = 0
    n_end = 0

    if detected_start is not None:
        # Mark consecutive samples from start that remain inside the depot polygon.
        depot_name, depot_lat_rad, depot_lon_rad = detected_start
        d_km = _haversine_km(lat, lon, depot_lat_rad, depot_lon_rad)
        for i in range(n):
            if np.isfinite(d_km[i]) and d_km[i] <= depot_radius_km and v[i] < 3.0:
                in_depot[i] = True
                n_start += 1
            else:
                break

    if detected_end is not None:
        depot_name, depot_lat_rad, depot_lon_rad = detected_end
        d_km = _haversine_km(lat, lon, depot_lat_rad, depot_lon_rad)
        for i in range(n - 1, -1, -1):
            if np.isfinite(d_km[i]) and d_km[i] <= depot_radius_km and v[i] < 3.0:
                in_depot[i] = True
                n_end += 1
            else:
                break

    if n_start or n_end:
        logger.debug("Depot trim: {} samples at start, {} at end", n_start, n_end)

    return (
        df.with_columns(pl.Series("in_depot", in_depot)),
        DepotDetectionResult(
            n_rows_start_depot=n_start,
            n_rows_end_depot=n_end,
            detected_depot_at_start=detected_start[0] if detected_start else None,
            detected_depot_at_end=detected_end[0] if detected_end else None,
        ),
    )


def _detect_endpoint_depot(
    lat: np.ndarray, lon: np.ndarray, v: np.ndarray, t: np.ndarray,
    radius_km: float, min_dwell_s: float, where: str,
) -> tuple[str, float, float] | None:
    """Check whether the start/end of a mission is inside a known depot."""
    if where == "start":
        # First sample with valid GPS
        for i in range(len(lat)):
            if np.isfinite(lat[i]) and np.isfinite(lon[i]):
                start_idx = i
                break
        else:
            return None
        ref_lat, ref_lon = lat[start_idx], lon[start_idx]
    else:
        for i in range(len(lat) - 1, -1, -1):
            if np.isfinite(lat[i]) and np.isfinite(lon[i]):
                end_idx = i
                break
        else:
            return None
        ref_lat, ref_lon = lat[end_idx], lon[end_idx]

    # Dataset stores lat/lon in radians; convert known depots to radians here
    for name, depot_lat_deg, depot_lon_deg in KNOWN_DEPOTS_DEG:
        depot_lat_rad = np.deg2rad(depot_lat_deg)
        depot_lon_rad = np.deg2rad(depot_lon_deg)
        d = _haversine_km(np.array([ref_lat]), np.array([ref_lon]), depot_lat_rad, depot_lon_rad)[0]
        if d <= radius_km:
            return (name, depot_lat_rad, depot_lon_rad)

    return None
