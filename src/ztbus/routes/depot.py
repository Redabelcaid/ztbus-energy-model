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
    # Derived from the data: 50m-bin histogram + connected components on stationary
    # positions across the full 1409-mission corpus. See
    # /scratch/users/rbelcaid/ztbus/reports/derived_depots.json for the full output.
    ("Depot_01", 47.39765, 8.54143),  # main: 1.3M samples, ~366h cumulative parking
    ("Depot_02", 47.36811, 8.49580),  # ~76h cumulative
    ("Depot_03", 47.34471, 8.52975),  # ~76h cumulative
    ("Depot_04", 47.35040, 8.56097),  # ~60h cumulative
    ("Depot_05", 47.41370, 8.47749),  # ~45h cumulative
]


@dataclass(frozen=True)
class DepotDetectionResult:
    n_rows_start_depot: int
    n_rows_end_depot: int
    detected_depot_at_start: str | None
    detected_depot_at_end: str | None


def _haversine_km(
    lat1_rad: np.ndarray, lon1_rad: np.ndarray, lat2_rad: float, lon2_rad: float
) -> np.ndarray:
    """Great-circle distance in km between arrays of points and a single point."""
    R = 6371.0088
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    return np.asarray(2 * R * np.arcsin(np.sqrt(a)))


def detect_depot_phases(  # noqa: PLR0912
    df: pl.DataFrame,
    *,
    depot_radius_km: float = 0.15,
    min_dwell_seconds: float = 60.0,
    speed_col: str = "speed_smoothed_mps",
) -> tuple[pl.DataFrame, DepotDetectionResult]:
    """Mark depot phases anywhere in the mission (start, mid, or end).

    Per-sample check: in depot polygon AND speed < 3 m/s. Sustained presence
    of >= ``min_dwell_seconds`` is required to count, so passing through a
    depot at full speed does not trigger.
    """
    n = df.height
    in_depot = np.zeros(n, dtype=bool)

    if n == 0 or "gnss_latitude" not in df.columns:
        return df.with_columns(pl.Series("in_depot", in_depot)), DepotDetectionResult(
            0, 0, None, None
        )

    lat = df["gnss_latitude"].to_numpy().astype(float)
    lon = df["gnss_longitude"].to_numpy().astype(float)
    t = (
        df["time_unix"].to_numpy().astype(float)
        if "time_unix" in df.columns
        else np.arange(n, dtype=float)
    )
    v = df[speed_col].to_numpy().astype(float) if speed_col in df.columns else np.zeros(n)

    # Per-sample distance to NEAREST known depot
    nearest_dist_km = np.full(n, np.inf)
    nearest_depot_idx = np.full(n, -1, dtype=int)
    for di, (_, depot_lat_deg, depot_lon_deg) in enumerate(KNOWN_DEPOTS_DEG):
        d_lat_rad = np.deg2rad(depot_lat_deg)
        d_lon_rad = np.deg2rad(depot_lon_deg)
        d_km = _haversine_km(lat, lon, d_lat_rad, d_lon_rad)
        closer = d_km < nearest_dist_km
        nearest_dist_km[closer] = d_km[closer]
        nearest_depot_idx[closer] = di

    # Candidate samples: in-radius, slow-moving, GPS-valid
    cand = np.isfinite(nearest_dist_km) & (nearest_dist_km <= depot_radius_km) & (v < 3.0)

    # Sustained-presence filter: keep only contiguous runs of >= min_dwell_seconds
    if cand.any():
        i = 0
        while i < n:
            if cand[i]:
                j = i
                while j < n and cand[j]:
                    j += 1
                duration = t[j - 1] - t[i] if j > i else 0
                if duration >= min_dwell_seconds:
                    in_depot[i:j] = True
                i = j
            else:
                i += 1

    # Identify start/end depot names if applicable
    detected_start = None
    detected_end = None
    if in_depot[0] and nearest_depot_idx[0] >= 0:
        detected_start = KNOWN_DEPOTS_DEG[nearest_depot_idx[0]][0]
    if in_depot[-1] and nearest_depot_idx[-1] >= 0:
        detected_end = KNOWN_DEPOTS_DEG[nearest_depot_idx[-1]][0]

    # Count contiguous start/end blocks for backward-compat with QC reporting
    n_start = 0
    if in_depot[0]:
        for i in range(n):
            if in_depot[i]:
                n_start += 1
            else:
                break
    n_end = 0
    if in_depot[-1]:
        for i in range(n - 1, -1, -1):
            if in_depot[i]:
                n_end += 1
            else:
                break

    if in_depot.any():
        logger.debug(
            "Depot phases: {} samples total ({} at start, {} at end)",
            int(in_depot.sum()),
            n_start,
            n_end,
        )

    return (
        df.with_columns(pl.Series("in_depot", in_depot)),
        DepotDetectionResult(
            n_rows_start_depot=n_start,
            n_rows_end_depot=n_end,
            detected_depot_at_start=detected_start,
            detected_depot_at_end=detected_end,
        ),
    )


def _detect_endpoint_depot(
    lat: np.ndarray,
    lon: np.ndarray,
    v: np.ndarray,
    t: np.ndarray,
    radius_km: float,
    min_dwell_s: float,
    where: str,
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
