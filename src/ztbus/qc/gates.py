"""Quality-control gates.

Each gate is a pure predicate: it takes a cleaned mission DataFrame plus the
mission metadata and returns a :class:`GateResult`. Gates are anchored to the
ZTBus paper's reported ranges, NOT to opinions.

The current set is intentionally small. Gates are added as we discover failure
modes during EDA — every new gate must cite the source that justifies the
threshold.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    value: float | None = None
    threshold: float | tuple[float, float] | None = None
    detail: str = ""


Gate = Callable[[pl.DataFrame], GateResult]


# ---------------------------------------------------------------------------
# Individual gates
# ---------------------------------------------------------------------------
def gate_minimum_duration(df: pl.DataFrame, *, min_seconds: float = 3600.0) -> GateResult:
    """ZTBus paper § Technical Validation rejects records < 1 h. We mirror this."""
    if df.is_empty() or "time_unix" not in df.columns:
        return GateResult("minimum_duration", False, 0.0, min_seconds, "no time column")
    duration = float(df["time_unix"].max() - df["time_unix"].min())
    return GateResult(
        "minimum_duration",
        passed=duration >= min_seconds,
        value=duration,
        threshold=min_seconds,
        detail=f"{duration / 3600:.2f} h",
    )


def gate_no_critical_gap(df: pl.DataFrame, *, max_gap_s: float = 10.0) -> GateResult:
    """Paper rejects missions with VCU gaps ≥ 10 s. We reuse the same threshold."""
    if "dt_s" not in df.columns:
        return GateResult("no_critical_gap", True, None, max_gap_s, "dt_s not present")
    max_gap = float(df["dt_s"].max() or 0.0)
    return GateResult(
        "no_critical_gap",
        passed=max_gap < max_gap_s,
        value=max_gap,
        threshold=max_gap_s,
    )


def gate_energy_intensity_in_range(
    df: pl.DataFrame, *, lo: float = 0.8, hi: float = 3.0
) -> GateResult:
    """Mission-mean kWh/km should land near the paper's 1.5–2.0 range.

    Bounds widened (0.8–3.0) to allow for outliers (very short missions,
    missions with unusual route mixes) without auto-rejecting. A failing
    mission isn't necessarily bad — it's flagged for inspection.
    """
    if {"energy_cum_kWh", "distance_m"}.issubset(df.columns) and df.height > 0:
        E = float(df["energy_cum_kWh"][-1])
        d_km = float(df["distance_m"][-1]) / 1000.0
        if d_km > 0.5:
            kwh_per_km = E / d_km
            return GateResult(
                "energy_intensity_in_range",
                passed=lo <= kwh_per_km <= hi,
                value=kwh_per_km,
                threshold=(lo, hi),
                detail=f"{kwh_per_km:.2f} kWh/km",
            )
    return GateResult("energy_intensity_in_range", True, None, (lo, hi), "insufficient data")


def gate_mean_speed_plausible(df: pl.DataFrame, *, lo: float = 1.0, hi: float = 7.0) -> GateResult:
    """Mean mission speed should be near the paper's reported ~15 km/h ≈ 4.2 m/s."""
    if "speed_smoothed_mps" not in df.columns or df.height == 0:
        return GateResult("mean_speed_plausible", True, None, (lo, hi), "no speed")
    moving = df.filter(pl.col("speed_smoothed_mps") > 0.5)
    if moving.height == 0:
        return GateResult("mean_speed_plausible", False, 0.0, (lo, hi), "never moving")
    mean_v = float(moving["speed_smoothed_mps"].mean() or 0.0)
    return GateResult(
        "mean_speed_plausible",
        passed=lo <= mean_v <= hi,
        value=mean_v,
        threshold=(lo, hi),
        detail=f"{mean_v * 3.6:.1f} km/h while moving",
    )


# ---------------------------------------------------------------------------
# Run all gates
# ---------------------------------------------------------------------------
DEFAULT_GATES: list[Gate] = [
    gate_minimum_duration,
    gate_no_critical_gap,
    gate_energy_intensity_in_range,
    gate_mean_speed_plausible,
]


def run_gates(df: pl.DataFrame, gates: list[Gate] | None = None) -> list[GateResult]:
    """Run all gates and return their results. Caller decides how to react."""
    return [g(df) for g in (gates or DEFAULT_GATES)]
