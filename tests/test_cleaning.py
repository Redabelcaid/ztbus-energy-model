"""Tests for the cleaning pipeline.

The strategy:

* Build small synthetic missions with known-bad properties.
* Assert the pipeline detects them via flag columns or QC rejection.
* Avoid pinning specific numeric outputs so the tests survive future
  smoothing-window tweaks.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ztbus.cleaning import CleaningConfig, TimestampQualityError, clean_mission


def _synthetic_mission(n: int = 3601, *, bad_speed: bool = False, bad_power: bool = False) -> pl.DataFrame:
    """1-hour, 1Hz mission template; toggle injected anomalies."""
    rng = np.random.default_rng(seed=42)
    t_unix = np.arange(1577836800, 1577836800 + n, dtype=np.int64)
    speed = 5.0 + rng.normal(0, 0.5, n)
    if bad_speed:
        speed[100] = -3.0      # large negative spike
        speed[200] = 30.0      # impossible high

    power = 25_000 + rng.normal(0, 5_000, n)
    if bad_power:
        power[50] = 1_000_000  # impossible CAN spike

    return pl.DataFrame({
        "time_iso": [f"2020-01-01T{(s % 86400) // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}+00:00"
                     for s in t_unix.tolist()],
        "time_unix": t_unix,
        "electric_powerDemand": power.astype(np.float32),
        "odometry_vehicleSpeed": speed.astype(np.float32),
        "gnss_altitude": (408 + rng.normal(0, 2, n)).astype(np.float32),
        "gnss_latitude": np.full(n, np.deg2rad(47.4)).astype(np.float64),
        "gnss_longitude": np.full(n, np.deg2rad(8.5)).astype(np.float64),
        "gnss_course": np.full(n, 0.0).astype(np.float32),
        "temperature_ambient": (293.15 + rng.normal(0, 0.1, n)).astype(np.float32),
        "itcs_numberOfPassengers": np.full(n, 15.0).astype(np.float32),
    })


def test_clean_mission_runs_on_well_formed_data() -> None:
    df = _synthetic_mission()
    cleaned, qc = clean_mission(df, mission_id="test_clean_01", bus=183, cfg=CleaningConfig())
    assert not qc.rejected
    assert cleaned.height == df.height
    assert "speed_smoothed_mps" in cleaned.columns
    assert "altitude_smoothed_m" in cleaned.columns
    assert "dt_s" in cleaned.columns


def test_clean_mission_flags_speed_outliers() -> None:
    df = _synthetic_mission(bad_speed=True)
    cleaned, qc = clean_mission(df, mission_id="test_clean_02", bus=183, cfg=CleaningConfig())
    assert qc.flag_counts.get("speed_negative_flag", 0) >= 1
    assert qc.flag_counts.get("speed_outlier_flag", 0) >= 1


def test_clean_mission_flags_power_outliers() -> None:
    df = _synthetic_mission(bad_power=True)
    cleaned, qc = clean_mission(df, mission_id="test_clean_03", bus=183, cfg=CleaningConfig())
    assert qc.flag_counts.get("power_outlier_flag", 0) >= 1


def test_clean_mission_rejects_non_monotonic_time() -> None:
    df = _synthetic_mission()
    # Reverse a chunk of time
    t = df["time_unix"].to_numpy().copy()
    t[100:200] = t[100:200][::-1]
    df = df.with_columns(pl.Series("time_unix", t))
    cleaned, qc = clean_mission(df, mission_id="test_clean_04", bus=183, cfg=CleaningConfig())
    assert qc.rejected
    assert "non-monotonic" in (qc.rejection_reason or "").lower()


def test_clean_mission_dedupes_timestamps() -> None:
    df = _synthetic_mission(n=10)
    df = pl.concat([df, df.head(3)])  # dupes
    cleaned, qc = clean_mission(df, mission_id="test_clean_05", bus=183, cfg=CleaningConfig())
    assert cleaned.height == 10  # dupes removed


def test_features_kinematics_adds_acceleration_and_distance() -> None:
    from ztbus.features import add_kinematics

    df = _synthetic_mission()
    cleaned, _ = clean_mission(df, mission_id="t", bus=183, cfg=CleaningConfig())
    out = add_kinematics(cleaned)
    assert "acceleration_mps2" in out.columns
    assert "distance_m" in out.columns
    # ~5 m/s for 3600 s ≈ 18 km
    assert 17_000 < out["distance_m"][-1] < 19_000


def test_features_energy_adds_cum_energy() -> None:
    from ztbus.features import add_energy, add_kinematics, add_mass

    df = _synthetic_mission()
    cleaned, _ = clean_mission(df, mission_id="t", bus=183, cfg=CleaningConfig())
    out = add_kinematics(cleaned)
    out = add_mass(out)
    out = add_energy(out)
    assert "energy_cum_kWh" in out.columns
    # ~25 kW for 1 hour ≈ 25 kWh
    assert 20 < out["energy_cum_kWh"][-1] < 30


def test_qc_gates_run_against_cleaned_mission() -> None:
    from ztbus.cleaning.grade import derive_grade
    from ztbus.features import add_energy, add_kinematics, add_mass
    from ztbus.qc import run_gates

    df = _synthetic_mission()
    cleaned, _ = clean_mission(df, mission_id="t", bus=183, cfg=CleaningConfig())
    cleaned = add_kinematics(cleaned)
    cleaned = derive_grade(cleaned, CleaningConfig().grade)
    cleaned = add_mass(cleaned)
    cleaned = add_energy(cleaned)

    gates = run_gates(cleaned)
    gate_names = {g.name for g in gates}
    assert "minimum_duration" in gate_names
    assert "energy_intensity_in_range" in gate_names
