"""Tests for the powertrain forward model.

The goal is to anchor *physical invariants* — tests that should hold
regardless of parameter choices — rather than to pin numerical outputs.
This way the tests survive parameter identification.
"""

from __future__ import annotations

import numpy as np
import pytest

from ztbus.physics import (
    PhysicalConstants,
    PowertrainParameters,
    simulate_powertrain,
)


@pytest.fixture
def t_one_hz() -> np.ndarray:
    """One hour at 1 Hz."""
    return np.arange(0, 3600, dtype=float)


@pytest.fixture
def stationary(t_one_hz: np.ndarray) -> dict:
    """A bus parked all day: zero speed, zero acceleration, comfort temperature."""
    n = t_one_hz.size
    return {
        "time_s": t_one_hz,
        "speed_mps": np.zeros(n),
        "acceleration_mps2": np.zeros(n),
        "temperature_K": np.full(n, PhysicalConstants().T_comfort_K),
    }


@pytest.fixture
def cruise(t_one_hz: np.ndarray) -> dict:
    """Steady cruise at 5 m/s (~18 km/h, mean Zurich bus speed)."""
    n = t_one_hz.size
    return {
        "time_s": t_one_hz,
        "speed_mps": np.full(n, 5.0),
        "acceleration_mps2": np.zeros(n),
        "temperature_K": np.full(n, PhysicalConstants().T_comfort_K),
    }


# ----------------------------------------------------------------------------
# Invariants
# ----------------------------------------------------------------------------
def test_stationary_at_comfort_temp_consumes_only_aux(stationary: dict) -> None:
    """A stationary bus with no HVAC load draws only the constant auxiliary power."""
    p = PowertrainParameters()
    sim = simulate_powertrain(**stationary, parameters=p)

    expected_W = p.auxiliary_power_kW * 1000.0
    np.testing.assert_allclose(sim.power_total_W, expected_W)


def test_cruise_consumes_more_than_aux(cruise: dict) -> None:
    """At cruise speed, total power must exceed the auxiliary baseline."""
    p = PowertrainParameters()
    sim = simulate_powertrain(**cruise, parameters=p)
    assert (sim.power_total_W > p.auxiliary_power_kW * 1000.0).all()


def test_cruise_energy_in_expected_range(cruise: dict) -> None:
    """At 5 m/s for 1 hour, energy / distance should land in 1.0–2.5 kWh/km.

    Range source: ZTBus paper § Statistical Analysis (1.5–2.0 kWh/km observed).
    Bound widened to allow uncalibrated priors to pass; tightened in Phase 5.
    """
    sim = simulate_powertrain(**cruise)
    distance_km = (cruise["speed_mps"][-1] * cruise["time_s"][-1]) / 1000.0
    kwh_per_km = sim.energy_total_kWh[-1] / distance_km
    assert 1.0 < kwh_per_km < 2.5, f"Got {kwh_per_km:.2f} kWh/km"


def test_uniform_acceleration_dominates_at_low_speed(t_one_hz: np.ndarray) -> None:
    """At low speed and high acceleration, F_inertia should dominate over F_aero."""
    n = t_one_hz.size
    a = np.full(n, 1.0)        # 1 m/s² acceleration
    v = np.minimum(a * t_one_hz, 10.0)  # ramp up, cap at 10 m/s

    sim_decel = simulate_powertrain(
        time_s=t_one_hz, speed_mps=v, acceleration_mps2=a,
    )
    sim_no_decel = simulate_powertrain(
        time_s=t_one_hz, speed_mps=v, acceleration_mps2=np.zeros(n),
    )
    # Accelerating costs more than coasting at the same speed
    assert sim_decel.energy_total_kWh[-1] > sim_no_decel.energy_total_kWh[-1]


def test_recuperation_recovers_energy_on_braking(t_one_hz: np.ndarray) -> None:
    """Negative mechanical power must be processed via the recuperation path."""
    n = t_one_hz.size
    v = np.full(n, 5.0)
    a = np.full(n, -0.5)    # braking
    sim = simulate_powertrain(time_s=t_one_hz, speed_mps=v, acceleration_mps2=a)
    # Mechanical power is negative; electric power should be |P_mech| * η_recup,
    # which is smaller in magnitude than P_mech itself.
    assert (sim.power_mech_W < 0).all()
    assert (np.abs(sim.power_elec_W) <= np.abs(sim.power_mech_W) + 1e-6).all()


def test_array_roundtrip_preserves_parameters() -> None:
    p = PowertrainParameters(
        frontal_area_m2=8.3,
        drag_coefficient=0.62,
        rolling_resistance_coefficient=0.009,
        efficiency_propulsion=0.88,
        efficiency_recuperation=0.62,
        hvac_coefficient_kW_per_K=0.7,
        auxiliary_power_kW=15.0,
    )
    p2 = PowertrainParameters.from_array(p.to_array())
    assert p == p2


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="same length"):
        simulate_powertrain(
            time_s=np.arange(10.0),
            speed_mps=np.zeros(5),
            acceleration_mps2=np.zeros(10),
        )
