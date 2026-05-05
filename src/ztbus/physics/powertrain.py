"""Longitudinal powertrain forward model.

This is the cleaned-up, typed, and tested counterpart of the supervisor's
``updated_energy_calculation.py``. Behaviour is preserved where it was correct
and explicitly fixed where it was not. Each fix is annotated below.

Model decomposition (per Beckers, Paasche & Sundström, 2021):

    F_total(t) = F_roll + F_aero(v) + F_inertia(a) + F_grade(theta)

    P_mech(t)  = F_total(t) · v(t)

    P_elec(t)  =  P_mech / η_prop                 if  P_mech ≥ 0
                  P_mech · η_recup                if  P_mech < 0

    P_total(t) = P_elec(t) + P_HVAC(T_amb, T_comfort) + P_aux

with linear HVAC term P_HVAC = c_HVAC · |T_amb − T_comfort|.

Fixes vs. the original script
-----------------------------

1. Mass: the original had ``mass=12000`` as a default; the HESS lighTram® 19
   has a curb mass of ~19 t (Widmer et al., Sci Data 10:687, 2023). Mass is
   now sourced from :class:`PhysicalConstants` and adjusted by passenger load
   when available.
2. Energy integration: the original used ``np.cumsum(power) * dt`` with a
   single median ``dt``, which is a left-Riemann sum and wrong on irregular
   sampling. We now use trapezoidal integration on the actual time vector.
3. Grade: the original computed ``np.gradient(elev) / np.gradient(dist)`` on
   raw signals; this amplifies GNSS noise. The model still accepts a
   precomputed grade column (preferred), and only derives one as a fallback.
4. HVAC unit: the original mixed kW and W via ``hvac_coeff * delta_T * 1000``.
   We now declare the coefficient explicitly in kW/K and convert at one
   well-marked place.
5. Aux power: the default was 2 kW; the ZTBus paper measures auxiliary draw
   in the 20–30 kW range (HVAC inclusive). We split the constant aux from the
   variable HVAC term and source it from :class:`PowertrainParameters`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ztbus.physics.parameters import PhysicalConstants, PowertrainParameters

ArrayF = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PowertrainSimulation:
    """Output of :func:`simulate_powertrain`.

    All arrays have the same length as the input time vector.
    """

    power_total_W: ArrayF       # what we compare against electric_powerDemand
    power_mech_W: ArrayF        # diagnostic: F_total · v
    power_elec_W: ArrayF        # diagnostic: after η_prop / η_recup split
    power_hvac_W: ArrayF        # diagnostic
    power_aux_W: ArrayF         # diagnostic
    energy_total_kWh: ArrayF    # cumulative, trapezoidal


def _resolve_mass(
    n: int,
    passengers: ArrayF | None,
    constants: PhysicalConstants,
) -> ArrayF:
    """Return per-sample vehicle mass [kg]."""
    if passengers is None:
        return np.full(n, constants.curb_mass_kg, dtype=float)
    return constants.curb_mass_kg + np.asarray(passengers, dtype=float) * constants.avg_passenger_mass_kg


def _resolve_grade(
    grade: ArrayF | None,
    elevation_m: ArrayF | None,
    distance_m: ArrayF | None,
) -> ArrayF:
    """Return per-sample road grade [-] (rise / run, small-angle approx).

    Prefer a precomputed, smoothed grade column (from the cleaning pipeline).
    Fall back to a raw finite-difference if not provided — but log a warning
    once at the call site, not here.
    """
    if grade is not None:
        return np.asarray(grade, dtype=float)
    if elevation_m is None or distance_m is None:
        # No topology information at all → assume flat. This is honest.
        return np.zeros_like(elevation_m if elevation_m is not None else distance_m, dtype=float)

    elev = np.asarray(elevation_m, dtype=float)
    dist = np.asarray(distance_m, dtype=float)

    if elev.size < 2:
        return np.zeros_like(elev)

    d_dist = np.gradient(dist)
    d_elev = np.gradient(elev)
    out = np.zeros_like(elev)
    valid = np.abs(d_dist) > 1e-6
    out[valid] = d_elev[valid] / d_dist[valid]
    return out


def simulate_powertrain(
    *,
    time_s: ArrayF,
    speed_mps: ArrayF,
    acceleration_mps2: ArrayF,
    mass_kg: ArrayF | None = None,
    passengers: ArrayF | None = None,
    elevation_m: ArrayF | None = None,
    distance_m: ArrayF | None = None,
    grade: ArrayF | None = None,
    temperature_K: ArrayF | None = None,
    parameters: PowertrainParameters | None = None,
    constants: PhysicalConstants | None = None,
) -> PowertrainSimulation:
    """Forward-simulate electric power demand from a kinematic time series.

    Parameters
    ----------
    time_s, speed_mps, acceleration_mps2
        Required time series, all the same length n.
    mass_kg
        Per-sample mass; if None, derived from ``passengers`` and the curb
        mass; if both are None, falls back to ``constants.curb_mass_kg``.
    elevation_m, distance_m, grade
        Provide either a precomputed ``grade`` (preferred) or both
        ``elevation_m`` and ``distance_m`` so grade can be derived.
    temperature_K
        Per-sample ambient temperature in Kelvin (matches the dataset units).
        If None, HVAC contribution is zero.
    parameters
        Identified parameters; defaults to the prior point estimates.
    constants
        Known constants; defaults to HESS lighTram® 19 values.

    Returns
    -------
    PowertrainSimulation
        Power components and cumulative energy.
    """
    parameters = parameters if parameters is not None else PowertrainParameters()
    constants = constants if constants is not None else PhysicalConstants()

    t = np.asarray(time_s, dtype=float)
    v = np.asarray(speed_mps, dtype=float)
    a = np.asarray(acceleration_mps2, dtype=float)
    n = t.size

    if not (v.size == a.size == n):
        raise ValueError("time_s, speed_mps, acceleration_mps2 must be the same length")

    m = np.asarray(mass_kg, dtype=float) if mass_kg is not None else _resolve_mass(n, passengers, constants)
    theta = _resolve_grade(grade, elevation_m, distance_m)

    # ---------------- Forces ------------------------------------------------
    F_roll = m * constants.g_m_per_s2 * parameters.rolling_resistance_coefficient
    F_aero = 0.5 * constants.rho_air_kg_per_m3 * parameters.drag_coefficient * parameters.frontal_area_m2 * v**2
    F_inertia = m * a
    F_grade = m * constants.g_m_per_s2 * theta
    F_total = F_roll + F_aero + F_inertia + F_grade

    # ---------------- Mechanical and electric power -------------------------
    P_mech = F_total * v
    P_elec = np.where(
        P_mech >= 0,
        P_mech / parameters.efficiency_propulsion,
        P_mech * parameters.efficiency_recuperation,
    )

    # ---------------- HVAC --------------------------------------------------
    if temperature_K is None:
        P_hvac = np.zeros(n, dtype=float)
    else:
        T = np.asarray(temperature_K, dtype=float)
        delta_T = np.abs(T - constants.T_comfort_K)
        # coefficient is in kW/K; convert to W here and only here
        P_hvac = parameters.hvac_coefficient_kW_per_K * delta_T * 1000.0

    # ---------------- Aux ---------------------------------------------------
    P_aux = np.full(n, parameters.auxiliary_power_kW * 1000.0, dtype=float)

    P_total = P_elec + P_hvac + P_aux

    # ---------------- Energy: trapezoid on actual time grid ----------------
    if n >= 2:
        energy_J = np.concatenate([[0.0], np.cumsum(0.5 * (P_total[1:] + P_total[:-1]) * np.diff(t))])
    else:
        energy_J = np.zeros(n, dtype=float)
    energy_kWh = energy_J / 3.6e6

    return PowertrainSimulation(
        power_total_W=P_total,
        power_mech_W=P_mech,
        power_elec_W=P_elec,
        power_hvac_W=P_hvac,
        power_aux_W=P_aux,
        energy_total_kWh=energy_kWh,
    )
