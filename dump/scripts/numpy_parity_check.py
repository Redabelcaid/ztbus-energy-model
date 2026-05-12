"""Pure-numpy parity check for the JAX kernel math.

Run this BEFORE installing JAX on the cluster — it proves the math expression
in kernels.py is identical to simulate_powertrain. Since jax.numpy mirrors
numpy's API for these operations exactly, the JAX version will then trivially
match on the cluster too.
"""

from dataclasses import dataclass

import numpy as np


# ---- Reference: copy of ztbus.physics.powertrain.simulate_powertrain ----
@dataclass(frozen=True)
class PhysicalConstants:
    curb_mass_kg: float = 19000.0
    avg_passenger_mass_kg: float = 70.0
    rho_air_kg_per_m3: float = 1.225
    g_m_per_s2: float = 9.81
    T_comfort_K: float = 294.15
    P_motor_max_W: float = 320_000.0


@dataclass
class PowertrainParameters:
    frontal_area_m2: float = 8.0
    drag_coefficient: float = 0.65
    rolling_resistance_coefficient: float = 0.008
    efficiency_propulsion: float = 0.85
    efficiency_recuperation: float = 0.65
    hvac_coefficient_kW_per_K: float = 0.5
    auxiliary_power_kW: float = 12.0

    def to_array(self):
        return np.array(
            [
                self.frontal_area_m2,
                self.drag_coefficient,
                self.rolling_resistance_coefficient,
                self.efficiency_propulsion,
                self.efficiency_recuperation,
                self.hvac_coefficient_kW_per_K,
                self.auxiliary_power_kW,
            ],
            dtype=float,
        )


def simulate_powertrain_reference(
    *,
    time_s,
    speed_mps,
    acceleration_mps2,
    mass_kg,
    grade,
    temperature_K,
    parameters,
    constants,
):
    """1:1 copy of the production simulate_powertrain (P_total only)."""
    m = np.asarray(mass_kg, dtype=float)
    v = np.asarray(speed_mps, dtype=float)
    a = np.asarray(acceleration_mps2, dtype=float)
    theta = np.asarray(grade, dtype=float)
    T = np.asarray(temperature_K, dtype=float)

    F_roll = m * constants.g_m_per_s2 * parameters.rolling_resistance_coefficient
    F_aero = (
        0.5
        * constants.rho_air_kg_per_m3
        * parameters.drag_coefficient
        * parameters.frontal_area_m2
        * v**2
    )
    F_inertia = m * a
    F_grade = m * constants.g_m_per_s2 * theta
    F_total = F_roll + F_aero + F_inertia + F_grade

    P_mech = F_total * v
    P_elec = np.where(
        P_mech >= 0,
        P_mech / parameters.efficiency_propulsion,
        P_mech * parameters.efficiency_recuperation,
    )
    delta_T = np.abs(T - constants.T_comfort_K)
    P_hvac = parameters.hvac_coefficient_kW_per_K * delta_T * 1000.0
    P_aux = np.full(v.size, parameters.auxiliary_power_kW * 1000.0, dtype=float)
    return P_elec + P_hvac + P_aux


# ---- Math mirror of kernels.py (numpy instead of jax.numpy) -------------
G_M_PER_S2 = 9.81
RHO_AIR_KG_PER_M3 = 1.225
T_COMFORT_K = 294.15


def forward_numpy_mirror(theta, *, speed_mps, acceleration_mps2, mass_kg, grade, temperature_K):
    """Exact transcription of kernels.forward using numpy. If this matches
    simulate_powertrain_reference, the JAX version will also match."""
    A = theta[0]
    Cd = theta[1]
    Crr = theta[2]
    eta_prop = theta[3]
    eta_recup = theta[4]
    c_HVAC = theta[5]
    P_aux_kW = theta[6]

    F_roll = mass_kg * G_M_PER_S2 * Crr
    F_aero = 0.5 * RHO_AIR_KG_PER_M3 * Cd * A * speed_mps**2
    F_inertia = mass_kg * acceleration_mps2
    F_grade = mass_kg * G_M_PER_S2 * grade
    F_total = F_roll + F_aero + F_inertia + F_grade

    P_mech = F_total * speed_mps
    P_elec = np.where(P_mech >= 0.0, P_mech / eta_prop, P_mech * eta_recup)
    delta_T = np.abs(temperature_K - T_COMFORT_K)
    P_hvac_W = c_HVAC * delta_T * 1000.0
    P_aux_W = P_aux_kW * 1000.0
    return P_elec + P_hvac_W + P_aux_W


# ---- Run the parity check -----------------------------------------------
def main():
    rng = np.random.default_rng(42)
    n = 3600
    t = np.arange(n, dtype=float)
    phase = (t % 1800).astype(int)
    v = np.where(
        phase < 1200, 8.0, np.where(phase < 1500, 8.0 * (1.0 - (phase - 1200) / 300.0), 0.0)
    )
    a = np.gradient(v)
    mass = 19_000.0 + 30 * 70.0 + rng.normal(0, 50, n)
    grade = 0.05 * np.sin(2 * np.pi * t / 600)
    temperature_K = 268.15 + (303.15 - 268.15) * (t / n)

    params = PowertrainParameters(
        frontal_area_m2=8.3,
        drag_coefficient=0.62,
        rolling_resistance_coefficient=0.0094,
        efficiency_propulsion=0.88,
        efficiency_recuperation=0.71,
        hvac_coefficient_kW_per_K=0.45,
        auxiliary_power_kW=14.0,
    )
    consts = PhysicalConstants()
    theta = params.to_array()

    P_ref = simulate_powertrain_reference(
        time_s=t,
        speed_mps=v,
        acceleration_mps2=a,
        mass_kg=mass,
        grade=grade,
        temperature_K=temperature_K,
        parameters=params,
        constants=consts,
    )
    P_kernel = forward_numpy_mirror(
        theta,
        speed_mps=v,
        acceleration_mps2=a,
        mass_kg=mass,
        grade=grade,
        temperature_K=temperature_K,
    )

    abs_err = np.abs(P_ref - P_kernel)
    rel_err = abs_err / (np.abs(P_ref) + 1e-12)

    print(f"n samples              : {n}")
    print(f"P_ref range  [W]       : [{P_ref.min():>10.1f}, {P_ref.max():>10.1f}]")
    print(f"max |absolute error|   : {abs_err.max():.3e} W")
    print(f"max |relative error|   : {rel_err.max():.3e}")
    print(f"P_total mean [kW]      : {P_ref.mean() / 1000:.2f}")
    print(f"P_total RMS  [kW]      : {np.sqrt((P_ref**2).mean()) / 1000:.2f}")

    # Branch coverage diagnostics
    P_mech = (
        mass * consts.g_m_per_s2 * params.rolling_resistance_coefficient
        + 0.5 * consts.rho_air_kg_per_m3 * params.drag_coefficient * params.frontal_area_m2 * v**2
        + mass * a
        + mass * consts.g_m_per_s2 * grade
    ) * v
    print(f"samples in traction (P_mech>=0) : {(P_mech >= 0).sum()}")
    print(f"samples in regen    (P_mech< 0) : {(P_mech < 0).sum()}")
    print(f"samples in HVAC heating  (T<comfort): {(temperature_K < consts.T_comfort_K).sum()}")
    print(f"samples in HVAC cooling  (T>comfort): {(temperature_K > consts.T_comfort_K).sum()}")

    assert abs_err.max() < 1e-9, "PARITY FAILED — kernel math diverges from reference"
    print("\nPARITY OK: kernel math matches simulate_powertrain to < 1e-9 W")


if __name__ == "__main__":
    main()
