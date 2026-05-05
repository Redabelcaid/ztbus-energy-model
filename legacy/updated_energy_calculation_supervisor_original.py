import numpy as np


def vehicle_power_profile(
    dft,
    A=?,
    Crr=?,
    eff_prop=?,
    eff_recup=?,
    hvac_coeff=?,
    P_aux_kW=2.0,
    rho=1.225,
    Cd=0.65,
    mass=12000,
    g=9.81,
    T_comfort=21
):

    v = dft["speed_m_per_s"].to_numpy(dtype=float)
    a = dft["acceleration_m_s2"].to_numpy(dtype=float)
    elev = dft["elevation_m"].to_numpy(dtype=float)
    dist = dft["distance_m"].to_numpy(dtype=float)
    t = dft["time_s"].to_numpy(dtype=float)

    n = len(dft)

    # ----------------------------------------------------
    # MASS (optional column)
    # ----------------------------------------------------

    if "mass_kg" in dft.columns:
        m = dft["mass_kg"].to_numpy(dtype=float)
    else:
        m = np.full(n, mass)

    # ----------------------------------------------------
    # TEMPERATURE (optional column)
    # ----------------------------------------------------

# ----------------------------------------------------
# TEMPERATURE (optional column)
# ----------------------------------------------------

    if "temperature_C" in dft.columns:
        temp = dft["temperature_C"].to_numpy(dtype=float)
        hvac_enabled = True
    else:
        temp = np.full(n, T_comfort)  # neutral temp → no HVAC
        hvac_enabled = False
    # ----------------------------------------------------
    # SLOPE
    # ----------------------------------------------------

    grade = np.zeros(n)

    if n > 1:

        d_dist = np.gradient(dist)
        d_elev = np.gradient(elev)

        valid = np.abs(d_dist) > 1e-6

        grade[valid] = d_elev[valid] / d_dist[valid]

    # ----------------------------------------------------
    # FORCES
    # ----------------------------------------------------

    F_roll = m * g * Crr
    F_aero = 0.5 * rho * Cd * A * v**2
    F_inertia = m * a
    F_grade = m * g * grade

    F_total = F_roll + F_aero + F_inertia + F_grade

    # ----------------------------------------------------
    # PROPULSION
    # ----------------------------------------------------

    P_mech = F_total * v

    P_prop = np.zeros_like(P_mech)

    prop_mask = P_mech >= 0
    recup_mask = ~prop_mask

    P_prop[prop_mask] = P_mech[prop_mask] / eff_prop
    P_prop[recup_mask] = P_mech[recup_mask] * eff_recup

    # ----------------------------------------------------
    # HVAC (only if temperature exists)
    # ----------------------------------------------------

    if hvac_enabled:

        delta_T = np.abs(temp - T_comfort)

        P_hvac = hvac_coeff * delta_T * 1000

    else:

        P_hvac = np.zeros(n)

    # ----------------------------------------------------
    # AUX LOAD (always present)
    # ----------------------------------------------------

    P_aux = np.full(n, P_aux_kW * 1000)

    # ----------------------------------------------------

    P_total = P_prop + P_hvac + P_aux

    power_kW = P_total / 1000

    if n > 1:
        dt = np.median(np.diff(t))
    else:
        dt = 0

    energy_kWh = np.cumsum(power_kW) * dt / 3600

    return power_kW, energy_kWh