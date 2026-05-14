"""Add the Hjelkrem regen min-speed gate to the forward kernel.

Physical motivation
-------------------
Below ~15 km/h the bus's traction motor cannot produce sufficient back-EMF
to push current into the battery (or grid). The inverter detects this and
disables the regen path; mechanical braking takes over and dissipates the
kinetic energy as heat in the friction brakes.

Hjelkrem 2021 cites Asamer et al. 2016 for the 15 km/h threshold; our
diagnosis from the first smoke run (eta_recup pegged at upper prior bound,
P_aux compensating) is consistent with this physics being absent from the
model.

Mathematically, the change is:

    Before:
        P_elec = P_mech / eta_prop      if P_mech >= 0
                 P_mech * eta_recup      if P_mech <  0

    After:
        P_elec = P_mech / eta_prop      if P_mech >= 0
                 P_mech * eta_recup      if P_mech <  0  AND  v >= 15 km/h
                 0                       if P_mech <  0  AND  v <  15 km/h

When regen is suppressed, the negative mechanical power is dissipated in
friction brakes (no electrical effect). HVAC and P_aux continue normally.

Apply this on the cluster:
    cd ~/ztbus-energy-model
    uv run python dump/scripts/add_regen_gate.py
"""

from pathlib import Path

KERNELS_PY = Path("src/ztbus/optim/kernels.py")
text = KERNELS_PY.read_text()

# ---------------------------------------------------------------------------
# 1. Add the new physical constant near the other constants
# ---------------------------------------------------------------------------
old_constants = """G_M_PER_S2: Final[float] = 9.81
RHO_AIR_KG_PER_M3: Final[float] = 1.225
T_COMFORT_K: Final[float] = 294.15  # = 21 °C"""

new_constants = """G_M_PER_S2: Final[float] = 9.81
RHO_AIR_KG_PER_M3: Final[float] = 1.225
T_COMFORT_K: Final[float] = 294.15  # = 21 °C

# Regen kill-switch threshold (Hjelkrem 2021, citing Asamer 2016): below this
# speed, the motor's back-EMF is insufficient and regen is mechanically
# unavailable. 15 km/h = 4.1667 m/s.
MIN_REGEN_SPEED_MPS: Final[float] = 15.0 / 3.6"""

assert old_constants in text, "Couldn't find constants block — kernels.py may have changed"
text = text.replace(old_constants, new_constants)

# ---------------------------------------------------------------------------
# 2. Replace the P_elec branching with the gated version
# ---------------------------------------------------------------------------
old_branching = """    # ---- Propulsion / recuperation split ----------------------------------
    # Hjelkrem's branching:
    #   P_elec = P_mech / eta_prop      if P_mech >= 0  (traction)
    #          = P_mech * eta_recup     if P_mech <  0  (regen)
    #
    # Extension hook: when supervisor confirms grid-aware regen split, replace
    # ``eta_recup`` here with
    #     jnp.where(grid_available, eta_recup_grid, eta_recup_battery)
    # and the function signature gains ``grid_available`` as an input.
    P_elec = jnp.where(
        P_mech >= 0.0,
        P_mech / eta_prop,
        P_mech * eta_recup,
    )"""

new_branching = """    # ---- Propulsion / recuperation split ----------------------------------
    # Hjelkrem's branching with the min-speed regen kill-switch:
    #   P_elec = P_mech / eta_prop                       if P_mech >= 0  (traction)
    #          = P_mech * eta_recup                      if P_mech <  0  AND v >= 15 km/h
    #          = 0                                       if P_mech <  0  AND v <  15 km/h
    #
    # The third case represents friction braking: kinetic energy is dissipated as
    # heat, with no electrical effect. HVAC and P_aux continue regardless.
    #
    # Extension hook: when supervisor confirms grid-aware regen split, replace
    # ``eta_recup`` here with
    #     jnp.where(grid_available, eta_recup_grid, eta_recup_battery)
    # and the function signature gains ``grid_available`` as an input.
    regen_active = (P_mech < 0.0) & (speed_mps >= MIN_REGEN_SPEED_MPS)
    P_elec = jnp.where(
        P_mech >= 0.0,
        P_mech / eta_prop,
        jnp.where(regen_active, P_mech * eta_recup, 0.0),
    )"""

assert old_branching in text, "Couldn't find P_elec branching — kernels.py may have changed"
text = text.replace(old_branching, new_branching)

# ---------------------------------------------------------------------------
# 3. Export the new constant
# ---------------------------------------------------------------------------
old_all = """__all__ = [
    "G_M_PER_S2",
    "NUM_PARAMS",
    "PARAM_NAMES",
    "RHO_AIR_KG_PER_M3",
    "T_COMFORT_K",
    "forward",
    "forward_jit",
    "forward_vmap",
    "forward_vmap_jit",
]"""

new_all = """__all__ = [
    "G_M_PER_S2",
    "MIN_REGEN_SPEED_MPS",
    "NUM_PARAMS",
    "PARAM_NAMES",
    "RHO_AIR_KG_PER_M3",
    "T_COMFORT_K",
    "forward",
    "forward_jit",
    "forward_vmap",
    "forward_vmap_jit",
]"""

assert old_all in text, "Couldn't find __all__ block — kernels.py may have changed"
text = text.replace(old_all, new_all)

KERNELS_PY.write_text(text)

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
final = KERNELS_PY.read_text()
assert "MIN_REGEN_SPEED_MPS" in final
assert "regen_active" in final
print("✓ MIN_REGEN_SPEED_MPS constant added (15 km/h = 4.1667 m/s)")
print("✓ P_elec branching updated with regen kill-switch")
print("✓ __all__ updated")
print("\nPatched. Verify with: git diff src/ztbus/optim/kernels.py")
print("\nNOTE: This will break the bit-exact parity test against the numpy")
print("reference (which doesn't have the gate). That's expected — we're now")
print("ahead of the numpy reference. We will need to either:")
print("  a) port the same gate to physics/powertrain.py simulate_powertrain, or")
print("  b) skip the parity test (it has served its purpose).")
