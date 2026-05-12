"""JAX-jitted forward model — Phase 5 Bayesian parameter identification.

This is a 1:1 functional port of :func:`ztbus.physics.powertrain.simulate_powertrain`
from numpy to ``jax.numpy``. The behaviour MUST match within 1e-6 on identical
inputs; see ``tests/test_kernels_parity.py`` for the parity test.

Why a JAX rewrite?
------------------
The Phase 5 optimizer (NUTS via NumPyro) needs three things from the forward
model that the numpy version cannot provide:

1. **Gradients** — NUTS uses Hamiltonian dynamics with gradients of the
   log-posterior w.r.t. the parameters. ``jax.grad`` supplies these by
   tracing through this function automatically.
2. **JIT compilation to XLA** — ``jax.jit`` fuses the entire forward pass
   into a single CPU or GPU kernel. Empirically ~50–100x faster than the
   pandas/numpy version on the same hardware.
3. **vmap over candidate parameters** — ``jax.vmap`` evaluates many parameter
   vectors against the same data in a single kernel call. This is what makes
   per-candidate parallelism free on GPU.

Design rules
------------
- **No Python-side control flow** based on data values. All branching uses
  ``jnp.where`` so the function is JIT-compilable.
- **Static shapes only.** Time-series length is part of the trace; arrays
  passed in must have a fixed length per mission.
- **Pure function.** No mutation, no I/O, no logging. The caller decides what
  to do with the returned arrays.
- **Parameters arrive as a flat ``jnp.ndarray`` of length 7** in the order
  given by :data:`PARAM_NAMES`. This keeps the NumPyro model and the gradient
  pipeline trivial.

Parameter order
---------------
The 7-parameter vector matches Hjelkrem et al. (2021) and the project's
``PowertrainParameters`` dataclass:

    theta = [A, Cd, Crr, eta_prop, eta_recup, c_HVAC, P_aux]

Future extension (``eta_recup_grid`` vs ``eta_recup_battery``) is left as a
single-call-site change — see the "extension hook" comment below.
"""

from __future__ import annotations

from typing import Final

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Parameter conventions
# ---------------------------------------------------------------------------

PARAM_NAMES: Final[tuple[str, ...]] = (
    "frontal_area_m2",
    "drag_coefficient",
    "rolling_resistance_coefficient",
    "efficiency_propulsion",
    "efficiency_recuperation",
    "hvac_coefficient_kW_per_K",
    "auxiliary_power_kW",
)
NUM_PARAMS: Final[int] = len(PARAM_NAMES)

# Indices into theta — explicit constants are clearer than tuple-unpacking
# when reading the forward pass below.
_IDX_A: Final[int] = 0
_IDX_CD: Final[int] = 1
_IDX_CRR: Final[int] = 2
_IDX_ETA_PROP: Final[int] = 3
_IDX_ETA_RECUP: Final[int] = 4
_IDX_C_HVAC: Final[int] = 5
_IDX_P_AUX: Final[int] = 6


# ---------------------------------------------------------------------------
# Physical constants — known, not identified
# ---------------------------------------------------------------------------

# Match values in ztbus.physics.parameters.PhysicalConstants. Kept as module-level
# floats (not a dataclass) so JAX can trace through them as static constants.
G_M_PER_S2: Final[float] = 9.81
RHO_AIR_KG_PER_M3: Final[float] = 1.225
T_COMFORT_K: Final[float] = 294.15  # = 21 °C


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------


def forward(
    theta: jnp.ndarray,  # shape (7,)
    *,
    speed_mps: jnp.ndarray,  # shape (n,)
    acceleration_mps2: jnp.ndarray,  # shape (n,)
    mass_kg: jnp.ndarray,  # shape (n,)
    grade: jnp.ndarray,  # shape (n,)  precomputed in cleaning
    temperature_K: jnp.ndarray,  # shape (n,)
) -> jnp.ndarray:
    """Predict ``P_total`` (W) at every timestamp.

    Mirrors :func:`ztbus.physics.powertrain.simulate_powertrain` exactly,
    except:

    - Grade must be precomputed (no in-function fallback to numerical
      differentiation). This matches what the Phase 5 pipeline does anyway,
      and avoids JIT trouble around the ``valid = |d_dist| > 1e-6`` mask.
    - Energy integration is not returned — the optimizer only needs power
      residuals. Cumulative energy can be reconstructed downstream.

    Parameters
    ----------
    theta
        Parameter vector of length 7, in :data:`PARAM_NAMES` order.
    speed_mps, acceleration_mps2, mass_kg, grade, temperature_K
        Per-sample input arrays, all the same length ``n``.

    Returns
    -------
    P_total_W
        Predicted electric power demand at every timestamp, shape ``(n,)``.
    """
    # Unpack — readability matters more here than micro-optimization;
    # XLA fuses these anyway.
    A = theta[_IDX_A]
    Cd = theta[_IDX_CD]
    Crr = theta[_IDX_CRR]
    eta_prop = theta[_IDX_ETA_PROP]
    eta_recup = theta[_IDX_ETA_RECUP]
    c_HVAC = theta[_IDX_C_HVAC]
    P_aux_kW = theta[_IDX_P_AUX]

    # ---- Forces -----------------------------------------------------------
    F_roll = mass_kg * G_M_PER_S2 * Crr
    F_aero = 0.5 * RHO_AIR_KG_PER_M3 * Cd * A * speed_mps**2
    F_inertia = mass_kg * acceleration_mps2
    F_grade = mass_kg * G_M_PER_S2 * grade
    F_total = F_roll + F_aero + F_inertia + F_grade

    # ---- Mechanical power -------------------------------------------------
    P_mech = F_total * speed_mps

    # ---- Propulsion / recuperation split ----------------------------------
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
    )

    # ---- HVAC (linear in |ΔT|) -------------------------------------------
    delta_T = jnp.abs(temperature_K - T_COMFORT_K)
    P_hvac_W = c_HVAC * delta_T * 1000.0  # coefficient is in kW/K → W here

    # ---- Auxiliary constant load -----------------------------------------
    P_aux_W = P_aux_kW * 1000.0

    return P_elec + P_hvac_W + P_aux_W


# JIT-compiled version. Use this everywhere except inside vmap/grad
# (those will JIT themselves).
forward_jit = jax.jit(forward, static_argnames=())


# vmap over the parameter axis: same data, many candidates.
# Use case: evaluate a NUTS ensemble or a CMA-ES population in one kernel.
#
#     theta_batch.shape == (num_candidates, 7)
#     forward_vmap(theta_batch, **kwargs).shape == (num_candidates, n_samples)
#
# vmap over the parameter axis only — the data is the same for every candidate.
# in_axes maps:
#   theta              → axis 0  (different per candidate)
#   speed_mps          → None    (same data for all candidates)
#   acceleration_mps2  → None
#   mass_kg            → None
#   grade              → None
#   temperature_K      → None
def _forward_for_vmap(
    theta: jnp.ndarray,
    speed_mps: jnp.ndarray,
    acceleration_mps2: jnp.ndarray,
    mass_kg: jnp.ndarray,
    grade: jnp.ndarray,
    temperature_K: jnp.ndarray,
) -> jnp.ndarray:
    """Positional-args adapter so jax.vmap can map only over theta."""
    return forward(
        theta,
        speed_mps=speed_mps,
        acceleration_mps2=acceleration_mps2,
        mass_kg=mass_kg,
        grade=grade,
        temperature_K=temperature_K,
    )


# vmap maps axis 0 of theta; None means "broadcast (don't map) for this argument"
forward_vmap_positional = jax.vmap(
    _forward_for_vmap,
    in_axes=(0, None, None, None, None, None),
    out_axes=0,
)


def forward_vmap(
    theta_batch: jnp.ndarray,
    *,
    speed_mps: jnp.ndarray,
    acceleration_mps2: jnp.ndarray,
    mass_kg: jnp.ndarray,
    grade: jnp.ndarray,
    temperature_K: jnp.ndarray,
) -> jnp.ndarray:
    """vmap forward over a batch of parameter vectors. Same kwargs as ``forward``."""
    return forward_vmap_positional(
        theta_batch,
        speed_mps,
        acceleration_mps2,
        mass_kg,
        grade,
        temperature_K,
    )


forward_vmap_jit = jax.jit(forward_vmap)

__all__ = [
    "G_M_PER_S2",
    "NUM_PARAMS",
    "PARAM_NAMES",
    "RHO_AIR_KG_PER_M3",
    "T_COMFORT_K",
    "forward",
    "forward_jit",
    "forward_vmap",
    "forward_vmap_jit",
]
