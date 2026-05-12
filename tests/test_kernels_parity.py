"""Parity test: kernels.forward must match physics.powertrain.simulate_powertrain.

This is the gate that protects us from silently introducing a bug in the JAX
rewrite. The forward model is the foundation everything else (NumPyro model,
log-posterior, sampler) builds on. If the JAX version disagrees with the numpy
reference on a single input, every downstream result is suspect.

Strategy
--------
1. Generate a realistic synthetic mission (speed/accel/temperature trajectory
   that exercises every branch: cruise, brake, idle, hot, cold).
2. Pick a non-default parameter vector so we don't accidentally pass via the
   identity transform.
3. Run both the numpy reference and the JAX port.
4. Assert agreement to 1e-6 absolute, 1e-6 relative.

If this passes, the parameter identification can proceed with confidence. If
it fails, the only honest answer is "stop and fix the kernel" — no point
running 12-hour NUTS chains on a buggy model.

The test does NOT require a GPU; it runs on CPU and is fast (<1 s).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Force JAX to use CPU + 64-bit precision for the parity check. The default is
# 32-bit, which would mask tiny but real differences in the math.
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from ztbus.optim.kernels import (
    NUM_PARAMS,
    PARAM_NAMES,
    forward_jit,
    forward_vmap,
)
from ztbus.optim.kernels import (
    forward as jax_forward,
)

# The reference numpy implementation lives in the main package. The test only
# imports it; we never modify it.
from ztbus.physics import (
    PowertrainParameters,
    simulate_powertrain,
)

# ---------------------------------------------------------------------------
# Fixtures: synthetic mission that exercises every model branch
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_mission() -> dict[str, np.ndarray]:
    """A 1-hour synthetic trajectory with cruise, brake, idle phases.

    Designed to trigger every code path in the forward model:
      - Positive P_mech (forward propulsion)  → η_prop branch
      - Negative P_mech (regen braking)       → η_recup branch
      - Non-zero grade                        → mgθ term active
      - Hot and cold ambient                  → HVAC term non-zero on both ends
      - Variable mass                         → passenger-aware mass term
    """
    rng = np.random.default_rng(42)
    n = 3600  # 1 Hz × 1 hour
    t = np.arange(n, dtype=float)

    # Speed: 20 min cruise at 8 m/s, 5 min brake to 0, 5 min idle, repeat
    v = np.zeros(n)
    phase = (t % 1800).astype(int)
    v = np.where(
        phase < 1200, 8.0, np.where(phase < 1500, 8.0 * (1.0 - (phase - 1200) / 300.0), 0.0)
    )

    # Acceleration: central diff of v
    a = np.gradient(v)

    # Mass: 19,000 kg + 30 passengers × 70 kg + small jitter
    mass = 19_000.0 + 30 * 70.0 + rng.normal(0, 50, n)

    # Grade: ±5% sine wave (hills)
    grade = 0.05 * np.sin(2 * np.pi * t / 600)  # period 10 min

    # Temperature: ramp from -5°C (cold morning) to 30°C (hot noon)
    temperature_K = 268.15 + (303.15 - 268.15) * (t / n)

    return {
        "time_s": t,
        "speed_mps": v,
        "acceleration_mps2": a,
        "mass_kg": mass,
        "grade": grade,
        "temperature_K": temperature_K,
    }


@pytest.fixture
def test_parameters() -> PowertrainParameters:
    """Non-default parameters to ensure we test the actual math, not the defaults."""
    return PowertrainParameters(
        frontal_area_m2=8.3,
        drag_coefficient=0.62,
        rolling_resistance_coefficient=0.0094,
        efficiency_propulsion=0.88,
        efficiency_recuperation=0.71,
        hvac_coefficient_kW_per_K=0.45,
        auxiliary_power_kW=14.0,
    )


def _params_to_theta(p: PowertrainParameters) -> jnp.ndarray:
    """Pack the dataclass into a flat 7-vector in PARAM_NAMES order."""
    return jnp.asarray(p.to_array(), dtype=jnp.float64)


# ---------------------------------------------------------------------------
# Core parity checks
# ---------------------------------------------------------------------------


def test_param_order_consistent_with_dataclass() -> None:
    """The 7-vector in kernels.py must use the same order as the dataclass."""
    from dataclasses import fields

    dataclass_names = tuple(f.name for f in fields(PowertrainParameters))
    assert dataclass_names == PARAM_NAMES, (
        f"Parameter order mismatch.\n"
        f"  kernels.PARAM_NAMES = {PARAM_NAMES}\n"
        f"  PowertrainParameters fields = {dataclass_names}"
    )
    assert NUM_PARAMS == 7


def test_jax_matches_numpy_on_synthetic_mission(
    synthetic_mission: dict[str, np.ndarray],
    test_parameters: PowertrainParameters,
) -> None:
    """The headline test: JAX P_total must equal numpy P_total to 1e-6."""
    # numpy reference
    sim_np = simulate_powertrain(
        time_s=synthetic_mission["time_s"],
        speed_mps=synthetic_mission["speed_mps"],
        acceleration_mps2=synthetic_mission["acceleration_mps2"],
        mass_kg=synthetic_mission["mass_kg"],
        grade=synthetic_mission["grade"],
        temperature_K=synthetic_mission["temperature_K"],
        parameters=test_parameters,
    )

    # JAX port
    theta = _params_to_theta(test_parameters)
    P_jax = jax_forward(
        theta,
        speed_mps=jnp.asarray(synthetic_mission["speed_mps"]),
        acceleration_mps2=jnp.asarray(synthetic_mission["acceleration_mps2"]),
        mass_kg=jnp.asarray(synthetic_mission["mass_kg"]),
        grade=jnp.asarray(synthetic_mission["grade"]),
        temperature_K=jnp.asarray(synthetic_mission["temperature_K"]),
    )

    np.testing.assert_allclose(
        np.asarray(P_jax),
        sim_np.power_total_W,
        rtol=1e-6,
        atol=1e-6,
        err_msg="JAX forward model disagrees with numpy reference",
    )


def test_jit_compiled_version_matches_eager(
    synthetic_mission: dict[str, np.ndarray],
    test_parameters: PowertrainParameters,
) -> None:
    """jax.jit must not change the math, only the speed."""
    theta = _params_to_theta(test_parameters)
    kwargs = {
        "speed_mps": jnp.asarray(synthetic_mission["speed_mps"]),
        "acceleration_mps2": jnp.asarray(synthetic_mission["acceleration_mps2"]),
        "mass_kg": jnp.asarray(synthetic_mission["mass_kg"]),
        "grade": jnp.asarray(synthetic_mission["grade"]),
        "temperature_K": jnp.asarray(synthetic_mission["temperature_K"]),
    }
    P_eager = jax_forward(theta, **kwargs)
    P_jit = forward_jit(theta, **kwargs)
    np.testing.assert_allclose(np.asarray(P_eager), np.asarray(P_jit), rtol=1e-12, atol=1e-12)


def test_vmap_over_candidates_equals_loop(
    synthetic_mission: dict[str, np.ndarray],
    test_parameters: PowertrainParameters,
) -> None:
    """vmap-ed evaluation across candidates must match an explicit Python loop."""
    # 5 different parameter vectors
    rng = np.random.default_rng(0)
    base = test_parameters.to_array()
    theta_batch = jnp.asarray(base[None, :] * (1.0 + 0.05 * rng.normal(size=(5, NUM_PARAMS))))

    kwargs = {
        "speed_mps": jnp.asarray(synthetic_mission["speed_mps"]),
        "acceleration_mps2": jnp.asarray(synthetic_mission["acceleration_mps2"]),
        "mass_kg": jnp.asarray(synthetic_mission["mass_kg"]),
        "grade": jnp.asarray(synthetic_mission["grade"]),
        "temperature_K": jnp.asarray(synthetic_mission["temperature_K"]),
    }

    # Reference: explicit loop
    P_loop = jnp.stack([jax_forward(theta_batch[i], **kwargs) for i in range(5)])
    # vmap
    P_vmap = forward_vmap(theta_batch, **kwargs)

    np.testing.assert_allclose(np.asarray(P_loop), np.asarray(P_vmap), rtol=1e-12, atol=1e-12)


def test_gradient_is_finite(
    synthetic_mission: dict[str, np.ndarray],
    test_parameters: PowertrainParameters,
) -> None:
    """jax.grad must return finite gradients — this is the property NUTS needs."""
    theta = _params_to_theta(test_parameters)
    kwargs = {
        "speed_mps": jnp.asarray(synthetic_mission["speed_mps"]),
        "acceleration_mps2": jnp.asarray(synthetic_mission["acceleration_mps2"]),
        "mass_kg": jnp.asarray(synthetic_mission["mass_kg"]),
        "grade": jnp.asarray(synthetic_mission["grade"]),
        "temperature_K": jnp.asarray(synthetic_mission["temperature_K"]),
    }

    # Scalar loss (so grad is a vector, not a Jacobian)
    def loss(t):
        return jnp.sum(jax_forward(t, **kwargs) ** 2)

    g = jax.grad(loss)(theta)
    assert g.shape == (NUM_PARAMS,)
    assert jnp.all(jnp.isfinite(g)), f"Non-finite gradient: {g}"
