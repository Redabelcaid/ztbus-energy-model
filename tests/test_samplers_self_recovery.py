"""Self-recovery test for NUTS — the gold-standard verification.

Strategy
--------
1. Pick a known ``theta_true`` (physically plausible values).
2. Generate synthetic ``P_obs = forward(theta_true, mission) + small_noise``.
3. Run NUTS on (mission, P_obs).
4. Check that the 95% credible interval of every parameter covers the truth.

If this test passes, the inference machinery works:
  - The model is correctly specified
  - The kernel + autodiff produces correct gradients
  - NUTS leapfrog integration is stable
  - The priors don't artificially exclude the truth

If it fails, we have a bug, and running on real data is meaningless.

Compute budget
--------------
~30 s on CPU for 2 chains × 300 warmup × 300 samples on a 200-sample mission.
We don't run it on every commit — marked ``@pytest.mark.slow``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

# Enforce CPU + float64 for reproducibility
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from ztbus.optim.kernels import PARAM_NAMES, forward
from ztbus.optim.samplers import nuts_fit, posterior_summary


@pytest.fixture
def synthetic_mission_with_truth() -> tuple[dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray]:
    """200 samples of synthetic mission + known theta + observed power.

    The mission deliberately exercises every branch:
    - Cruise at 8 m/s with mild acceleration
    - Braking phase (negative P_mech)
    - Hot temperature spread (HVAC active in both directions)
    - Modest grade variation
    """
    n = 200
    t = jnp.arange(n, dtype=jnp.float64)
    # Triangular speed: ramp up to 12, hold, ramp down
    v = jnp.where(
        t < 60, t * 0.2, jnp.where(t < 140, 12.0, jnp.maximum(12.0 - 0.2 * (t - 140), 0.0))
    )
    a = jnp.gradient(v)
    mass = jnp.full(n, 19_000.0 + 30 * 70.0)
    grade = 0.03 * jnp.sin(2 * jnp.pi * t / 50)
    # Temperature ramps from 5°C (278 K) to 30°C (303 K)
    temperature_K = 278.15 + (303.15 - 278.15) * (t / n)

    data = {
        "speed_mps": v,
        "acceleration_mps2": a,
        "mass_kg": mass,
        "grade": grade,
        "temperature_K": temperature_K,
    }

    # The truth — well inside every prior bound declared in model.py
    theta_true = jnp.array(
        [
            8.4,  # A
            0.65,  # Cd
            0.010,  # Crr
            0.85,  # eta_prop
            0.65,  # eta_recup
            0.5,  # c_HVAC (kW/K)
            12.0,  # P_aux (kW)
        ],
        dtype=jnp.float64,
    )

    # Forward + Gaussian noise (sigma_true = 2 kW)
    P_true = forward(theta_true, **data)
    key = jax.random.PRNGKey(123)
    noise = 2000.0 * jax.random.normal(key, shape=P_true.shape)
    P_obs = P_true + noise
    return data, theta_true, P_obs


@pytest.mark.slow
def test_self_recovery_credible_intervals_cover_truth(
    synthetic_mission_with_truth: tuple[dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray],
) -> None:
    """The 95% credible interval of every parameter must cover its true value.

    This is the headline correctness test for the inference machinery.
    Allowing 1 parameter to miss the CI (out of 7) is a typical false-positive
    rate at 95% nominal coverage — that's fine. Strictness on all 7 would be
    over-tight.
    """
    data, theta_true, P_obs = synthetic_mission_with_truth

    result = nuts_fit(
        data=data,
        observed_power_W=P_obs,
        num_warmup=300,
        num_samples=300,
        num_chains=2,
        chain_method="sequential",
        progress_bar=False,
        rng_seed=42,
    )

    # Check chain convergence first — if R-hat is bad, the coverage test
    # below is meaningless.
    assert (
        result.diagnostics["r_hat_max"] < 1.05
    ), f"Chains haven't converged: r_hat_max = {result.diagnostics['r_hat_max']}"

    summary = posterior_summary(result.samples)
    summary_dict = {row["parameter"]: row for row in summary.to_dicts()}

    # Map kernel PARAM_NAMES (dataclass field names) to model variable names
    kernel_to_model = dict(
        zip(
            PARAM_NAMES,
            ("A", "Cd", "Crr", "eta_prop", "eta_recup", "c_HVAC", "P_aux"),
            strict=True,
        )
    )

    missed = []
    for kernel_name, model_name in kernel_to_model.items():
        param_idx = PARAM_NAMES.index(kernel_name)
        true_value = float(theta_true[param_idx])
        row = summary_dict[model_name]
        lo, hi = row["ci_lo_95"], row["ci_hi_95"]
        if not (lo <= true_value <= hi):
            missed.append((model_name, true_value, lo, hi))

    # Allow at most 1 of 7 to miss (95% nominal coverage means ~5% miss rate)
    assert len(missed) <= 1, "More than 1 parameter missed its 95% CI:\n" + "\n".join(
        f"  {n}: true={t:.4f} not in [{lo:.4f}, {hi:.4f}]" for n, t, lo, hi in missed
    )


@pytest.mark.slow
def test_self_recovery_sigma_posterior_is_near_truth(
    synthetic_mission_with_truth: tuple[dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray],
) -> None:
    """The posterior on sigma should concentrate near the 2000 W we injected."""
    data, _, P_obs = synthetic_mission_with_truth
    result = nuts_fit(
        data=data,
        observed_power_W=P_obs,
        num_warmup=300,
        num_samples=300,
        num_chains=2,
        chain_method="sequential",
        progress_bar=False,
        rng_seed=42,
    )
    sigma_mean = float(result.samples["sigma_W"].mean())
    # True noise sigma was 2000 W. With 200 samples and 7-param model, posterior
    # should be within ~30% of truth.
    assert (
        1400.0 < sigma_mean < 2700.0
    ), f"sigma posterior mean = {sigma_mean:.0f} W, expected near 2000 W"
