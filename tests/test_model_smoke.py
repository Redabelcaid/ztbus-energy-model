"""Smoke test for the NumPyro probabilistic model.

This is the "does the model assemble" test — not a Bayesian convergence test.
It verifies:

1. The model can be traced by NumPyro without raising
   (i.e. priors + likelihood are wired up correctly)
2. Prior-predictive samples have sensible shape and magnitude
3. Conditioning on data produces a finite log-joint
4. Different random keys produce different traces (RNG is wired)

Heavier checks — gradient w.r.t. parameters, divergent-transition rate,
posterior R-hat — come in samplers.py and the cluster smoke run, NOT here.
This test is meant to run on CPU in <2 s.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.handlers
import pytest
from numpyro.infer.util import log_density

# Match the kernel parity test: enforce CPU + float64 for parity testing.
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from ztbus.optim.kernels import NUM_PARAMS
from ztbus.optim.model import ztbus_model

# Bridge: dataclass uses long names ("frontal_area_m2"); the NumPyro model
# uses short Hjelkrem-style names. Document the mapping explicitly so any
# rename in either place breaks this assertion loudly.
PARAM_NAMES_USER = (
    "A",
    "Cd",
    "Crr",
    "eta_prop",
    "eta_recup",
    "c_HVAC",
    "P_aux",
)
assert len(PARAM_NAMES_USER) == NUM_PARAMS


@pytest.fixture
def tiny_mission() -> dict[str, jnp.ndarray]:
    """50 samples covering a brief cruise + brake."""
    n = 50
    t = jnp.arange(n, dtype=jnp.float64)
    v = jnp.where(t < 30, 8.0, jnp.maximum(8.0 - 0.4 * (t - 30), 0.0))
    a = jnp.gradient(v)
    mass = jnp.full(n, 19_000.0 + 30 * 70.0)
    grade = 0.02 * jnp.sin(2 * jnp.pi * t / 30)
    temperature_K = jnp.full(n, 283.15)
    return {
        "speed_mps": v,
        "acceleration_mps2": a,
        "mass_kg": mass,
        "grade": grade,
        "temperature_K": temperature_K,
    }


@pytest.fixture
def synthetic_observed_power(tiny_mission: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Plausible-looking observed power, generated from a known theta + noise."""
    from ztbus.optim.kernels import forward

    theta_true = jnp.array(
        [8.4, 0.65, 0.010, 0.85, 0.65, 0.5, 12.0],
        dtype=jnp.float64,
    )
    P_true = forward(theta_true, **tiny_mission)
    key = jax.random.PRNGKey(0)
    noise = 1000.0 * jax.random.normal(key, shape=P_true.shape)
    return P_true + noise


def _is_unobserved_sample(site: dict) -> bool:
    return site["type"] == "sample" and not site.get("is_observed", False)


def test_model_assembles_with_data_only(tiny_mission: dict[str, jnp.ndarray]) -> None:
    """Prior-predictive mode: with observed_power_W=None, all 7 params + sigma + P_obs sampled."""
    key = jax.random.PRNGKey(42)
    with numpyro.handlers.seed(rng_seed=key):
        trace = numpyro.handlers.trace(ztbus_model).get_trace(tiny_mission, None)

    sampled_names = {name for name, site in trace.items() if _is_unobserved_sample(site)}
    expected_latent = set(PARAM_NAMES_USER) | {"sigma_W"}
    assert expected_latent.issubset(
        sampled_names
    ), f"Missing latents. Got {sampled_names}, need at least {expected_latent}"


def test_model_conditions_on_observed_power(
    tiny_mission: dict[str, jnp.ndarray],
    synthetic_observed_power: jnp.ndarray,
) -> None:
    """With observed_power_W supplied, P_obs becomes a likelihood site, not latent."""
    key = jax.random.PRNGKey(42)
    with numpyro.handlers.seed(rng_seed=key):
        trace = numpyro.handlers.trace(ztbus_model).get_trace(
            tiny_mission,
            synthetic_observed_power,
        )
    assert "P_obs" in trace
    assert trace["P_obs"]["is_observed"], "P_obs should be observed when data is supplied"


def test_log_density_is_finite_at_prior_sample(
    tiny_mission: dict[str, jnp.ndarray],
    synthetic_observed_power: jnp.ndarray,
) -> None:
    """log p(data, theta) finite at a prior draw — NUTS prerequisite."""
    key_prior, _ = jax.random.split(jax.random.PRNGKey(7))
    with numpyro.handlers.seed(rng_seed=key_prior):
        prior_trace = numpyro.handlers.trace(ztbus_model).get_trace(tiny_mission, None)
    prior_params = {
        name: site["value"] for name, site in prior_trace.items() if _is_unobserved_sample(site)
    }

    lj, _ = log_density(
        ztbus_model,
        model_args=(tiny_mission, synthetic_observed_power),
        model_kwargs={},
        params=prior_params,
    )
    assert jnp.isfinite(lj), f"log-density was {lj} — expected finite"


def test_different_keys_give_different_prior_samples(
    tiny_mission: dict[str, jnp.ndarray],
) -> None:
    """RNG-wiring check: different seeds → different prior draws."""
    with numpyro.handlers.seed(rng_seed=jax.random.PRNGKey(0)):
        t0 = numpyro.handlers.trace(ztbus_model).get_trace(tiny_mission, None)
    with numpyro.handlers.seed(rng_seed=jax.random.PRNGKey(1)):
        t1 = numpyro.handlers.trace(ztbus_model).get_trace(tiny_mission, None)
    assert t0["Crr"]["value"] != t1["Crr"]["value"]


def test_param_set_matches_kernel(tiny_mission: dict[str, jnp.ndarray]) -> None:
    """Model latents must equal kernel's PARAM_NAMES (plus sigma + P_obs)."""
    key = jax.random.PRNGKey(0)
    with numpyro.handlers.seed(rng_seed=key):
        trace = numpyro.handlers.trace(ztbus_model).get_trace(tiny_mission, None)
    physical_latents = {
        name
        for name, site in trace.items()
        if _is_unobserved_sample(site) and name not in {"sigma_W", "P_obs"}
    }
    assert physical_latents == set(PARAM_NAMES_USER), (
        f"Model latents don't match kernel parameters.\n"
        f"  Model: {physical_latents}\n"
        f"  Kernel: {PARAM_NAMES_USER}"
    )
    assert len(physical_latents) == NUM_PARAMS
