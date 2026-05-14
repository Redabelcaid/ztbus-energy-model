"""NumPyro probabilistic model — Phase 5 Bayesian parameter identification.

This module turns the JAX forward kernel (kernels.forward) into a Bayesian
inverse problem:

    posterior(theta | data)  ∝  likelihood(data | theta)  ×  prior(theta)

NumPyro encodes this as a generative program: each ``numpyro.sample(...)``
declares a latent random variable; the final ``numpyro.sample(..., obs=...)``
declares the likelihood by conditioning on observed data. The sampler
(NUTS in our case) then traces through this program to compute the
log-posterior and its gradient.

What this file does NOT do
--------------------------
- It does not run the sampler. That is ``samplers.py`` (next step).
- It does not load data from parquet. Data is passed in as plain JAX arrays
  by the caller — keeping the model pure makes it trivially testable.
- It does not do data masking (depot phases, low-speed regen gate). Those
  are caller responsibilities: the caller filters samples before passing
  them in, and / or supplies an optional ``sample_weights`` array.

Prior choices
-------------
Priors are anchored on Hjelkrem et al. (2021) Table 3 (their literature
priors for the same 7-parameter model), widened where appropriate:

    A  : frontal area [m²]
        Truncated Normal, mean 8.4 (Widmer Table 2 for the HESS lighTram®19),
        sd 0.3, bounds [7.0, 9.5]. Physical bus dimensions barely vary.

    Cd : drag coefficient [-]
        Uniform(0.50, 0.85). Hjelkrem uses 0.70; we widen because trolley-bus
        aerodynamics with pantograph hardware aren't well-constrained.

    Crr : rolling resistance coefficient [-]
        Uniform(0.005, 0.020). Hjelkrem uses 0.010 prior; widen to cover
        winter tires and worn tyres.

    eta_prop : propulsion efficiency [-]
        Beta-like via Uniform(0.70, 0.95). Hjelkrem uses 0.82.

    eta_recup : recuperation efficiency [-]
        Uniform(0.40, 0.85). Hjelkrem uses 0.82; we widen the lower bound
        to allow for the trolley-bus regime where regen sometimes can't flow
        (status_gridIsAvailable=0 with full battery → wasted). This is the
        parameter most likely to surprise us.

    c_HVAC : HVAC coefficient [kW/K]
        HalfNormal(scale=1.0). Strictly positive, prior mean ~0.8 kW/K
        matching the 3.5-year seasonality envelope visible in
        03_energy_seasonality.png (winter delta ~30%).

    P_aux : auxiliary power [kW]
        Uniform(1.0, 30.0). Hjelkrem uses 2 kW; the ZTBus paper documents
        aux draw of 20–30 kW (HVAC inclusive). We let the data decide where
        in this very wide range the bus actually lives. This is the
        parameter that the η_battery degeneracy will likely make hard to
        identify on its own (see ADR 0002).

Noise model
-----------
We add a per-sample observation noise sigma (also identified):

    P_observed ~ Normal(P_model(theta), sigma)

sigma : kW. Prior: HalfNormal(scale=20.0).
A scale of 20 kW is generous — Widmer's measurements have <1 kW
quantization noise from the CAN bus, but residual model bias (HVAC linearity,
mass uncertainty, slope smoothing) will dominate. Starting wide lets the
posterior on sigma tell us how bad the model fits — a tight, small sigma
posterior means "model fits well"; a wide one means "model misses
something structural."

Optional sample weights
-----------------------
The caller can pass per-sample weights (e.g. inverse cleaning flag counts)
via ``sample_weights``. When present, the likelihood becomes weighted:

    log p(y | theta) = sum_i w_i · log Normal(y_i | mu_i, sigma)

Default behaviour (sample_weights=None) is unit weights — every sample
counts equally.

Caller contract
---------------
The ``ztbus_model`` function below is intended to be passed to
NumPyro's MCMC machinery directly:

    from numpyro.infer import MCMC, NUTS
    mcmc = MCMC(NUTS(ztbus_model), num_warmup=1000, num_samples=2000)
    mcmc.run(rng_key, data=mission_data, observed_power_W=P_obs)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from ztbus.optim.kernels import forward

if TYPE_CHECKING:
    from collections.abc import Mapping

    from jax import Array


# ---------------------------------------------------------------------------
# Prior specifications — all bounds + scales documented in the module docstring
# ---------------------------------------------------------------------------

# Frontal area
_A_PRIOR_MEAN: float = 8.4
_A_PRIOR_SD: float = 0.3
_A_LO: float = 7.0
_A_HI: float = 9.5

# Drag coefficient
_CD_LO: float = 0.50
_CD_HI: float = 1.10

# Rolling resistance
_CRR_LO: float = 0.005
_CRR_HI: float = 0.020

# Propulsion efficiency
_ETA_PROP_LO: float = 0.70
_ETA_PROP_HI: float = 0.95

# Recuperation efficiency
_ETA_RECUP_LO: float = 0.30
_ETA_RECUP_HI: float = 0.95

# HVAC coefficient
_C_HVAC_PRIOR_SCALE: float = 1.0  # HalfNormal scale [kW/K]

# Auxiliary power
_P_AUX_LO_KW: float = 0.1
_P_AUX_HI_KW: float = 30.0

# Noise model
_SIGMA_PRIOR_SCALE_W: float = 20_000.0  # HalfNormal scale on power-residual sigma in WATTS


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


def ztbus_model(
    data: Mapping[str, Array],
    observed_power_W: Array | None = None,
    sample_weights: Array | None = None,
) -> None:
    """NumPyro generative model for the 7-parameter Hjelkrem powertrain.

    Parameters
    ----------
    data
        Dict-like with arrays ``speed_mps``, ``acceleration_mps2``,
        ``mass_kg``, ``grade``, ``temperature_K``. All same length ``n``.
        Filtering (e.g. depot mask) is the caller's responsibility.
    observed_power_W
        Measured ``electric_powerDemand`` in W, length ``n``. When ``None``,
        the model runs in prior-predictive mode (useful for sanity checks).
    sample_weights
        Optional per-sample weights. ``None`` means unit weights.

    Returns
    -------
    None
        NumPyro models are side-effecting (they record samples and observations
        via context-manager primitives). The MCMC machinery uses the
        program trace, not the return value.
    """
    # --- Priors -----------------------------------------------------------
    A = numpyro.sample(
        "A",
        dist.TruncatedNormal(_A_PRIOR_MEAN, _A_PRIOR_SD, low=_A_LO, high=_A_HI),
    )
    Cd = numpyro.sample("Cd", dist.Uniform(_CD_LO, _CD_HI))
    Crr = numpyro.sample("Crr", dist.Uniform(_CRR_LO, _CRR_HI))
    eta_prop = numpyro.sample("eta_prop", dist.Uniform(_ETA_PROP_LO, _ETA_PROP_HI))
    eta_recup = numpyro.sample("eta_recup", dist.Uniform(_ETA_RECUP_LO, _ETA_RECUP_HI))
    c_HVAC = numpyro.sample("c_HVAC", dist.HalfNormal(_C_HVAC_PRIOR_SCALE))
    P_aux = numpyro.sample("P_aux", dist.Uniform(_P_AUX_LO_KW, _P_AUX_HI_KW))

    # Observation noise sigma in WATTS. HalfNormal keeps it positive.
    sigma = numpyro.sample("sigma_W", dist.HalfNormal(_SIGMA_PRIOR_SCALE_W))

    # --- Forward model ----------------------------------------------------
    # Pack into the 7-vector in canonical order (matches kernels.PARAM_NAMES).
    theta = jnp.stack([A, Cd, Crr, eta_prop, eta_recup, c_HVAC, P_aux])

    P_model_W = forward(
        theta,
        speed_mps=data["speed_mps"],
        acceleration_mps2=data["acceleration_mps2"],
        mass_kg=data["mass_kg"],
        grade=data["grade"],
        temperature_K=data["temperature_K"],
    )

    # --- Likelihood -------------------------------------------------------
    # We use a plate to declare the n samples as i.i.d. given theta + sigma.
    # numpyro.plate is the right primitive here (not a python for-loop) because
    # it tells NUTS the samples are conditionally independent — the gradient
    # of the log-likelihood w.r.t. theta is then a simple sum over samples.
    n = P_model_W.shape[0]
    with numpyro.plate("samples", n):
        if sample_weights is None:
            numpyro.sample(
                "P_obs",
                dist.Normal(P_model_W, sigma),
                obs=observed_power_W,
            )
        else:
            # Weighted likelihood: scale each sample's log-density by w_i.
            # numpyro.factor adds a scalar to the log-joint without declaring
            # a new sampled variable, which is exactly what we need.
            log_lik = dist.Normal(P_model_W, sigma).log_prob(observed_power_W)
            numpyro.factor("weighted_log_lik", jnp.sum(sample_weights * log_lik))


__all__ = ["ztbus_model"]
