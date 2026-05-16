"""Stage 2 model: eta_prop, eta_recup, c_HVAC, P_aux given F_traction.

With mechanical traction power P_mech = F_traction * v as a KNOWN input,
the remaining unknowns are purely electrical-domain. P_aux now has its own
signal (the constant offset between predicted electrical-traction and
observed electric_powerDemand) and should be identifiable.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

T_COMFORT_K: float = 294.15
MIN_REGEN_SPEED_MPS: float = 15.0 / 3.6

_ETA_PROP_LO: float = 0.70
_ETA_PROP_HI: float = 0.95
_ETA_RECUP_LO: float = 0.30
_ETA_RECUP_HI: float = 0.95
_C_HVAC_SCALE: float = 1.0
_P_AUX_PRIOR_MEAN_KW: float = 4.0
_P_AUX_PRIOR_SD_KW: float = 1.5
_P_AUX_LO_KW: float = 0.5
_P_AUX_HI_KW: float = 12.0
_SIGMA_SCALE_W: float = 25_000.0


def model(arrays: dict[str, Any], observed_P_W: Any) -> None:
    """NumPyro Stage-2 model."""
    eta_prop = numpyro.sample("eta_prop", dist.Uniform(_ETA_PROP_LO, _ETA_PROP_HI))
    eta_recup = numpyro.sample("eta_recup", dist.Uniform(_ETA_RECUP_LO, _ETA_RECUP_HI))
    c_HVAC = numpyro.sample("c_HVAC", dist.HalfNormal(_C_HVAC_SCALE))
    P_aux = numpyro.sample(
        "P_aux",
        dist.TruncatedNormal(
            _P_AUX_PRIOR_MEAN_KW,
            _P_AUX_PRIOR_SD_KW,
            low=_P_AUX_LO_KW,
            high=_P_AUX_HI_KW,
        ),
    )
    sigma_W = numpyro.sample("sigma_W", dist.HalfNormal(_SIGMA_SCALE_W))

    F = arrays["F_traction_N"]
    v = arrays["speed_mps"]
    T = arrays["temperature_K"]

    P_mech = F * v
    is_traction = P_mech >= 0.0
    regen_active = (P_mech < 0.0) & (v >= MIN_REGEN_SPEED_MPS)

    P_elec_traction = jnp.where(
        is_traction,
        P_mech / eta_prop,
        jnp.where(regen_active, P_mech * eta_recup, 0.0),
    )

    P_hvac_W = c_HVAC * jnp.abs(T - T_COMFORT_K) * 1000.0
    P_aux_W = P_aux * 1000.0

    P_pred = P_elec_traction + P_hvac_W + P_aux_W

    numpyro.sample("obs_P", dist.Normal(P_pred, sigma_W), obs=observed_P_W)


PARAM_NAMES: tuple[str, ...] = ("eta_prop", "eta_recup", "c_HVAC", "P_aux", "sigma_W")
