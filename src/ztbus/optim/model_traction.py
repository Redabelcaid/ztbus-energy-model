"""Stage 1 model: identify Cd*A and Crr from the directly-measured traction force.

The ZTBus dataset provides ``traction_tractionForce`` (N), an estimate of the
total traction force from the two motors. Fitting the longitudinal force
balance against this mechanical signal eliminates electrical-domain noise
that contaminates the joint fit.

Stage 1 model:
    F_traction ~ Normal(F_pred, sigma_N)
    F_pred = m*g*grade + m*a + Crr*m*g + CdA*0.5*rho*v^2

Cd and A are physically degenerate (only the product CdA appears); we
identify the product as a single parameter.
"""

from __future__ import annotations

from typing import Any

import numpyro
import numpyro.distributions as dist

G_M_PER_S2: float = 9.81
RHO_AIR_KG_PER_M3: float = 1.225

_CRR_LO: float = 0.005
_CRR_HI: float = 0.025
_CDA_LO: float = 2.0
_CDA_HI: float = 8.0
_SIGMA_SCALE_N: float = 20_000.0


def model(arrays: dict[str, Any], observed_F_N: Any) -> None:
    """NumPyro Stage-1 model."""
    Crr = numpyro.sample("Crr", dist.Uniform(_CRR_LO, _CRR_HI))
    CdA = numpyro.sample("CdA", dist.Uniform(_CDA_LO, _CDA_HI))
    sigma_N = numpyro.sample("sigma_N", dist.HalfNormal(_SIGMA_SCALE_N))

    m = arrays["mass_kg"]
    a = arrays["acceleration_mps2"]
    g_road = arrays["grade"]
    v = arrays["speed_mps"]

    F_pred = (
        m * G_M_PER_S2 * g_road
        + m * a
        + Crr * m * G_M_PER_S2
        + CdA * 0.5 * RHO_AIR_KG_PER_M3 * v**2
    )

    numpyro.sample("obs_F", dist.Normal(F_pred, sigma_N), obs=observed_F_N)


PARAM_NAMES: tuple[str, ...] = ("Crr", "CdA", "sigma_N")
