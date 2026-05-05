"""Typed parameter containers for the longitudinal powertrain model.

Two distinct concepts:

  - :class:`PhysicalConstants` holds quantities we treat as known (vehicle mass,
    air density, gravity, comfort temperature, motor power limit). These are
    NOT optimized.
  - :class:`PowertrainParameters` holds the quantities we identify from data
    (frontal area, drag coefficient, rolling-resistance coefficient,
    propulsion and recuperation efficiencies, HVAC coefficient, auxiliary load).

The split mirrors the strategic-model decomposition in Beckers et al. (2021)
and matches the placeholders left in the supervisor's
``updated_energy_calculation.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Known constants
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PhysicalConstants:
    """Constants treated as known; not optimization targets.

    Attributes
    ----------
    curb_mass_kg
        Empty vehicle mass.
    avg_passenger_mass_kg
        Used to derive an instantaneous mass estimate from the passenger count.
    rho_air_kg_per_m3
        Air density at sea level standard atmosphere.
    g_m_per_s2
        Gravitational acceleration.
    T_comfort_K
        Cabin setpoint used by the simple linear HVAC model.
    P_motor_max_W
        Combined motor power limit (used for QC, not as a soft constraint).
    """

    curb_mass_kg: float = 19000.0
    avg_passenger_mass_kg: float = 70.0
    rho_air_kg_per_m3: float = 1.225
    g_m_per_s2: float = 9.81
    T_comfort_K: float = 294.15
    P_motor_max_W: float = 320_000.0


# ---------------------------------------------------------------------------
# Parameters identified from data
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class PowertrainParameters:
    """Parameters identified from data.

    Default values match the priors in
    ``configs/physics/hess_lightram_19.yaml``; the optimizer should overwrite
    them. The class is intentionally mutable so the optimizer can update fields
    in-place across iterations.
    """

    frontal_area_m2: float = 8.0
    drag_coefficient: float = 0.65
    rolling_resistance_coefficient: float = 0.008
    efficiency_propulsion: float = 0.85
    efficiency_recuperation: float = 0.65
    hvac_coefficient_kW_per_K: float = 0.5
    auxiliary_power_kW: float = 12.0

    # ------------------------------------------------------------------
    # (de)serialization
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> PowertrainParameters:
        """Load priors / point estimates from a physics config file."""
        cfg = OmegaConf.load(Path(path))
        params = cfg.parameters  # type: ignore[union-attr]
        return cls(
            frontal_area_m2=float(params.frontal_area_m2.init),
            drag_coefficient=float(params.drag_coefficient.init),
            rolling_resistance_coefficient=float(params.rolling_resistance_coefficient.init),
            efficiency_propulsion=float(params.efficiency_propulsion.init),
            efficiency_recuperation=float(params.efficiency_recuperation.init),
            hvac_coefficient_kW_per_K=float(params.hvac_coefficient_kW_per_K.init),
            auxiliary_power_kW=float(params.auxiliary_power_kW.init),
        )

    def to_array(self) -> np.ndarray:
        """Pack into a numpy vector in field order — useful for scipy.optimize."""
        return np.array([getattr(self, f.name) for f in fields(self)], dtype=float)

    @classmethod
    def from_array(cls, x: np.ndarray) -> PowertrainParameters:
        """Inverse of :meth:`to_array`."""
        names = [f.name for f in fields(cls)]
        if len(x) != len(names):
            raise ValueError(f"Expected {len(names)} parameters, got {len(x)}.")
        return cls(**dict(zip(names, x.astype(float).tolist(), strict=True)))

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}
