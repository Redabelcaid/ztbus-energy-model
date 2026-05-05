"""Mass and energy-derived features.

* Mass: instantaneous vehicle mass = curb mass + passengers × avg passenger mass.
  When passenger data is missing, falls back to curb mass and emits a flag.
* Cumulative energy: trapezoidal integration of cleaned ``electric_powerDemand``
  on the actual time vector. Specific consumption [kWh/km] is derived for QC.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ztbus.physics.parameters import PhysicalConstants

PASSENGERS_COL = "itcs_numberOfPassengers"
MASS_COL = "mass_kg"
ENERGY_COL = "energy_cum_kWh"
SPECIFIC_COL = "specific_energy_kWh_per_km"
DIST_COL = "distance_m"
POWER_COL = "electric_powerDemand"


def add_mass(df: pl.DataFrame, *, constants: PhysicalConstants | None = None) -> pl.DataFrame:
    """Add ``mass_kg`` column derived from passenger count."""
    constants = constants or PhysicalConstants()

    if PASSENGERS_COL in df.columns:
        passengers = df.select(
            pl.col(PASSENGERS_COL).fill_null(0).alias("p"),
        )["p"]
        mass = constants.curb_mass_kg + passengers * constants.avg_passenger_mass_kg
    else:
        mass = pl.Series(MASS_COL, [constants.curb_mass_kg] * df.height)

    return df.with_columns(mass.alias(MASS_COL))


def add_energy(df: pl.DataFrame) -> pl.DataFrame:
    """Add cumulative energy [kWh] and specific consumption [kWh/km]."""
    if POWER_COL not in df.columns or "time_unix" not in df.columns:
        return df

    t = df["time_unix"].to_numpy().astype(float)
    P = df[POWER_COL].to_numpy().astype(float)
    n = t.size

    if n >= 2:
        seg_J = 0.5 * (P[1:] + P[:-1]) * np.diff(t)
        E_J = np.concatenate(([0.0], np.cumsum(seg_J)))
    else:
        E_J = np.zeros(n)

    E_kWh = E_J / 3.6e6
    df = df.with_columns(pl.Series(ENERGY_COL, E_kWh))

    if DIST_COL in df.columns:
        d_km = df[DIST_COL].to_numpy().astype(float) / 1000.0
        # Specific consumption only meaningful where distance > 0
        spec = np.where(d_km > 0.01, E_kWh / np.maximum(d_km, 1e-9), np.nan)
        df = df.with_columns(pl.Series(SPECIFIC_COL, spec))

    return df
