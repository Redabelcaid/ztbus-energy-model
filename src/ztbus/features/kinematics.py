"""Kinematic features derived from the cleaned speed signal.

* Acceleration: central finite difference of the smoothed speed against the
  actual time vector. Central differences cancel the leading-order error of
  forward/backward differences and are well-defined on irregular time grids.
* Distance: cumulative trapezoidal integration of the smoothed speed. Using
  the SMOOTHED speed yields a stable distance series suitable as the abscissa
  for grade calculation.

We compute these AFTER cleaning so that the differentiation/integration acts
on a quality-controlled signal.
"""

from __future__ import annotations

import numpy as np
import polars as pl

SPEED_COL = "speed_smoothed_mps"
ACCEL_COL = "acceleration_mps2"
DIST_COL = "distance_m"


def add_kinematics(df: pl.DataFrame) -> pl.DataFrame:
    """Add acceleration [m/s²] and cumulative distance [m] columns."""
    if SPEED_COL not in df.columns or "time_unix" not in df.columns:
        return df

    t = df["time_unix"].to_numpy().astype(float)
    v = df[SPEED_COL].to_numpy().astype(float)
    n = t.size

    # ---- Acceleration (central difference) --------------------------------
    a = np.zeros(n, dtype=float)
    if n >= 3:
        a[1:-1] = (v[2:] - v[:-2]) / np.maximum(t[2:] - t[:-2], 1e-9)
        a[0] = (v[1] - v[0]) / max(t[1] - t[0], 1e-9)
        a[-1] = (v[-1] - v[-2]) / max(t[-1] - t[-2], 1e-9)
    elif n == 2:
        a[:] = (v[1] - v[0]) / max(t[1] - t[0], 1e-9)
    # n < 2 → leave a = 0

    # ---- Cumulative distance (trapezoidal) --------------------------------
    if n >= 2:
        seg = 0.5 * (v[1:] + v[:-1]) * np.diff(t)
        # Don't accumulate negative segments — clamping a tiny artifact-induced
        # negative speed * dt to zero avoids cumulative bias.
        seg = np.maximum(seg, 0.0)
        d = np.concatenate(([0.0], np.cumsum(seg)))
    else:
        d = np.zeros(n, dtype=float)

    return df.with_columns(
        pl.Series(ACCEL_COL, a),
        pl.Series(DIST_COL, d),
    )
