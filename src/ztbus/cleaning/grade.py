"""Road grade derivation and cleaning.

Grade θ ≈ dh/ds (small-angle approximation) where h is altitude and s is
travelled distance. Both are derivatives of noisy signals, so the order of
operations matters:

1. Use the SMOOTHED altitude (from :func:`clean_altitude`).
2. Use cumulative distance from the SMOOTHED speed (computed in features).
3. Compute Δh / Δs.
4. Smooth the grade itself, because two noisy ratios still produce noise.
5. Flag values outside the plausibility envelope rather than clipping
   silently — Zurich routes are urban-flat for the most part, so values
   outside ±12 % almost always indicate GNSS artifacts.

Adds:

- ``grade``: float, [-1, 1] dimensionless, NaN where altitude was unavailable.
- ``grade_outlier_flag``: bool, where the value exceeded the bounds before flagging.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ztbus.cleaning.config import GradeConfig

ALT_COL = "altitude_smoothed_m"
DIST_COL = "distance_m"   # produced by features.kinematics
OUT_COL = "grade"


def derive_grade(df: pl.DataFrame, cfg: GradeConfig) -> pl.DataFrame:
    """Compute road grade from smoothed altitude and cumulative distance."""
    if ALT_COL not in df.columns or DIST_COL not in df.columns:
        return df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(OUT_COL),
            pl.lit(False).alias("grade_outlier_flag"),
        )

    h = df[ALT_COL].to_numpy().astype(float)
    s = df[DIST_COL].to_numpy().astype(float)

    dh = np.gradient(h)
    ds = np.gradient(s)

    grade = np.full_like(h, np.nan)
    valid = (np.abs(ds) > 1e-3) & np.isfinite(dh) & np.isfinite(ds)
    grade[valid] = dh[valid] / ds[valid]

    # Smooth the grade time series (rolling median)
    if cfg.smooth_after_derive and np.isfinite(grade).any():
        window = max(5, int(round(cfg.smoothing_window_seconds)))
        if window % 2 == 0:
            window += 1
        grade = _rolling_median_nan_aware(grade, window)

    lo, hi = cfg.plausibility_bounds
    flag = ~((grade >= lo) & (grade <= hi)) & np.isfinite(grade)

    return df.with_columns(
        pl.Series(OUT_COL, grade),
        pl.Series("grade_outlier_flag", flag),
    )


def _rolling_median_nan_aware(x: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling median that propagates NaN for all-NaN windows only."""
    n = x.size
    half = window // 2
    out = np.full_like(x, np.nan)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        block = x[lo:hi]
        finite = block[np.isfinite(block)]
        if finite.size > 0:
            out[i] = np.median(finite)
    return out
