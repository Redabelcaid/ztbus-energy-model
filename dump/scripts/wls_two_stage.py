"""WLS + sandwich variance for the two-stage ZTBus parameter identification.

Background
----------
The Stage 1 and Stage 2 models are *linear in the unknown parameters* once
we (a) move known kinematic terms to the LHS and (b) reparameterize the
inverse drivetrain efficiency: alpha = 1/eta_prop. Under linear-Gaussian
assumptions with N >> p, the maximum-likelihood estimator is the closed-form
ordinary least squares solution:

    beta_hat = (X^T X)^-1 X^T y

The Bayesian posterior under flat priors is N(beta_hat, V) with the same
beta_hat. We use the heteroskedasticity-consistent (HC0) sandwich variance:

    V = (X^T X)^-1  X^T diag(r_i^2) X  (X^T X)^-1

This is robust to misspecification of the error variance structure, which
matters here because our residuals are NOT iid Gaussian (driver behaviour,
friction-brake exclusion, HVAC non-linearity are heavy-tailed contributors).

Stage 1: mechanical force balance
    F_traction = m*g*grade + m*a  +  Crr*(m*g)  +  CdA*(0.5*rho*v^2)  +  eps

    Move known terms to LHS:
        y = F_traction - m*g*grade - m*a
        X = [ m*g,  0.5*rho*v^2 ]   ->   [Crr, CdA]

Stage 2: electrical, with grid-aware regen and the v < 15 km/h regen kill
    Let alpha = 1/eta_prop. Then:

    P_obs = alpha * P_traction
          + eta_grid * P_regen_grid
          + eta_batt * P_regen_batt
          + c_HVAC * |dT|*1000   (kW/K -> W)
          + P_aux * 1000          (kW -> W)

    where each component is 0 when the regime indicator is false (sample
    is not in that regime). The v<15 km/h regen samples contribute 0 to all
    three traction columns, leaving HVAC and P_aux to explain them.

Usage
-----
    # Quick smoke (one month, ~5s on CPU)
    uv run python dump/scripts/wls_two_stage.py --bus-id 183 --year-month 2021-07

    # Full corpus (~30s on CPU, both buses)
    uv run python dump/scripts/wls_two_stage.py --bus-id 183 --year-month ALL
    uv run python dump/scripts/wls_two_stage.py --bus-id 208 --year-month ALL

    # Side-by-side comparison against the v1.0 NUTS posterior
    uv run python dump/scripts/wls_two_stage.py \\
        --bus-id 183 --year-month ALL \\
        --compare-nuts /scratch/users/$USER/ztbus/reports/two_stage_full/5408071_b183
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("wls")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
log.info("JAX devices: %s", jax.devices())

import jax.numpy as jnp  # noqa: E402

from ztbus.optim.data import load_corpus  # noqa: E402

# Physical constants — mirror kernels.py / model_traction.py / model_electrical.py
G_M_PER_S2: float = 9.81
RHO_AIR_KG_PER_M3: float = 1.225
T_COMFORT_K: float = 294.15            # 21 °C
MIN_REGEN_SPEED_MPS: float = 15.0 / 3.6


# ============================================================================
# Closed-form WLS with HC0 sandwich variance
# ============================================================================
def fit_wls(X, y) -> dict[str, Any]:
    """Solve y = X beta + eps via OLS; return beta_hat + HC0 sandwich SEs.

    Returns
    -------
    beta_hat : (p,) array
    se       : (p,) array of heteroskedasticity-consistent (HC0) standard errors
    sigma    : scalar residual RMS
    cov      : (p, p) sandwich covariance matrix
    """
    # solve(X^T X, X^T y) is numerically preferable to (X^T X)^{-1} X^T y
    XtX = X.T @ X
    Xty = X.T @ y
    beta_hat = jnp.linalg.solve(XtX, Xty)

    residuals = y - X @ beta_hat
    sigma = jnp.sqrt(jnp.mean(residuals**2))

    XtX_inv = jnp.linalg.inv(XtX)
    # Meat of sandwich: X^T diag(r^2) X computed without forming diag(r^2)
    meat = (X.T * residuals**2) @ X
    cov = XtX_inv @ meat @ XtX_inv
    se = jnp.sqrt(jnp.diag(cov))

    return {"beta_hat": beta_hat, "se": se, "sigma": sigma, "cov": cov}


# ============================================================================
# Stage 1: mechanical force balance
# ============================================================================
def stage1_wls(arrays: dict[str, Any]) -> dict[str, Any]:
    m = arrays["mass_kg"]
    a = arrays["acceleration_mps2"]
    grade = arrays["grade"]
    v = arrays["speed_mps"]
    F = arrays["F_traction_N"]

    # Move known kinematic terms to LHS (no parameters)
    y = F - m * G_M_PER_S2 * grade - m * a

    # Design matrix: columns are coefficients of [Crr, CdA]
    X = jnp.column_stack(
        [
            m * G_M_PER_S2,                         # Crr column
            0.5 * RHO_AIR_KG_PER_M3 * v**2,         # CdA column
        ]
    )

    out = fit_wls(X, y)
    beta = out["beta_hat"]
    se = out["se"]
    return {
        "Crr": (float(beta[0]), float(se[0])),
        "CdA": (float(beta[1]), float(se[1])),
        "sigma_N": float(out["sigma"]),
        "n_obs": int(X.shape[0]),
        "cov": out["cov"],
    }


# ============================================================================
# Stage 2: electrical, with grid-aware regen and v<15 km/h regen kill
# ============================================================================
def stage2_wls(arrays: dict[str, Any]) -> dict[str, Any]:
    F = arrays["F_traction_N"]
    v = arrays["speed_mps"]
    T = arrays["temperature_K"]
    grid_available = arrays["grid_available"] > 0.5

    P_mech = F * v

    # Regime masks
    is_traction = P_mech >= 0.0
    can_regen = (P_mech < 0.0) & (v >= MIN_REGEN_SPEED_MPS)
    is_regen_grid = can_regen & grid_available
    is_regen_batt = can_regen & ~grid_available
    # v < 15 km/h regen samples implicitly get 0 in all three traction columns

    # Design matrix.
    # Columns -> parameter coefficients in:
    #   [alpha = 1/eta_prop, eta_recup_grid, eta_recup_battery, c_HVAC, P_aux]
    # Note the *1000 scaling so c_HVAC comes out in kW/K and P_aux in kW.
    X = jnp.column_stack(
        [
            jnp.where(is_traction,     P_mech, 0.0),  # alpha
            jnp.where(is_regen_grid,   P_mech, 0.0),  # eta_recup_grid
            jnp.where(is_regen_batt,   P_mech, 0.0),  # eta_recup_battery
            jnp.abs(T - T_COMFORT_K) * 1000.0,         # c_HVAC (kW/K)
            jnp.ones_like(P_mech) * 1000.0,            # P_aux (kW)
        ]
    )

    y = arrays["P_obs_W"]
    out = fit_wls(X, y)
    beta = out["beta_hat"]
    se = out["se"]

    # eta_prop from alpha via delta method:  Var(1/a) ≈ Var(a) / a^4
    alpha, alpha_se = beta[0], se[0]
    eta_prop = 1.0 / alpha
    eta_prop_se = alpha_se / (alpha**2)

    return {
        "eta_prop":          (float(eta_prop),    float(eta_prop_se)),
        "eta_recup_grid":    (float(beta[1]),     float(se[1])),
        "eta_recup_battery": (float(beta[2]),     float(se[2])),
        "c_HVAC":            (float(beta[3]),     float(se[3])),
        "P_aux":             (float(beta[4]),     float(se[4])),
        "sigma_W":           float(out["sigma"]),
        "n_obs":             int(X.shape[0]),
        "cov":               out["cov"],
        "alpha":             (float(alpha), float(alpha_se)),
    }


# ============================================================================
# Optional comparison against the NUTS posteriors we already have
# ============================================================================
def compare_to_nuts(nuts_dir: Path, wls_s1: dict, wls_s2: dict) -> None:
    import polars as pl  # local import — only needed for comparison

    s1_pq = nuts_dir / "stage1_posterior_summary.parquet"
    s2_pq = nuts_dir / "stage2_posterior_summary.parquet"
    if not s1_pq.exists() or not s2_pq.exists():
        log.warning("NUTS posteriors not found in %s — skipping comparison", nuts_dir)
        return

    s1 = pl.read_parquet(s1_pq)
    s2 = pl.read_parquet(s2_pq)

    def _row(df: pl.DataFrame, name: str) -> tuple[float, float]:
        r = df.filter(pl.col("parameter") == name)
        return float(r["mean"][0]), float(r["sd"][0])

    def _fmt(label: str, nuts: tuple[float, float], wls: tuple[float, float],
             fmt: str = "{:.6f}") -> str:
        nm, ns = nuts
        wm, ws = wls
        diff = wm - nm
        diff_in_sd = abs(diff) / max(ns, 1e-12)
        return (f"  {label:<20s} NUTS={fmt.format(nm)} +-{fmt.format(ns)}   "
                f"WLS={fmt.format(wm)} +-{fmt.format(ws)}   "
                f"|diff|={fmt.format(abs(diff))} ({diff_in_sd:.2f} NUTS-sigma)")

    log.info("")
    log.info("=" * 78)
    log.info("WLS vs. NUTS comparison")
    log.info("=" * 78)
    log.info("Stage 1 (mechanical):")
    log.info(_fmt("Crr",     _row(s1, "Crr"), wls_s1["Crr"]))
    log.info(_fmt("CdA",     _row(s1, "CdA"), wls_s1["CdA"], fmt="{:.4f}"))
    log.info("Stage 2 (electrical):")
    log.info(_fmt("eta_prop",          _row(s2, "eta_prop"),          wls_s2["eta_prop"]))
    log.info(_fmt("eta_recup_grid",    _row(s2, "eta_recup_grid"),    wls_s2["eta_recup_grid"]))
    log.info(_fmt("eta_recup_battery", _row(s2, "eta_recup_battery"), wls_s2["eta_recup_battery"]))
    log.info(_fmt("c_HVAC",             _row(s2, "c_HVAC"),             wls_s2["c_HVAC"], fmt="{:.4f}"))
    log.info(_fmt("P_aux",              _row(s2, "P_aux"),              wls_s2["P_aux"], fmt="{:.4f}"))
    log.info(f"  sigma_W              NUTS={_row(s2, 'sigma_W')[0]:.0f}   "
             f"WLS={wls_s2['sigma_W']:.0f}   "
             f"|diff|={abs(_row(s2, 'sigma_W')[0] - wls_s2['sigma_W']):.0f} W")


# ============================================================================
# Driver
# ============================================================================
def _year_months_from_arg(arg: str) -> tuple[str, ...]:
    if arg == "ALL_2021":
        return tuple(f"2021-{m:02d}" for m in range(1, 13))
    if arg == "ALL":
        ym = []
        for year in (2019, 2020, 2021, 2022):
            for month in range(1, 13):
                if year == 2019 and month < 5:
                    continue
                ym.append(f"{year}-{month:02d}")
        return tuple(ym)
    return (arg,)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-month", default="2021-07",
                    help="YYYY-MM, 'ALL_2021', or 'ALL' (May 2019 - Dec 2022)")
    ap.add_argument("--bus-id", default="183")
    ap.add_argument("--subsample", type=int, default=-1,
                    help="row cap; -1 = full data (default)")
    ap.add_argument("--compare-nuts", default=None,
                    help="directory with stage{1,2}_posterior_summary.parquet "
                         "to compare WLS estimates against")
    ap.add_argument("--output-dir", default=None,
                    help="optional: write Polars parquet summaries here")
    args = ap.parse_args()

    PROCESSED = Path(os.environ.get(
        "ZTBUS_PROCESSED_DIR",
        "/scratch/users/rbelcaid/ztbus/processed",
    ))
    year_months = _year_months_from_arg(args.year_month)
    log.info("Bus: %s   Months: %d (%s..%s)   Subsample: %s",
             args.bus_id, len(year_months), year_months[0], year_months[-1],
             "(no cap)" if args.subsample == -1 else f"{args.subsample:,}")

    # ---- Load ------------------------------------------------------------
    t0 = time.time()
    arrays, _ = load_corpus(
        PROCESSED,
        bus_ids=(args.bus_id,),
        year_months=year_months,
        subsample=None if args.subsample == -1 else args.subsample,
        subsample_seed=0,
    )
    n = arrays["speed_mps"].shape[0]
    log.info("Loaded %s samples in %.1f s", f"{n:,}", time.time() - t0)
    log.info("")

    # ---- Stage 1 ---------------------------------------------------------
    log.info("=" * 78)
    log.info("STAGE 1 (WLS): mechanical fit on traction_tractionForce")
    log.info("=" * 78)
    t0 = time.time()
    s1 = stage1_wls(arrays)
    s1_time = time.time() - t0
    log.info("Wall time: %.3f s   (compare: NUTS Stage 1 wall ~3-15 min)", s1_time)
    log.info("  Crr     = %.6f +- %.6f", *s1["Crr"])
    log.info("  CdA     = %.4f +- %.4f  m^2", *s1["CdA"])
    log.info("  sigma_N = %.0f  N", s1["sigma_N"])
    log.info("")

    # ---- Stage 2 ---------------------------------------------------------
    log.info("=" * 78)
    log.info("STAGE 2 (WLS): electrical fit with grid-aware regen split")
    log.info("=" * 78)
    t0 = time.time()
    s2 = stage2_wls(arrays)
    s2_time = time.time() - t0
    log.info("Wall time: %.3f s   (compare: NUTS Stage 2 wall ~5-25 min)", s2_time)
    log.info("  eta_prop          = %.6f +- %.6f", *s2["eta_prop"])
    log.info("  eta_recup_grid    = %.6f +- %.6f", *s2["eta_recup_grid"])
    log.info("  eta_recup_battery = %.6f +- %.6f", *s2["eta_recup_battery"])
    log.info("  c_HVAC            = %.4f +- %.4f  kW/K", *s2["c_HVAC"])
    log.info("  P_aux             = %.4f +- %.4f  kW",   *s2["P_aux"])
    log.info("  sigma_W           = %.0f  W", s2["sigma_W"])
    log.info("")
    log.info("Total WLS wall: %.3f s", s1_time + s2_time)

    # ---- Optional: compare to NUTS posteriors ----------------------------
    if args.compare_nuts:
        compare_to_nuts(Path(args.compare_nuts), s1, s2)

    # ---- Optional: write parquet summaries -------------------------------
    if args.output_dir:
        import polars as pl

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        def _summary(name: str, value: float, se: float) -> dict:
            return {"parameter": name, "mean": value, "se_sandwich": se}

        pl.DataFrame([
            _summary("Crr",     *s1["Crr"]),
            _summary("CdA",     *s1["CdA"]),
            {"parameter": "sigma_N", "mean": s1["sigma_N"], "se_sandwich": None},
        ]).write_parquet(out_dir / "stage1_wls_summary.parquet")

        pl.DataFrame([
            _summary("eta_prop",          *s2["eta_prop"]),
            _summary("eta_recup_grid",    *s2["eta_recup_grid"]),
            _summary("eta_recup_battery", *s2["eta_recup_battery"]),
            _summary("c_HVAC",            *s2["c_HVAC"]),
            _summary("P_aux",             *s2["P_aux"]),
            {"parameter": "sigma_W", "mean": s2["sigma_W"], "se_sandwich": None},
        ]).write_parquet(out_dir / "stage2_wls_summary.parquet")

        log.info("Wrote: %s", out_dir)

    log.info("Done.")


if __name__ == "__main__":
    main()
