"""Build the two-stage fit pipeline that uses traction_tractionForce.

What this creates / modifies on the cluster
-------------------------------------------

1. Verifies that traction_tractionForce is present in the cleaned parquets
   (fails loud with remediation instructions if not).

2. Patches src/ztbus/optim/data.py:
   - Adds 'traction_tractionForce' to the required schema columns.
   - Adds 'F_traction_N' to the arrays dict produced by load_corpus.
   - Drops samples where F_traction is null.

3. Creates src/ztbus/optim/model_traction.py:
   - Stage 1 NumPyro model: fits F_traction = m·g·θ + m·a + Crr·m·g
     + CdA·½·ρ·v² with parameters {Crr, CdA, sigma_N}.

4. Creates src/ztbus/optim/model_electrical.py:
   - Stage 2 NumPyro model: with P_mech = F_traction · v as known input,
     fits P_obs = P_mech/eta_prop (if traction) or P_mech·eta_recup
     (if regen, v ≥ 15 km/h) or 0 (if regen, v < 15 km/h), plus
     c_HVAC·|ΔT| + P_aux. Parameters: {eta_prop, eta_recup, c_HVAC,
     P_aux, sigma_W}.

5. Creates dump/scripts/two_stage_fit.py:
   - Driver that loads the corpus, runs stage 1, then stage 2.
   - CLI: --year-month, --bus-id, --subsample
   - Writes posteriors for both stages with a stage prefix.

Why this resolves the identifiability problems
----------------------------------------------

The joint fit had two unidentifiable parameters:

* Cd × A appear only as a product in the aero force term. Both Stage 1
  and the joint fit have this degeneracy — but Stage 1 isolates the
  product and identifies it tightly without electrical-domain noise.

* P_aux is a small constant offset on driving samples when traction
  dominates. In the joint fit it absorbs miscellaneous bias from the
  propulsion side. In Stage 2 the propulsion contribution is a KNOWN
  input (F_traction × v from the dataset), so P_aux is identified
  directly from samples where the propulsion-electrical balance
  doesn't match — which is exactly the auxiliary load.

Apply on cluster:
    uv run python dump/scripts/build_two_stage_fit.py
"""

import sys
from pathlib import Path
from textwrap import dedent

import polars as pl

REPO = Path(".").resolve()

# ===========================================================================
# Step 0: verify traction_tractionForce is in the cleaned parquets
# ===========================================================================

PROCESSED_DIR = Path("/scratch/users/rbelcaid/ztbus/processed")
sample_pq = next(
    (p for p in PROCESSED_DIR.rglob("*.parquet")
     if "qc_summary" not in p.name and "summary" not in p.name),
    None,
)
if sample_pq is None:
    sys.exit(f"ERROR: no parquet files found under {PROCESSED_DIR}")

df_head = pl.scan_parquet(str(sample_pq)).head(1).collect()
cols = set(df_head.columns)

print(f"Inspected: {sample_pq.relative_to(PROCESSED_DIR)}")
print(f"Total columns: {len(cols)}")

if "traction_tractionForce" not in cols:
    sys.exit(dedent(f"""
        ERROR: 'traction_tractionForce' column not found in processed parquets.
        Available columns: {sorted(cols)}

        The cleaning pipeline dropped this column. To proceed:
          1. Edit src/ztbus/cleaning/pipeline.py — add 'traction_tractionForce'
             to the list of columns preserved through cleaning.
          2. Re-run cleaning:  uv run ztbus clean
          3. Re-run this build script.

        Estimated re-clean time: ~22 min for the full corpus.
    """))

print("✓ traction_tractionForce found in cleaned parquets")

# Quick value-sanity check on a single mission
df_full = pl.scan_parquet(str(sample_pq)).collect()
n_total = df_full.shape[0]
F = df_full["traction_tractionForce"]
n_null = F.null_count()
print(f"  Rows in {sample_pq.name}: {n_total}")
print(f"  Null F_traction: {n_null} ({100 * n_null / n_total:.1f}%)")
if n_null < n_total:
    F_valid = F.drop_nulls()
    print(f"  F_traction range: [{F_valid.min():.0f}, {F_valid.max():.0f}] N")
    print(f"  F_traction mean:  {F_valid.mean():.0f} N")
    print(f"  F_traction std:   {F_valid.std():.0f} N")
print()


# ===========================================================================
# Step 1: patch data.py to expose F_traction_N
# ===========================================================================

DATA_PY = REPO / "src/ztbus/optim/data.py"
text = DATA_PY.read_text()

# 1a. Add F_traction key constant near _JAX_OBS_KEY
old_keys_marker = '_JAX_OBS_KEY = "P_obs_W"'
new_keys_marker = '_JAX_OBS_KEY = "P_obs_W"\n_JAX_F_TRACTION_KEY = "F_traction_N"'
if "_JAX_F_TRACTION_KEY" not in text:
    assert old_keys_marker in text, "_JAX_OBS_KEY anchor not found in data.py"
    text = text.replace(old_keys_marker, new_keys_marker)
    print("✓ data.py: added _JAX_F_TRACTION_KEY constant")
else:
    print("  data.py: _JAX_F_TRACTION_KEY already present, skipping")

# 1b. Add traction_tractionForce to the column list read from parquet
#     The exact anchor depends on existing code — try several patterns.
column_marker_candidates = [
    '"electric_powerDemand",\n        "altitude_smoothed_m",',
    '"electric_powerDemand",\n    "altitude_smoothed_m",',
]
patched_col = False
for marker in column_marker_candidates:
    if marker in text:
        new_marker = marker.replace(
            '"electric_powerDemand",',
            '"electric_powerDemand",\n        "traction_tractionForce",',
            1,
        )
        text = text.replace(marker, new_marker, 1)
        patched_col = True
        print("✓ data.py: added traction_tractionForce to read columns")
        break
if not patched_col:
    print("⚠ data.py: couldn't find column-list anchor; you may need to add")
    print("  'traction_tractionForce' manually to the parquet scan column list.")

# 1c. Add a null-drop filter for traction_tractionForce in the audit cascade
old_drop_nulls = '("drop_nulls", lambda l: l.drop_nulls()),'
new_drop_nulls = (
    '("drop_nulls (incl. F_traction)", lambda l: l.drop_nulls()),'
)
if "drop_nulls (incl. F_traction)" not in text:
    if old_drop_nulls in text:
        text = text.replace(old_drop_nulls, new_drop_nulls)
        print("✓ data.py: relabeled drop_nulls step (now also drops null F_traction)")

# 1d. Add F_traction to the arrays dict at the end of load_corpus
#     Look for the arrays={...} construction and inject our new key.
old_arrays = '''    arrays[_JAX_OBS_KEY] = jnp.asarray(df[_PARQUET_OBS_COLUMN].to_numpy(), dtype=dtype)'''
new_arrays = '''    arrays[_JAX_OBS_KEY] = jnp.asarray(df[_PARQUET_OBS_COLUMN].to_numpy(), dtype=dtype)
    arrays[_JAX_F_TRACTION_KEY] = jnp.asarray(
        df["traction_tractionForce"].to_numpy(), dtype=dtype,
    )'''
if "_JAX_F_TRACTION_KEY] = " not in text:
    assert old_arrays in text, "arrays-construction anchor not found in data.py"
    text = text.replace(old_arrays, new_arrays, 1)
    print("✓ data.py: arrays dict now exposes F_traction_N")
else:
    print("  data.py: arrays dict already has F_traction_N, skipping")

DATA_PY.write_text(text)


# ===========================================================================
# Step 2: write src/ztbus/optim/model_traction.py (Stage 1)
# ===========================================================================

MODEL_TRACTION_PY = REPO / "src/ztbus/optim/model_traction.py"
MODEL_TRACTION_PY.write_text('''"""Stage 1 model: identify Cd*A and Crr from the directly-measured traction force.

Background
----------
The ZTBus dataset provides ``traction_tractionForce`` (N), an estimate of the
total traction force from the two motors derived from the motor torque and
the compound transmission ratio. This signal is the mechanical traction
force only — electrical losses, HVAC, and auxiliary loads are absent.

Fitting the longitudinal force balance against this clean signal eliminates
two sources of bias that the joint electrical-domain fit suffered from:

  1. The Cd × A degeneracy is still present (only the product enters the
     physics), but it is cleanly identified without any contamination from
     unmodelled HVAC or auxiliary loads.
  2. Crr is identifiable independently because mass varies with passenger
     count, breaking the Crr-vs-constant-offset degeneracy.

Stage 1 model
-------------
    F_traction ~ Normal(F_pred, sigma_N)

    F_pred = m·g·grade + m·a + Crr·m·g + CdA·½·ρ·v²

Parameters: {Crr, CdA, sigma_N}.

Posterior on CdA is reported as a single number (effective drag area, m^2);
the model does not attempt to separate Cd and A.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

# Physical constants (mirror kernels.py)
G_M_PER_S2: float = 9.81
RHO_AIR_KG_PER_M3: float = 1.225

# Priors — broad uniform over physically plausible ranges
_CRR_LO: float = 0.005
_CRR_HI: float = 0.025
_CDA_LO: float = 2.0   # m^2; Cd=0.3 × A=7 (low end)
_CDA_HI: float = 8.0   # m^2; Cd=0.9 × A=9 (high end)
_SIGMA_SCALE_N: float = 20_000.0


def model(arrays: dict, observed_F_N) -> None:
    """NumPyro Stage-1 model: F_traction = mechanical force balance."""
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
''')
print(f"✓ wrote {MODEL_TRACTION_PY.relative_to(REPO)}")


# ===========================================================================
# Step 3: write src/ztbus/optim/model_electrical.py (Stage 2)
# ===========================================================================

MODEL_ELECTRICAL_PY = REPO / "src/ztbus/optim/model_electrical.py"
MODEL_ELECTRICAL_PY.write_text('''"""Stage 2 model: identify eta_prop, eta_recup, c_HVAC, P_aux given F_traction.

With the mechanical traction force from the dataset, the mechanical traction
power P_mech = F_traction × v is a KNOWN per-sample input. The remaining
unknowns are purely electrical-domain: the propulsion / recuperation
efficiencies, the HVAC coefficient, and the constant auxiliary load.

P_obs = (P_mech / eta_prop)   if  P_mech >= 0           (traction)
        (P_mech · eta_recup)  if  P_mech <  0  and v >= 15 km/h (regen)
         0                    if  P_mech <  0  and v <  15 km/h (regen-killed)
       + c_HVAC · |T - 21°C| · 1000  (W)
       + P_aux · 1000               (W)
       + noise ~ Normal(0, sigma_W)

By making P_mech a known input rather than something the model has to
predict via parameters, this stage breaks the P_aux unidentifiability that
plagued the joint fit. P_aux now has its own signal: samples where
electric_powerDemand differs from the electrical-traction prediction by a
roughly constant offset, regardless of P_mech magnitude.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

# Physical constants
T_COMFORT_K: float = 294.15        # 21 °C
MIN_REGEN_SPEED_MPS: float = 15.0 / 3.6   # 4.166... m/s

# Priors — narrow, physically motivated
_ETA_PROP_LO: float = 0.70
_ETA_PROP_HI: float = 0.95
_ETA_RECUP_LO: float = 0.30
_ETA_RECUP_HI: float = 0.95
_C_HVAC_SCALE: float = 1.0     # HalfNormal scale, kW/K
_P_AUX_PRIOR_MEAN_KW: float = 4.0
_P_AUX_PRIOR_SD_KW: float = 1.5
_P_AUX_LO_KW: float = 0.5
_P_AUX_HI_KW: float = 12.0
_SIGMA_SCALE_W: float = 25_000.0


def model(arrays: dict, observed_P_W) -> None:
    """NumPyro Stage-2 model: electric power = traction-electric + HVAC + aux."""
    eta_prop = numpyro.sample(
        "eta_prop", dist.Uniform(_ETA_PROP_LO, _ETA_PROP_HI),
    )
    eta_recup = numpyro.sample(
        "eta_recup", dist.Uniform(_ETA_RECUP_LO, _ETA_RECUP_HI),
    )
    c_HVAC = numpyro.sample(
        "c_HVAC", dist.HalfNormal(_C_HVAC_SCALE),
    )
    P_aux = numpyro.sample(
        "P_aux",
        dist.TruncatedNormal(
            _P_AUX_PRIOR_MEAN_KW, _P_AUX_PRIOR_SD_KW,
            low=_P_AUX_LO_KW, high=_P_AUX_HI_KW,
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


PARAM_NAMES: tuple[str, ...] = (
    "eta_prop", "eta_recup", "c_HVAC", "P_aux", "sigma_W",
)
''')
print(f"✓ wrote {MODEL_ELECTRICAL_PY.relative_to(REPO)}")


# ===========================================================================
# Step 4: write dump/scripts/two_stage_fit.py (driver)
# ===========================================================================

DRIVER_PY = REPO / "dump/scripts/two_stage_fit.py"
DRIVER_PY.write_text('''"""Two-stage fit driver: Stage 1 (mechanical) then Stage 2 (electrical).

Stage 1 uses traction_tractionForce as the target to identify Crr and CdA.
Stage 2 uses electric_powerDemand as the target with F_traction · v
as a known mechanical-power input to identify eta_prop, eta_recup,
c_HVAC, P_aux.

Usage:
    # Default smoke: bus 183, July 2021, 20k subsample
    uv run python dump/scripts/two_stage_fit.py

    # Full data on July 2021
    uv run python dump/scripts/two_stage_fit.py --year-month 2021-07 --subsample -1

    # Production on full 2021
    uv run python dump/scripts/two_stage_fit.py \\
        --year-month ALL_2021 --subsample -1 --num-warmup 1000 --num-samples 2000
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("two_stage")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
log.info("JAX devices: %s", jax.devices())

from ztbus.optim.data import load_corpus  # noqa: E402
from ztbus.optim import model_traction, model_electrical  # noqa: E402
from ztbus.optim.samplers import (  # noqa: E402
    nuts_fit_generic, posterior_summary, posterior_to_dataframe,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-month", default="2021-07",
                    help="YYYY-MM or 'ALL_2021' (default: 2021-07)")
    ap.add_argument("--bus-id", default="183")
    ap.add_argument("--subsample", type=int, default=20_000,
                    help="row cap; -1 for no cap")
    ap.add_argument("--num-warmup", type=int, default=300)
    ap.add_argument("--num-samples", type=int, default=300)
    ap.add_argument("--num-chains", type=int, default=2)
    args = ap.parse_args()

    PROCESSED = Path(os.environ.get(
        "ZTBUS_PROCESSED_DIR",
        "/scratch/users/rbelcaid/ztbus/processed",
    ))

    if args.year_month == "ALL_2021":
        year_months = tuple(f"2021-{m:02d}" for m in range(1, 13))
    else:
        year_months = (args.year_month,)

    log.info("Bus: %s   Months: %s   Subsample: %s",
             args.bus_id, year_months,
             "(no cap)" if args.subsample == -1 else args.subsample)

    t0 = time.time()
    arrays, _ = load_corpus(
        PROCESSED,
        bus_ids=(args.bus_id,),
        year_months=year_months,
        subsample=None if args.subsample == -1 else args.subsample,
        subsample_seed=0,
    )
    n = arrays["speed_mps"].shape[0]
    log.info("Loaded %d samples in %.1f s", n, time.time() - t0)

    out_dir = Path(f"/scratch/users/{os.environ.get('USER', 'rbelcaid')}/"
                   f"ztbus/reports/two_stage/{args.bus_id}_{args.year_month}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # =================================================================
    # Stage 1: mechanical fit on traction_tractionForce
    # =================================================================
    log.info("=" * 70)
    log.info("STAGE 1: mechanical fit on traction_tractionForce")
    log.info("=" * 70)

    result_1 = nuts_fit_generic(
        model_fn=model_traction.model,
        model_args=(arrays,),
        observed_kwarg="observed_F_N",
        observed=arrays["F_traction_N"],
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        chain_method="vectorized",
        progress_bar=True,
        rng_seed=0,
    )
    s1 = posterior_summary(result_1.samples)
    log.info("\\n=== Stage 1 posterior ===\\n%s", s1)
    log.info("R-hat max: %.4f  ESS min: %.0f  divergent: %d  wall: %.1f s",
             result_1.diagnostics["r_hat_max"],
             result_1.diagnostics["ess_bulk_min"],
             result_1.diagnostics["num_divergent"],
             result_1.diagnostics["wall_seconds"])
    s1.write_parquet(out_dir / "stage1_posterior_summary.parquet")
    posterior_to_dataframe(result_1.samples).write_parquet(
        out_dir / "stage1_posterior_samples.parquet",
    )

    # =================================================================
    # Stage 2: electrical fit on electric_powerDemand
    # =================================================================
    log.info("=" * 70)
    log.info("STAGE 2: electrical fit on electric_powerDemand")
    log.info("=" * 70)

    result_2 = nuts_fit_generic(
        model_fn=model_electrical.model,
        model_args=(arrays,),
        observed_kwarg="observed_P_W",
        observed=arrays["P_obs_W"],
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        chain_method="vectorized",
        progress_bar=True,
        rng_seed=0,
    )
    s2 = posterior_summary(result_2.samples)
    log.info("\\n=== Stage 2 posterior ===\\n%s", s2)
    log.info("R-hat max: %.4f  ESS min: %.0f  divergent: %d  wall: %.1f s",
             result_2.diagnostics["r_hat_max"],
             result_2.diagnostics["ess_bulk_min"],
             result_2.diagnostics["num_divergent"],
             result_2.diagnostics["wall_seconds"])
    s2.write_parquet(out_dir / "stage2_posterior_summary.parquet")
    posterior_to_dataframe(result_2.samples).write_parquet(
        out_dir / "stage2_posterior_samples.parquet",
    )

    log.info("Wrote: %s", out_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()
''')
print(f"✓ wrote {DRIVER_PY.relative_to(REPO)}")


# ===========================================================================
# Step 5: ensure samplers.py exposes a generic nuts_fit
# ===========================================================================

SAMPLERS_PY = REPO / "src/ztbus/optim/samplers.py"
samp_text = SAMPLERS_PY.read_text()

if "def nuts_fit_generic(" not in samp_text:
    # Add a generic wrapper that accepts any (model_fn, model_args, observed) triple.
    generic_fn = '''


def nuts_fit_generic(
    *,
    model_fn,
    model_args: tuple,
    observed_kwarg: str,
    observed,
    num_warmup: int = 300,
    num_samples: int = 300,
    num_chains: int = 2,
    chain_method: str = "sequential",
    progress_bar: bool = True,
    rng_seed: int = 0,
    target_accept_prob: float = 0.85,
):
    """Generic NUTS fit for any (model_fn, observed_kwarg) pair.

    Lets us reuse the same machinery for Stage 1 (model_traction, observed_F_N)
    and Stage 2 (model_electrical, observed_P_W) without duplicating the
    sampler boilerplate.
    """
    import time

    import jax.random as jrandom
    import numpyro
    from numpyro.infer import MCMC, NUTS

    kernel = NUTS(model_fn, target_accept_prob=target_accept_prob)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=progress_bar,
    )
    t0 = time.time()
    mcmc.run(jrandom.PRNGKey(rng_seed), *model_args, **{observed_kwarg: observed})
    wall = time.time() - t0

    samples = mcmc.get_samples(group_by_chain=True)

    import numpy as np

    # Diagnostics — match nuts_fit
    summary_stats = numpyro.diagnostics.summary(samples, group_by_chain=True)
    r_hat_max = float(max(stat["r_hat"] for stat in summary_stats.values()))
    ess_bulk_min = float(min(stat["n_eff"] for stat in summary_stats.values()))
    num_divergent = int(np.asarray(
        mcmc.get_extra_fields(group_by_chain=True).get("diverging", np.array(0))
    ).sum())

    from dataclasses import dataclass

    @dataclass
    class _Result:
        samples: dict
        diagnostics: dict

    diagnostics = {
        "r_hat_max": r_hat_max,
        "ess_bulk_min": ess_bulk_min,
        "num_divergent": num_divergent,
        "wall_seconds": wall,
    }
    return _Result(samples=samples, diagnostics=diagnostics)
'''
    SAMPLERS_PY.write_text(samp_text + generic_fn)
    print(f"✓ samplers.py: added nuts_fit_generic")
else:
    print("  samplers.py: nuts_fit_generic already present, skipping")


# ===========================================================================
# Summary
# ===========================================================================

print()
print("=" * 60)
print("Two-stage fit pipeline ready.")
print("=" * 60)
print()
print("Next steps:")
print("  1. Inspect the diff:")
print("       git diff src/ztbus/optim/data.py src/ztbus/optim/samplers.py")
print("       git status")
print()
print("  2. Smoke-test on July 2021 (subsample for fast iteration):")
print("       salloc --partition=batch --time=00:30:00 "
      "--cpus-per-task=4 --mem=8G")
print("       cd ~/ztbus-energy-model")
print("       uv run python dump/scripts/two_stage_fit.py "
      "--year-month 2021-07 --subsample 20000")
print()
print("  3. If both stages look identified, scale up:")
print("       uv run python dump/scripts/two_stage_fit.py "
      "--year-month 2021-07 --subsample -1")
print()
print("  4. If that works, run full year (probably needs GPU):")
print("       sbatch slurm/fit_one_bus_v0.sbatch  "
      "(after editing it to call two_stage_fit.py)")
