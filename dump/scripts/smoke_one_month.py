"""First real-data NUTS run. Measures wall time + produces first posterior.

Scope: bus 183, July 2021 only, subsampled to 20k samples. Small enough to
finish in ~5-10 minutes on CPU, large enough to be a real result.
"""

import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("smoke")

import jax

jax.config.update("jax_enable_x64", True)
log.info("JAX devices: %s", jax.devices())

from ztbus.optim.data import load_corpus
from ztbus.optim.samplers import nuts_fit, posterior_summary

PROCESSED = Path(os.environ.get("ZTBUS_PROCESSED_DIR", "/scratch/users/rbelcaid/ztbus/processed"))
log.info("Processed dir: %s", PROCESSED)

# ---- Load ------------------------------------------------------------------
t0 = time.time()
arrays, audit = load_corpus(
    PROCESSED,
    bus_ids=("183",),
    year_months=("2021-07",),
    subsample=20_000,
    subsample_seed=0,
)
log.info("Load + filter wall time: %.1f s", time.time() - t0)
log.info("Final sample count: %d", arrays["speed_mps"].shape[0])

# ---- Run NUTS --------------------------------------------------------------
result = nuts_fit(
    data=arrays,
    observed_power_W=arrays["P_obs_W"],
    num_warmup=300,
    num_samples=300,
    num_chains=2,
    chain_method="sequential",
    progress_bar=True,
    rng_seed=0,
)

# ---- Report ---------------------------------------------------------------
log.info("\n=========== POSTERIOR SUMMARY ===========")
summary = posterior_summary(result.samples)
print(summary)

log.info("\n=========== TIMING & QUALITY ===========")
log.info("Wall time:     %.1f s", result.diagnostics["wall_seconds"])
log.info("R-hat max:     %.4f  (target < 1.01)", result.diagnostics["r_hat_max"])
log.info("ESS bulk min:  %.0f   (target > 100)", result.diagnostics["ess_bulk_min"])
log.info("Divergences:   %d     (target = 0)", result.diagnostics["num_divergent"])

# ---- Save for downstream analysis -----------------------------------------
out_dir = Path("/scratch/users/rbelcaid/ztbus/reports/smoke_one_month")
out_dir.mkdir(parents=True, exist_ok=True)
summary.write_parquet(out_dir / "posterior_summary.parquet")

# Long-form samples for posterior pair plots later
from ztbus.optim.samplers import posterior_to_dataframe

long_form = posterior_to_dataframe(result.samples)
long_form.write_parquet(out_dir / "posterior_samples.parquet")

log.info("Wrote: %s", out_dir / "posterior_summary.parquet")
log.info("Wrote: %s", out_dir / "posterior_samples.parquet")
log.info("\nDone.")
