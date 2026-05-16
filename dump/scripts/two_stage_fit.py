"""Two-stage fit driver: Stage 1 (mechanical) then Stage 2 (electrical).

Usage:
    uv run python dump/scripts/two_stage_fit.py
    uv run python dump/scripts/two_stage_fit.py --year-month 2021-07 --subsample -1
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

from ztbus.optim import model_traction, model_electrical  # noqa: E402
from ztbus.optim.data import load_corpus  # noqa: E402
from ztbus.optim.samplers import (  # noqa: E402
    nuts_fit_generic, posterior_summary, posterior_to_dataframe,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-month", default="2021-07")
    ap.add_argument("--bus-id", default="183")
    ap.add_argument("--subsample", type=int, default=20_000,
                    help="row cap; -1 for no cap")
    ap.add_argument("--num-warmup", type=int, default=300)
    ap.add_argument("--num-samples", type=int, default=300)
    ap.add_argument("--num-chains", type=int, default=2)
    args = ap.parse_args()

    PROCESSED = Path(os.environ.get(
        "ZTBUS_PROCESSED_DIR", "/scratch/users/rbelcaid/ztbus/processed",
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

    # ---- Stage 1 ------------------------------------------------------
    log.info("=" * 70)
    log.info("STAGE 1: mechanical fit on traction_tractionForce")
    log.info("=" * 70)
    r1 = nuts_fit_generic(
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
    s1 = posterior_summary(r1.samples)
    log.info("\n=== Stage 1 posterior ===\n%s", s1)
    log.info("R-hat %.4f  ESS %.0f  div %d  wall %.1fs",
             r1.diagnostics["r_hat_max"], r1.diagnostics["ess_bulk_min"],
             r1.diagnostics["num_divergent"], r1.diagnostics["wall_seconds"])
    s1.write_parquet(out_dir / "stage1_posterior_summary.parquet")
    posterior_to_dataframe(r1.samples).write_parquet(
        out_dir / "stage1_posterior_samples.parquet")

    # ---- Stage 2 ------------------------------------------------------
    log.info("=" * 70)
    log.info("STAGE 2: electrical fit on electric_powerDemand")
    log.info("=" * 70)
    r2 = nuts_fit_generic(
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
    s2 = posterior_summary(r2.samples)
    log.info("\n=== Stage 2 posterior ===\n%s", s2)
    log.info("R-hat %.4f  ESS %.0f  div %d  wall %.1fs",
             r2.diagnostics["r_hat_max"], r2.diagnostics["ess_bulk_min"],
             r2.diagnostics["num_divergent"], r2.diagnostics["wall_seconds"])
    s2.write_parquet(out_dir / "stage2_posterior_summary.parquet")
    posterior_to_dataframe(r2.samples).write_parquet(
        out_dir / "stage2_posterior_samples.parquet")

    log.info("Wrote: %s", out_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()
