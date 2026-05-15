"""Smoke run: NUTS on one month of data. Measures wall time + sane params.

Usage:
    # July 2021 (default, our baseline)
    uv run python dump/scripts/smoke_one_month.py

    # Any other month
    uv run python dump/scripts/smoke_one_month.py --year-month 2022-01
    uv run python dump/scripts/smoke_one_month.py --year-month 2022-01 --bus-id 208
"""

import argparse
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
from ztbus.optim.samplers import (
    nuts_fit,
    posterior_summary,
    posterior_to_dataframe,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--year-month", default="2021-07", help="YYYY-MM (default: 2021-07, our baseline summer)"
    )
    ap.add_argument("--bus-id", default="183", help="183 or 208 (default: 183)")
    ap.add_argument(
        "--subsample", type=int, default=20_000, help="random subsample cap; -1 for no cap"
    )
    ap.add_argument("--num-warmup", type=int, default=300)
    ap.add_argument("--num-samples", type=int, default=300)
    ap.add_argument("--num-chains", type=int, default=2)
    args = ap.parse_args()

    PROCESSED = Path(
        os.environ.get(
            "ZTBUS_PROCESSED_DIR",
            "/scratch/users/rbelcaid/ztbus/processed",
        )
    )
    log.info("Processed dir: %s", PROCESSED)
    log.info(
        "Bus: %s   Month: %s   Subsample: %s",
        args.bus_id,
        args.year_month,
        "(no cap)" if args.subsample == -1 else args.subsample,
    )

    t0 = time.time()
    arrays, _audit = load_corpus(
        PROCESSED,
        bus_ids=(args.bus_id,),
        year_months=(args.year_month,),
        subsample=None if args.subsample == -1 else args.subsample,
        subsample_seed=0,
    )
    log.info("Load + filter wall time: %.1f s", time.time() - t0)
    log.info("Final sample count: %d", arrays["speed_mps"].shape[0])

    result = nuts_fit(
        data=arrays,
        observed_power_W=arrays["P_obs_W"],
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        chain_method="sequential",
        progress_bar=True,
        rng_seed=0,
    )

    log.info("\n=========== POSTERIOR SUMMARY ===========")
    summary = posterior_summary(result.samples)
    print(summary)

    log.info("\n=========== TIMING & QUALITY ===========")
    log.info("Wall time:     %.1f s", result.diagnostics["wall_seconds"])
    log.info("R-hat max:     %.4f  (target < 1.01)", result.diagnostics["r_hat_max"])
    log.info("ESS bulk min:  %.0f   (target > 100)", result.diagnostics["ess_bulk_min"])
    log.info("Divergences:   %d     (target = 0)", result.diagnostics["num_divergent"])

    # Save into a per-run subdirectory so successive runs don't clobber
    out_dir = Path(
        f"/scratch/users/{os.environ.get('USER', 'rbelcaid')}/ztbus/reports/"
        f"smoke_one_month/{args.bus_id}_{args.year_month}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.write_parquet(out_dir / "posterior_summary.parquet")
    posterior_to_dataframe(result.samples).write_parquet(out_dir / "posterior_samples.parquet")
    log.info("Wrote: %s", out_dir)
    log.info("\nDone.")


if __name__ == "__main__":
    main()
