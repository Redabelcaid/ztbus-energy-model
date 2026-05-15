"""Production NUTS fit on the full corpus. Designed for SLURM (no progress bar).

Usage examples:

  # bus 183, all 2021
  uv run python dump/scripts/fit_production.py \
      --bus-ids 183 \
      --year-months 2021-01,2021-02,2021-03,2021-04,2021-05,2021-06,\
2021-07,2021-08,2021-09,2021-10,2021-11,2021-12 \
      --output-dir /scratch/users/$USER/ztbus/reports/prod_b183_2021

  # both buses, full corpus, all months we have
  uv run python dump/scripts/fit_production.py \
      --bus-ids 183,208 \
      --year-months ALL \
      --output-dir /scratch/users/$USER/ztbus/reports/prod_full

Outputs to OUTPUT_DIR:
  posterior_summary.parquet    — mean / sd / CI / R-hat / ESS per parameter
  posterior_samples.parquet    — every (chain, draw, parameter) sample, long-form
  metadata.json                — config, audit, diagnostics, git hash, JAX device
  fit.log                      — full stdout (mirrors what runs on the terminal)
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# JAX config MUST precede any numpyro / kernels import
import jax

jax.config.update("jax_enable_x64", True)

from ztbus.optim.data import load_corpus
from ztbus.optim.samplers import (
    nuts_fit,
    posterior_summary,
    posterior_to_dataframe,
)


def _git_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .decode()
        )
    except Exception:
        return "unknown"


def _all_year_months() -> tuple[str, ...]:
    """Default --year-months ALL: every YYYY-MM the dataset covers (May 2019 to Dec 2022)."""
    ym = []
    for year in (2019, 2020, 2021, 2022):
        for month in range(1, 13):
            if year == 2019 and month < 5:
                continue
            ym.append(f"{year}-{month:02d}")
    return tuple(ym)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--processed-dir",
        default=os.environ.get(
            "ZTBUS_PROCESSED_DIR",
            "/scratch/users/rbelcaid/ztbus/processed",
        ),
    )
    ap.add_argument("--bus-ids", default="183", help="comma-separated, e.g. '183' or '183,208'")
    ap.add_argument(
        "--year-months",
        default="ALL",
        help="comma-separated 'YYYY-MM' or 'ALL' for the full dataset",
    )
    ap.add_argument("--num-warmup", type=int, default=1000)
    ap.add_argument("--num-samples", type=int, default=2000)
    ap.add_argument("--num-chains", type=int, default=2)
    ap.add_argument(
        "--chain-method", default="vectorized", choices=["sequential", "parallel", "vectorized"]
    )
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--target-accept-prob", type=float, default=0.85)
    ap.add_argument(
        "--subsample", type=int, default=-1, help="row cap; -1 = no cap (production default)"
    )
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Log to both stdout (for SLURM .out file) and a per-run logfile
    log_path = out_dir / "fit.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("fit")

    log.info("================================================================")
    log.info("ZTBus production NUTS fit")
    log.info("================================================================")
    log.info("JAX devices:   %s", jax.devices())
    log.info("JAX platform:  %s", jax.default_backend())
    log.info("Git HEAD:      %s", _git_hash())
    log.info("Output dir:    %s", out_dir)

    bus_ids = tuple(args.bus_ids.split(","))
    year_months = (
        _all_year_months()
        if args.year_months.upper() == "ALL"
        else tuple(args.year_months.split(","))
    )

    log.info("Bus IDs:       %s", bus_ids)
    log.info(
        "Year-months:   %s%s",
        year_months[:3],
        f" ... ({len(year_months)} months)" if len(year_months) > 3 else "",
    )
    log.info("Subsample:     %s", "no cap" if args.subsample == -1 else f"{args.subsample:,}")
    log.info("Chains:        %d via %s", args.num_chains, args.chain_method)
    log.info("Warmup:        %d", args.num_warmup)
    log.info("Samples:       %d", args.num_samples)
    log.info("Target accept: %.2f", args.target_accept_prob)
    log.info("RNG seed:      %d", args.rng_seed)
    log.info("")

    # ---- Load -----------------------------------------------------------
    t_load = time.time()
    arrays, audit = load_corpus(
        args.processed_dir,
        bus_ids=bus_ids,
        year_months=year_months,
        subsample=None if args.subsample == -1 else args.subsample,
        subsample_seed=args.rng_seed,
    )
    log.info("Data load wall time:  %.1f s", time.time() - t_load)
    log.info("Final sample count:   %s", f"{arrays['speed_mps'].shape[0]:,}")

    # ---- Fit ------------------------------------------------------------
    result = nuts_fit(
        data=arrays,
        observed_power_W=arrays["P_obs_W"],
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        chain_method=args.chain_method,
        progress_bar=False,
        rng_seed=args.rng_seed,
        target_accept_prob=args.target_accept_prob,
    )

    # ---- Persist --------------------------------------------------------
    summary = posterior_summary(result.samples)
    samples_long = posterior_to_dataframe(result.samples)
    summary.write_parquet(out_dir / "posterior_summary.parquet")
    samples_long.write_parquet(out_dir / "posterior_samples.parquet")

    metadata = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_hash": _git_hash(),
        "jax_devices": [str(d) for d in jax.devices()],
        "jax_platform": jax.default_backend(),
        "config": {
            "bus_ids": list(bus_ids),
            "year_months": list(year_months),
            "num_warmup": args.num_warmup,
            "num_samples": args.num_samples,
            "num_chains": args.num_chains,
            "chain_method": args.chain_method,
            "rng_seed": args.rng_seed,
            "target_accept_prob": args.target_accept_prob,
            "subsample": args.subsample,
        },
        "audit": audit.to_dict(),
        "diagnostics": {
            k: float(v) if not isinstance(v, int) else v for k, v in result.diagnostics.items()
        },
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    log.info("")
    log.info("=========== POSTERIOR SUMMARY ===========")
    log.info("\n%s", summary)
    log.info("")
    log.info("=========== TIMING & QUALITY ===========")
    log.info("Wall time:     %.1f s", result.diagnostics["wall_seconds"])
    log.info("R-hat max:     %.4f  (target < 1.01)", result.diagnostics["r_hat_max"])
    log.info("ESS bulk min:  %.0f   (target > 100)", result.diagnostics["ess_bulk_min"])
    log.info("Divergences:   %d     (target = 0)", result.diagnostics["num_divergent"])
    log.info("")
    log.info("Wrote: %s", out_dir)


if __name__ == "__main__":
    main()
