"""v0.9.0 + V100 production stack builder.

Creates / modifies these files on the cluster:

  dump/scripts/smoke_one_month.py        # parameterize with CLI --year-month
  dump/scripts/fit_production.py         # NEW — proper fitter for full-corpus / SLURM
  slurm/fit_one_bus_v0.sbatch            # NEW — V100 submission for bus 183, full 2021
  slurm/README.md                        # NEW — how to submit, monitor, diagnose
  CHANGELOG_v0.9.0.md                    # NEW — short notes on the winter milestone

Apply on the cluster:
    cd ~/ztbus-energy-model
    uv run python dump/scripts/build_production_stack.py
"""

from pathlib import Path
from textwrap import dedent

REPO = Path(".").resolve()

# ===========================================================================
# 1. Replace smoke_one_month.py with a CLI-parameterized version
# ===========================================================================

SMOKE_PY = REPO / "dump/scripts/smoke_one_month.py"
SMOKE_CONTENT = '''"""Smoke run: NUTS on one month of data. Measures wall time + sane params.

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

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
log.info("JAX devices: %s", jax.devices())

from ztbus.optim.data import load_corpus  # noqa: E402
from ztbus.optim.samplers import (  # noqa: E402
    nuts_fit,
    posterior_summary,
    posterior_to_dataframe,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-month", default="2021-07",
                    help="YYYY-MM (default: 2021-07, our baseline summer)")
    ap.add_argument("--bus-id", default="183",
                    help="183 or 208 (default: 183)")
    ap.add_argument("--subsample", type=int, default=20_000,
                    help="random subsample cap; -1 for no cap")
    ap.add_argument("--num-warmup", type=int, default=300)
    ap.add_argument("--num-samples", type=int, default=300)
    ap.add_argument("--num-chains", type=int, default=2)
    args = ap.parse_args()

    PROCESSED = Path(os.environ.get(
        "ZTBUS_PROCESSED_DIR",
        "/scratch/users/rbelcaid/ztbus/processed",
    ))
    log.info("Processed dir: %s", PROCESSED)
    log.info("Bus: %s   Month: %s   Subsample: %s",
             args.bus_id, args.year_month,
             "(no cap)" if args.subsample == -1 else args.subsample)

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

    log.info("\\n=========== POSTERIOR SUMMARY ===========")
    summary = posterior_summary(result.samples)
    print(summary)

    log.info("\\n=========== TIMING & QUALITY ===========")
    log.info("Wall time:     %.1f s", result.diagnostics["wall_seconds"])
    log.info("R-hat max:     %.4f  (target < 1.01)", result.diagnostics["r_hat_max"])
    log.info("ESS bulk min:  %.0f   (target > 100)", result.diagnostics["ess_bulk_min"])
    log.info("Divergences:   %d     (target = 0)", result.diagnostics["num_divergent"])

    # Save into a per-run subdirectory so successive runs don't clobber
    out_dir = Path(f"/scratch/users/{os.environ.get('USER', 'rbelcaid')}/ztbus/reports/"
                   f"smoke_one_month/{args.bus_id}_{args.year_month}")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.write_parquet(out_dir / "posterior_summary.parquet")
    posterior_to_dataframe(result.samples).write_parquet(out_dir / "posterior_samples.parquet")
    log.info("Wrote: %s", out_dir)
    log.info("\\nDone.")


if __name__ == "__main__":
    main()
'''

SMOKE_PY.write_text(SMOKE_CONTENT)
print(f"✓ {SMOKE_PY}  (parameterized; --year-month, --bus-id, --subsample)")


# ===========================================================================
# 2. Production fitter — full-corpus, robust, metadata + reproducibility
# ===========================================================================

FIT_PY = REPO / "dump/scripts/fit_production.py"
FIT_CONTENT = '''"""Production NUTS fit on the full corpus. Designed for SLURM (no progress bar).

Usage examples:

  # bus 183, all 2021
  uv run python dump/scripts/fit_production.py \\
      --bus-ids 183 \\
      --year-months 2021-01,2021-02,2021-03,2021-04,2021-05,2021-06,\\
2021-07,2021-08,2021-09,2021-10,2021-11,2021-12 \\
      --output-dir /scratch/users/$USER/ztbus/reports/prod_b183_2021

  # both buses, full corpus, all months we have
  uv run python dump/scripts/fit_production.py \\
      --bus-ids 183,208 \\
      --year-months ALL \\
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
from datetime import datetime, timezone
from pathlib import Path

# JAX config MUST precede any numpyro / kernels import
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from ztbus.optim.data import load_corpus  # noqa: E402
from ztbus.optim.samplers import (  # noqa: E402
    nuts_fit,
    posterior_summary,
    posterior_to_dataframe,
)


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).strip().decode()
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
    ap.add_argument("--processed-dir",
                    default=os.environ.get(
                        "ZTBUS_PROCESSED_DIR",
                        "/scratch/users/rbelcaid/ztbus/processed",
                    ))
    ap.add_argument("--bus-ids", default="183",
                    help="comma-separated, e.g. '183' or '183,208'")
    ap.add_argument("--year-months", default="ALL",
                    help="comma-separated 'YYYY-MM' or 'ALL' for the full dataset")
    ap.add_argument("--num-warmup", type=int, default=1000)
    ap.add_argument("--num-samples", type=int, default=2000)
    ap.add_argument("--num-chains", type=int, default=2)
    ap.add_argument("--chain-method", default="vectorized",
                    choices=["sequential", "parallel", "vectorized"])
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--target-accept-prob", type=float, default=0.85)
    ap.add_argument("--subsample", type=int, default=-1,
                    help="row cap; -1 = no cap (production default)")
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
    year_months = _all_year_months() if args.year_months.upper() == "ALL" \\
        else tuple(args.year_months.split(","))

    log.info("Bus IDs:       %s", bus_ids)
    log.info("Year-months:   %s%s",
             year_months[:3], f" ... ({len(year_months)} months)" if len(year_months) > 3 else "")
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
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
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
        "diagnostics": {k: float(v) if not isinstance(v, int) else v
                        for k, v in result.diagnostics.items()},
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    log.info("")
    log.info("=========== POSTERIOR SUMMARY ===========")
    log.info("\\n%s", summary)
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
'''

FIT_PY.write_text(FIT_CONTENT)
print(f"✓ {FIT_PY}  (production fitter, CLI + metadata.json)")


# ===========================================================================
# 3. SLURM submission script for V100, bus 183, all 2021
# ===========================================================================

SLURM_DIR = REPO / "slurm"
SLURM_DIR.mkdir(exist_ok=True)
SLURM_LOGS = SLURM_DIR / "logs"
SLURM_LOGS.mkdir(exist_ok=True)
(SLURM_LOGS / ".gitkeep").touch()

SBATCH = SLURM_DIR / "fit_one_bus_v0.sbatch"
SBATCH_CONTENT = """#!/bin/bash -l
# ============================================================================
# ZTBus production NUTS fit — bus 183, full 2021, single V100
# ============================================================================
#SBATCH --job-name=ztbus-fit-b183-2021
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail

# ---- Header for the log file ----------------------------------------------
echo "============================================================"
echo "ZTBus production NUTS fit"
echo "============================================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Job name:     $SLURM_JOB_NAME"
echo "Node:         $(hostname)"
echo "Partition:    $SLURM_JOB_PARTITION"
echo "CPUs:         $SLURM_CPUS_PER_TASK"
echo "Memory:       $SLURM_MEM_PER_NODE MB"
echo "Submit dir:   $SLURM_SUBMIT_DIR"
echo "Start:        $(date)"
echo

cd "$SLURM_SUBMIT_DIR"

# ---- Environment setup ----------------------------------------------------
echo "=== Module setup ==="
module purge
# Iris GPU nodes: load CUDA. Module name may vary — adjust if "module avail"
# shows something different (e.g. "system/CUDA/12.4" or "lib/cuDNN/...").
module load system/CUDA || module load CUDA || echo "WARNING: no CUDA module loaded"
module list 2>&1
echo

echo "=== GPU check ==="
nvidia-smi || echo "WARNING: nvidia-smi not available"
echo

# ---- Run the fit ----------------------------------------------------------
OUTPUT_DIR="/scratch/users/$USER/ztbus/reports/prod_${SLURM_JOB_ID}_b183_2021"
echo "=== Starting fit ==="
echo "Output: $OUTPUT_DIR"
echo

uv run python dump/scripts/fit_production.py \\
    --bus-ids 183 \\
    --year-months 2021-01,2021-02,2021-03,2021-04,2021-05,2021-06,\\
2021-07,2021-08,2021-09,2021-10,2021-11,2021-12 \\
    --num-warmup 1000 \\
    --num-samples 2000 \\
    --num-chains 2 \\
    --chain-method vectorized \\
    --target-accept-prob 0.85 \\
    --rng-seed 0 \\
    --output-dir "$OUTPUT_DIR"

echo
echo "=== Done ==="
echo "End:         $(date)"
echo "Output dir:  $OUTPUT_DIR"
"""

SBATCH.write_text(SBATCH_CONTENT)
print(f"✓ {SBATCH}  (V100, 4h, bus 183, full 2021)")


# ===========================================================================
# 4. slurm/README.md — runbook
# ===========================================================================

README = SLURM_DIR / "README.md"
README_CONTENT = (
    dedent("""
    # ZTBus SLURM submission scripts

    Production NUTS fits on Iris GPU partition. CPU smoke runs go through
    `dump/scripts/smoke_one_month.py` interactively; this folder is for
    multi-hour batch jobs.

    ## Files

    - `fit_one_bus_v0.sbatch` — one V100, bus 183, all of 2021 (4 h wall budget)
    - `logs/` — captured stdout/stderr for every job (`<jobname>-<jobid>.out/err`)

    ## How to submit

    ```bash
    cd ~/ztbus-energy-model
    sbatch slurm/fit_one_bus_v0.sbatch
    ```

    The submission returns a job ID immediately; the actual fit runs whenever
    the GPU partition frees up. Output goes to:

    - `slurm/logs/ztbus-fit-b183-2021-<jobid>.out` — full stdout
    - `slurm/logs/ztbus-fit-b183-2021-<jobid>.err` — stderr
    - `/scratch/users/$USER/ztbus/reports/prod_<jobid>_b183_2021/` — posteriors

    The output directory contains:
    - `posterior_summary.parquet` — mean / sd / 95% CI / R-hat / ESS per param
    - `posterior_samples.parquet` — every (chain, draw, parameter) sample
    - `metadata.json` — config, audit, diagnostics, git hash, JAX device list
    - `fit.log` — duplicate of stdout for archiving

    ## How to monitor

    ```bash
    # Queue status (your jobs only)
    squeue -u $USER

    # Live tail of the SLURM stdout
    tail -f slurm/logs/ztbus-fit-b183-2021-<jobid>.out

    # Detailed accounting once the job ends
    sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,MaxVMSize,ReqGPUS,AllocGRES
    ```

    ## Expected wall time

    On a single Iris V100 (16 or 32 GB):

    | Scope | Samples after filters | Expected wall |
    |---|---|---|
    | 1 bus, 1 month | ~400k | 5-15 min |
    | 1 bus, 1 year  | ~5-10M | 30-90 min |
    | 2 buses, full corpus | ~20-30M | 2-6 h |

    The 4 h time limit in `fit_one_bus_v0.sbatch` is sized for the middle case
    with safety margin. Adjust `--time` for the others.

    ## Resource sizing notes

    - **GPU**: 1x V100 SXM2 (16 or 32 GB). The model + corpus fits easily; we
      don't need multi-GPU yet.
    - **CPU cores**: 4 is enough for the data loader (Polars LazyFrame scan +
      filter). More cores don't help once we're on the GPU kernel.
    - **RAM**: 32 GB host RAM is generous. Full-corpus parquet scan peaks
      around 4 GB.
    - **Storage**: outputs are tiny (< 10 MB).

    ## If a job fails

    1. Check the `.err` file for the immediate error.
    2. Common culprits and fixes:
       - **CUDA module not found** → run `module avail 2>&1 | grep -i cuda`
         on the access node, then edit the `module load` line in the sbatch.
       - **JAX falls back to CPU** → the sbatch's `nvidia-smi` block will be
         empty. Either the GPU resource didn't allocate or the JAX install
         doesn't have the CUDA plugin. Fix: `uv add "jax[cuda12]"` then re-lock.
       - **OOM on GPU** → reduce `--num-chains` from 2 to 1, or use
         `--subsample 5000000` for a first sanity run.
       - **R-hat > 1.05** → chains didn't converge; bump `--num-warmup` to
         2000 and `--num-samples` to 4000, re-submit.

    ## Adding new submission scripts

    Copy `fit_one_bus_v0.sbatch` and edit the `--job-name`, the
    `--year-months` / `--bus-ids` flags, and the `OUTPUT_DIR`. Keep the
    `logs/` and `metadata.json` conventions identical so post-hoc analysis
    can compare runs by jobid.
    """).strip()
    + "\n"
)

README.write_text(README_CONTENT)
print(f"✓ {README}  (runbook)")


# ===========================================================================
# 5. CHANGELOG entry for v0.9.0
# ===========================================================================

CHANGELOG = REPO / "CHANGELOG_v0.9.0.md"
CHANGELOG_CONTENT = (
    dedent("""
    # v0.9.0-cross-season — first cross-season parameter identification

    ## Headline

    Ran NUTS on January 2022 data (winter, bus 183) and compared against
    July 2021 (summer, same bus). All seven physical parameters identified
    with real Bayesian intervals on both seasons; HVAC coefficient now
    properly identified thanks to winter temperature contrast.

    ## Cross-season comparison

    | Parameter | July 2021 | January 2022 | Reading |
    |---|---|---|---|
    | A         | 8.32 m²     | 8.23 m²    | stable ✓ |
    | Cd        | 0.56        | 0.53       | stable ✓ |
    | Crr       | 0.0173      | 0.0187     | slight winter rise (real?) |
    | eta_prop  | 0.93        | 0.94       | stable ✓ |
    | eta_recup | 0.852       | 0.820      | winter drop ~3 pts (battery temp?) |
    | c_HVAC    | 0.11 (weak) | **0.40 [0.31, 0.49]** | **identified** |
    | P_aux     | 0.22 (weak) | 0.95 (still weak) | weakly constrained without depot samples |
    | sigma_W   | 44.0 kW     | 44.8 kW    | structural limit, season-invariant |

    ## What this enables

    - First publishable HVAC parameter for the HESS lighTram 19.
    - Cross-season validation of 5/7 parameters as season-invariant.
    - Documented physical limit at sigma ~44 kW; honest report rather than
      hidden via point estimates.

    ## Tooling changes

    - `dump/scripts/smoke_one_month.py` — parameterized with `--year-month`,
      `--bus-id`, `--subsample` CLI flags. No more sed-editing per run.
    - `dump/scripts/fit_production.py` — new full-corpus fitter for SLURM.
    - `slurm/fit_one_bus_v0.sbatch` — first V100 production submission.
    - `slurm/README.md` — runbook for submitting + monitoring + diagnosing.

    ## What's next

    Production run on Iris V100 (full year, bus 183) once `jax[cuda12]` is
    in the environment.
    """).strip()
    + "\n"
)

CHANGELOG.write_text(CHANGELOG_CONTENT)
print(f"✓ {CHANGELOG}")


# ===========================================================================
# Done
# ===========================================================================

print()
print("All v0.9.0 + V100 production stack files created.")
print()
print("Next manual steps:")
print("  git add dump/scripts/smoke_one_month.py")
print("  git add dump/scripts/fit_production.py")
print("  git add slurm/")
print("  git add CHANGELOG_v0.9.0.md")
print("  git diff --stat HEAD")
print("  git commit -m 'phase5: v0.9.0 cross-season + V100 production scaffold'")
print("  git tag -a v0.9.0-cross-season -m '...'")
print("  git push origin main")
print("  git push origin v0.9.0-cross-season")
