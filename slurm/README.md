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

- **GPU**: 1 × V100 SXM2 (16 or 32 GB). The model + corpus fits easily; we
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
