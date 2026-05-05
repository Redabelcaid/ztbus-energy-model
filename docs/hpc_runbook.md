# HPC runbook

This document covers how the project runs on a SLURM cluster. Adjust the
specifics (partition names, modules) for your site and commit those tweaks to
`configs/local/slurm.yaml` (gitignored) so they survive across users.

## One-time setup

1. **Install `uv` on a login node** (no root required):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
   ```
2. **Clone the repo into your home or work directory:**
   ```bash
   git clone <repo-url> ztbus-energy-model
   cd ztbus-energy-model
   ```
3. **Symlink data directories to scratch / project storage:**
   ```bash
   # Example — adjust to your cluster's paths
   ln -s "$SCRATCH/ztbus_raw"        data/raw
   ln -s "$SCRATCH/ztbus_interim"    data/interim
   ln -s "$SCRATCH/ztbus_processed"  data/processed
   ln -s "$SCRATCH/ztbus_reports"    data/reports
   ```
   Putting Parquet output on scratch keeps the repo tree small and writes fast.
4. **Install the environment:**
   ```bash
   make install
   ```

## Sanity checks

```bash
make check          # lint + type-check + tests
uv run ztbus version
uv run ztbus ingest --help
```

## Phase 1 — Ingest

The ingest converts ~1409 mission CSVs (≈10 GB total) into mission-partitioned
Parquet (≈1–1.5 GB total). On a single node with 8 workers this takes ~5
minutes; on the cluster as an array job it finishes in under 2.

```bash
sbatch slurm/ingest.sbatch
```

The job partitions files round-robin across array tasks. Tune
`#SBATCH --array=0-9` if you have many more or fewer files. Logs land under
`logs/ingest-{job_id}_{task_id}.{out,err}`.

## Phase 2+ — pipeline via Snakemake

Once Phases 2 and 5 are implemented, the same DAG runs locally or on SLURM:

```bash
# Local, 8 cores
make pipeline

# Cluster, up to 100 simultaneous SLURM jobs
make pipeline-hpc
```

Snakemake reads per-rule resources from `configs/hpc/slurm.yaml`. Add a
cluster-specific override in `configs/local/slurm.yaml` rather than editing
the committed defaults.

## Common operations

```bash
# Watch the queue
squeue --me

# Inspect a finished job
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ReqMem

# Re-run the pipeline incrementally (Snakemake skips up-to-date outputs)
make pipeline-hpc

# Force a phase to re-run
uv run snakemake --executor slurm --jobs 100 --forcerun ingest_summary
```

## Why scratch, why partitioned Parquet

The cluster's scratch filesystem (Lustre / GPFS) is much faster for many small
random-access reads than home directories. Mission-partitioned Parquet makes
each phase's I/O a stream of independent file reads rather than one giant
file, which plays well with parallel filesystems and lets array tasks read
disjoint shards without contention.

Compression is `zstd` level 3, chosen for a balance: ~10× smaller than the raw
CSV with negligible decompression CPU.

## Resource sizing

The defaults in `configs/hpc/slurm.yaml` and the `#SBATCH` headers in
`slurm/*.sbatch` are sized for the actual workload measured during
development. Most rules complete well inside the requested time/memory.
The exception is `fit_parameters`, which benefits from many cores and which
will be tuned in Phase 5 once the optimizer choice is settled.
