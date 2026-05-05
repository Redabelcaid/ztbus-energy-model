#!/bin/bash
#
# Ingest the raw ZTBus CSVs into mission-partitioned Parquet on the cluster.
#
# Each array task processes a contiguous chunk of mission files in parallel
# inside the task. Tune --array=0-N and CHUNK_SIZE so that
# (N+1) * CHUNK_SIZE  ≈ <number of mission CSVs>.
#
# Submit with:
#   sbatch slurm/ingest.sbatch
#
# Inspect with:
#   squeue --me
#   sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
#
#SBATCH --job-name=ztbus-ingest
#SBATCH --output=logs/ingest-%A_%a.out
#SBATCH --error=logs/ingest-%A_%a.err
#SBATCH --array=0-9                # 10 array tasks; adjust to taste
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4          # parallel workers per task
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --partition=compute        # CHANGE ME per cluster
# #SBATCH --account=YOUR_ACCOUNT     # uncomment if your cluster requires it

set -euo pipefail

# ---------------------------------------------------------------------------
# Cluster-specific bootstrap (uncomment what applies)
# ---------------------------------------------------------------------------
# module purge
# module load python/3.11

# ---------------------------------------------------------------------------
# Project-relative paths (resolve via the submitting directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
mkdir -p logs

# uv must be available; install once on a login node:
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#   export PATH="$HOME/.local/bin:$PATH"

# Use scratch for raw data if available; symlink data/raw → $SCRATCH/ztbus_raw
RAW_DIR="${ZTBUS_RAW_DIR:-$PROJECT_ROOT/data/raw}"
INTERIM_DIR="${ZTBUS_INTERIM_DIR:-$PROJECT_ROOT/data/interim}"

# ---------------------------------------------------------------------------
# Partition the work across array tasks deterministically
# ---------------------------------------------------------------------------
mapfile -t ALL_FILES < <(find "$RAW_DIR" -maxdepth 1 -name 'B*.csv' | sort)
N_FILES=${#ALL_FILES[@]}
N_TASKS=$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))
TASK_IDX=$((SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN))

# Round-robin assignment to balance unequal file sizes
TASK_FILES=()
for ((i = TASK_IDX; i < N_FILES; i += N_TASKS)); do
    TASK_FILES+=("${ALL_FILES[$i]}")
done

echo "Task $TASK_IDX/$N_TASKS handles ${#TASK_FILES[@]} of $N_FILES files"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
TASK_RAW_DIR=$(mktemp -d -p "${TMPDIR:-/tmp}" ztbus_task_XXXXXX)
trap 'rm -rf "$TASK_RAW_DIR"' EXIT

# Symlink (cheap, no copy) just this task's files
for f in "${TASK_FILES[@]}"; do
    ln -s "$f" "$TASK_RAW_DIR/$(basename "$f")"
done

uv run ztbus ingest \
    --raw-dir "$TASK_RAW_DIR" \
    --interim-dir "$INTERIM_DIR" \
    --max-workers "$SLURM_CPUS_PER_TASK" \
    --log-level INFO

echo "Task $TASK_IDX done."
