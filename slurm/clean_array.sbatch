#!/bin/bash
#
# Per-mission cleaning array job. Each array task cleans one mission parquet.
#
# Submit with:
#   sbatch --array=0-1408 slurm/clean_array.sbatch
# Or (after deciding chunk size):
#   sbatch slurm/clean_array.sbatch
#
#SBATCH --job-name=ztbus-clean
#SBATCH --output=logs/clean-%A_%a.out
#SBATCH --error=logs/clean-%A_%a.err
#SBATCH --array=0-99               # adjust to mission count / chunk size
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --partition=compute        # CHANGE ME per cluster

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
mkdir -p logs

INTERIM_DIR="${ZTBUS_INTERIM_DIR:-$PROJECT_ROOT/data/interim}"
PROCESSED_DIR="${ZTBUS_PROCESSED_DIR:-$PROJECT_ROOT/data/processed}"

# Phase 2 will populate the actual cleaning entry point.
echo "Cleaning task ${SLURM_ARRAY_TASK_ID} — placeholder; phase 2 wires this up."
uv run ztbus clean --interim-dir "$INTERIM_DIR" --processed-dir "$PROCESSED_DIR"
