#!/bin/bash -l
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

uv run python dump/scripts/fit_production.py \
    --bus-ids 183 \
    --year-months 2021-01,2021-02,2021-03,2021-04,2021-05,2021-06,2021-07,2021-08,2021-09,2021-10,2021-11,2021-12 \
    --num-warmup 1000 \
    --num-samples 2000 \
    --num-chains 2 \
    --chain-method vectorized \
    --target-accept-prob 0.85 \
    --rng-seed 0 \
    --output-dir "$OUTPUT_DIR"

echo
echo "=== Done ==="
echo "End:         $(date)"
echo "Output dir:  $OUTPUT_DIR"
