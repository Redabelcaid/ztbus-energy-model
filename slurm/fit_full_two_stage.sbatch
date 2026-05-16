#!/bin/bash -l
# ============================================================================
# Full-corpus two-stage fit: both buses, May 2019 - Dec 2022, one V100 each
# ============================================================================
#SBATCH --job-name=ztbus-two-stage
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

set -euo pipefail

# ---- Map array index to bus ID --------------------------------------------
BUS_IDS=(183 208)
BUS_ID="${BUS_IDS[$SLURM_ARRAY_TASK_ID]}"

echo "============================================================"
echo "ZTBus two-stage full-corpus fit"
echo "============================================================"
echo "Array job:    $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID"
echo "Bus ID:       $BUS_ID"
echo "Node:         $(hostname)"
echo "CPUs:         $SLURM_CPUS_PER_TASK"
echo "Memory:       $SLURM_MEM_PER_NODE MB"
echo "Submit dir:   $SLURM_SUBMIT_DIR"
echo "Start:        $(date)"
echo

cd "$SLURM_SUBMIT_DIR"

# ---- Environment ---------------------------------------------------------
echo "=== Module setup ==="
module purge
module load system/CUDA/12.6.0 || module load CUDA || true
module list 2>&1
echo

# Force GPU and ensure bundled CUDA libs are findable (same trick that
# worked for the joint-fit production run).
export JAX_PLATFORMS=cuda
VENV_SITE=$(uv run python -c 'import site; print(site.getsitepackages()[0])')
NVIDIA_LIBS=$(find "$VENV_SITE/nvidia" -name "lib" -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:-}"
echo "JAX_PLATFORMS=$JAX_PLATFORMS"

echo
echo "=== GPU check ==="
nvidia-smi || echo "WARNING: nvidia-smi unavailable"
echo

# ---- Run ------------------------------------------------------------------
OUTPUT_DIR="/scratch/users/$USER/ztbus/reports/two_stage_full/${SLURM_ARRAY_JOB_ID}_b${BUS_ID}"
mkdir -p "$OUTPUT_DIR"

echo "=== Starting two-stage fit for bus $BUS_ID ==="
echo "Output dir: $OUTPUT_DIR"
echo

uv run python dump/scripts/two_stage_fit.py \
    --bus-id "$BUS_ID" \
    --year-month ALL \
    --subsample -1 \
    --num-warmup 1000 \
    --num-samples 2000 \
    --num-chains 2 \
    2>&1 | tee "$OUTPUT_DIR/run.log"

# Copy the per-bus posterior outputs into our SLURM-aware output dir for
# easier provenance (the driver writes by default to
# /scratch/.../reports/two_stage/<bus>_ALL/).
DRIVER_OUT="/scratch/users/$USER/ztbus/reports/two_stage/${BUS_ID}_ALL"
if [ -d "$DRIVER_OUT" ]; then
    cp -r "$DRIVER_OUT"/* "$OUTPUT_DIR/" || true
fi

echo
echo "=== Done ==="
echo "End:         $(date)"
echo "Output dir:  $OUTPUT_DIR"
