"""Grid-aware regen split + full-corpus driver + SLURM array submission.

This script does three things atomically:

1. Patches src/ztbus/optim/data.py to expose status_gridIsAvailable in the
   arrays dict under the key 'grid_available' (cast to float 0.0/1.0).

2. Patches src/ztbus/optim/model_electrical.py to split eta_recup into:
     - eta_recup_grid     (regen flows back into the catenary)
     - eta_recup_battery  (regen charges onboard battery)
   Selected at every sample via the grid_available mask. Priors widened to
   reflect motor-only (grid) vs storage-included (battery) expectations.

3. Extends dump/scripts/two_stage_fit.py to accept --year-month ALL, which
   sweeps every YYYY-MM in the documented dataset range (May 2019 - Dec 2022).

4. Creates slurm/fit_full_two_stage.sbatch: a 2-element SLURM array job
   (one task per bus) that runs the full corpus on a V100 each.

Apply on cluster:
    uv run python dump/scripts/build_grid_aware_full_run.py

Then submit:
    sbatch slurm/fit_full_two_stage.sbatch
"""

from pathlib import Path

REPO = Path(".").resolve()


def replace_or_die(p: Path, old: str, new: str, label: str) -> None:
    text = p.read_text()
    if new in text and old not in text:
        print(f"  --  {p.relative_to(REPO)}: {label} (already applied)")
        return
    assert old in text, f"anchor missing in {p}: {label!r}"
    p.write_text(text.replace(old, new, 1))
    print(f"  OK  {p.relative_to(REPO)}: {label}")


# =============================================================================
# 1. data.py — expose status_gridIsAvailable -> arrays['grid_available']
# =============================================================================
print("\n[1] Patching data.py to expose status_gridIsAvailable")
replace_or_die(
    REPO / "src/ztbus/optim/data.py",
    '"traction_tractionForce": "F_traction_N",',
    '"traction_tractionForce": "F_traction_N",\n    "status_gridIsAvailable": "grid_available",',
    "added status_gridIsAvailable -> grid_available mapping",
)

# =============================================================================
# 2. model_electrical.py — split eta_recup into grid / battery
# =============================================================================
print("\n[2] Patching model_electrical.py for grid-aware regen split")

ME = REPO / "src/ztbus/optim/model_electrical.py"
me_text = ME.read_text()

# 2a. Replace the eta_recup constants block with two separate ones
old_eta_recup_consts = """_ETA_RECUP_LO: float = 0.30
_ETA_RECUP_HI: float = 0.95"""
new_eta_recup_consts = """# eta_recup split: grid-feedback regen vs battery-acceptance regen
# Grid regen: motor + inverter only, no battery losses.  Plausible ~0.85-0.95.
# Battery regen: motor + inverter + battery acceptance.  Plausible ~0.65-0.85.
_ETA_RECUP_GRID_LO: float = 0.70
_ETA_RECUP_GRID_HI: float = 0.97
_ETA_RECUP_BATTERY_LO: float = 0.40
_ETA_RECUP_BATTERY_HI: float = 0.92"""

if "_ETA_RECUP_GRID_LO" not in me_text:
    assert old_eta_recup_consts in me_text, "eta_recup constants anchor missing"
    me_text = me_text.replace(old_eta_recup_consts, new_eta_recup_consts, 1)
    print("  OK  split eta_recup constants into grid / battery")
else:
    print("  --  eta_recup constants already split")

# 2b. Replace the single eta_recup sample with two samples + jnp.where selector
old_sample_block = '''    eta_recup = numpyro.sample("eta_recup", dist.Uniform(_ETA_RECUP_LO, _ETA_RECUP_HI))'''
new_sample_block = '''    eta_recup_grid = numpyro.sample(
        "eta_recup_grid",
        dist.Uniform(_ETA_RECUP_GRID_LO, _ETA_RECUP_GRID_HI),
    )
    eta_recup_battery = numpyro.sample(
        "eta_recup_battery",
        dist.Uniform(_ETA_RECUP_BATTERY_LO, _ETA_RECUP_BATTERY_HI),
    )'''

if "eta_recup_grid = numpyro.sample" not in me_text:
    assert old_sample_block in me_text, "eta_recup sample anchor missing"
    me_text = me_text.replace(old_sample_block, new_sample_block, 1)
    print("  OK  split eta_recup sample into grid / battery")
else:
    print("  --  eta_recup sample already split")

# 2c. Inject grid-aware selector + use eta_recup_effective in P_elec_traction
old_compute = '''    F = arrays["F_traction_N"]
    v = arrays["speed_mps"]
    T = arrays["temperature_K"]

    P_mech = F * v
    is_traction = P_mech >= 0.0
    regen_active = (P_mech < 0.0) & (v >= MIN_REGEN_SPEED_MPS)

    P_elec_traction = jnp.where(
        is_traction,
        P_mech / eta_prop,
        jnp.where(regen_active, P_mech * eta_recup, 0.0),
    )'''

new_compute = '''    F = arrays["F_traction_N"]
    v = arrays["speed_mps"]
    T = arrays["temperature_K"]
    grid_available = arrays["grid_available"] > 0.5

    P_mech = F * v
    is_traction = P_mech >= 0.0
    regen_active = (P_mech < 0.0) & (v >= MIN_REGEN_SPEED_MPS)

    # Grid-aware regen: catenary feedback (~motor+inverter) vs battery storage
    eta_recup_effective = jnp.where(grid_available, eta_recup_grid, eta_recup_battery)

    P_elec_traction = jnp.where(
        is_traction,
        P_mech / eta_prop,
        jnp.where(regen_active, P_mech * eta_recup_effective, 0.0),
    )'''

if "eta_recup_effective" not in me_text:
    assert old_compute in me_text, "compute anchor missing"
    me_text = me_text.replace(old_compute, new_compute, 1)
    print("  OK  P_elec_traction now uses grid-aware eta_recup")
else:
    print("  --  P_elec_traction already grid-aware")

# 2d. Update PARAM_NAMES
old_param_names = '''PARAM_NAMES: tuple[str, ...] = ("eta_prop", "eta_recup", "c_HVAC", "P_aux", "sigma_W")'''
new_param_names = '''PARAM_NAMES: tuple[str, ...] = (
    "eta_prop", "eta_recup_grid", "eta_recup_battery", "c_HVAC", "P_aux", "sigma_W",
)'''
if "eta_recup_grid" not in me_text.split("PARAM_NAMES")[1]:
    me_text = me_text.replace(old_param_names, new_param_names, 1)
    print("  OK  PARAM_NAMES updated")
else:
    print("  --  PARAM_NAMES already updated")

ME.write_text(me_text)

# =============================================================================
# 3. Extend two_stage_fit.py for --year-month ALL
# =============================================================================
print("\n[3] Extending two_stage_fit.py for --year-month ALL")

TS = REPO / "dump/scripts/two_stage_fit.py"
ts_text = TS.read_text()

old_year_months = '''    if args.year_month == "ALL_2021":
        year_months = tuple(f"2021-{m:02d}" for m in range(1, 13))
    else:
        year_months = (args.year_month,)'''

new_year_months = '''    if args.year_month == "ALL_2021":
        year_months = tuple(f"2021-{m:02d}" for m in range(1, 13))
    elif args.year_month == "ALL":
        # Full documented ZTBus range: May 2019 - Dec 2022
        ym = []
        for year in (2019, 2020, 2021, 2022):
            for month in range(1, 13):
                if year == 2019 and month < 5:
                    continue
                ym.append(f"{year}-{month:02d}")
        year_months = tuple(ym)
    else:
        year_months = (args.year_month,)'''

if 'args.year_month == "ALL"' not in ts_text:
    assert old_year_months in ts_text, "year-month anchor missing"
    ts_text = ts_text.replace(old_year_months, new_year_months, 1)
    TS.write_text(ts_text)
    print("  OK  two_stage_fit.py: --year-month ALL added")
else:
    print("  --  --year-month ALL already supported")

# =============================================================================
# 4. SLURM array job: one V100 per bus
# =============================================================================
print("\n[4] Writing slurm/fit_full_two_stage.sbatch")

SBATCH = REPO / "slurm/fit_full_two_stage.sbatch"
SBATCH.write_text('''#!/bin/bash -l
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
NVIDIA_LIBS=$(find "$VENV_SITE/nvidia" -name "lib" -type d 2>/dev/null | tr '\\n' ':')
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

uv run python dump/scripts/two_stage_fit.py \\
    --bus-id "$BUS_ID" \\
    --year-month ALL \\
    --subsample -1 \\
    --num-warmup 1000 \\
    --num-samples 2000 \\
    --num-chains 2 \\
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
''')
print(f"  OK  {SBATCH.relative_to(REPO)}")

# =============================================================================
# Summary
# =============================================================================
print()
print("=" * 60)
print("All four changes applied.")
print("=" * 60)
print()
print("Next steps:")
print()
print("  1. Verify the diff:")
print("       git diff src/ztbus/optim/data.py")
print("       git diff src/ztbus/optim/model_electrical.py")
print("       git diff dump/scripts/two_stage_fit.py")
print()
print("  2. Quick CPU smoke to confirm grid-aware works on July 2021:")
print("       salloc --partition=batch --time=00:30:00 "
      "--cpus-per-task=4 --mem=16G")
print("       cd ~/ztbus-energy-model")
print("       uv run python dump/scripts/two_stage_fit.py "
      "--year-month 2021-07 --subsample 20000")
print()
print("  3. If the smoke shows eta_recup_grid and eta_recup_battery both "
      "identified,")
print("     submit the full-corpus GPU array:")
print("       sbatch slurm/fit_full_two_stage.sbatch")
print("       squeue -u $USER")
print()
print("  4. Monitor:")
print("       squeue -u $USER")
print("       ls -la slurm/logs/ztbus-two-stage-*")
print("       tail -f slurm/logs/ztbus-two-stage-<JOBID>_0.out")
