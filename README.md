# ZTBus Energy Model

Physics-based energy-consumption model for the HESS lighTram® 19 trolley buses in the
[ZTBus dataset](https://www.research-collection.ethz.ch/entities/researchdata/61ac2f6e-2ca9-4229-8242-aed3b0c0d47c),
calibrated and validated against ~1409 driving missions recorded between April 2019 and
December 2022.

## What this project is

A reproducible pipeline that goes from raw ZTBus CSVs to a calibrated longitudinal
powertrain model whose parameters (frontal area `A`, rolling-resistance coefficient
`Crr`, propulsion and recuperation efficiencies, HVAC coefficient, auxiliary load) are
identified from data using the modelling framework of [Beckers, Paasche & Sundström,
TRD 2021](https://doi.org/10.1016/j.trd.2021.102776). The reference forward model is
the supervisor's `updated_energy_calculation.py`, refactored into `src/ztbus/physics/`
and extended with the unknowns made into a typed parameter object.


## Status

| Phase | What it covers | State |
|-------|----------------|-------|
| 0 | Project scaffold, env, conventions, archive of prior work | ✓ done |
| 1 | Raw CSV → Parquet ingest with schema validation, dataset profile | ✓ done |
| 2 | Mission-level cleaning (timestamps, speed, power, altitude, grade) | ✓ done |
| 3 | Feature engineering (mass, kinematics, distance, energy) | ✓ done |
| 4 | Route reconstruction (depot detection, GTFS map-match) | partial — depot done; map-match next |
| 5 | Parameter identification (per-bus, per-season, with cross-validation) | next |
| 6 | Validation against held-out missions, reporting, paper figures | next |

## Quick start

```bash
# 1. Install uv (one-time, no root needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the environment and install
uv sync --all-extras

# 3. Run sanity checks
uv run pytest -q

# 4. Point the pipeline at your raw data
export ZTBUS_RAW_DIR=/path/to/ztbus/csvs
uv run ztbus ingest --max-workers 8

# 5. On HPC, prefer the SLURM array submission
sbatch slurm/ingest.sbatch
```

See `docs/architecture.md` for the design rationale and `docs/hpc_runbook.md` for
the cluster-side workflow.

## Layout

See `docs/architecture.md` — kept in sync with the actual tree.

## Relation to prior work

The previous exploratory scripts and PDFs live in `legacy/`, untouched, with
a `legacy/README.md` explaining what was kept and what was replaced. None of that code
is on the import path of the current pipeline, still pending in the tasks to be done.

## References

1. Widmer, F., Ritter, A., Onder, C. H. *ZTBus: A Large Dataset of Time-Resolved City
   Bus Driving Missions.* Scientific Data 10, 687 (2023). doi:10.1038/s41597-023-02600-6
2. Beckers, C. J. J., Paasche, A., Sundström, O. *A battery electric bus energy
   consumption model for strategic purposes: Validation of a proposed model structure
   with data from bus fleets in China and Norway.* Transportation Research Part D 96
   (2021). doi:10.1016/j.trd.2021.102776
