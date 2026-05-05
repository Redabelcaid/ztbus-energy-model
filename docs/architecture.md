# Architecture

This document explains *why* the project is laid out the way it is, so that
choices can be challenged on substance rather than rediscovered later.

## Research framing

The supervisor's `updated_energy_calculation.py` defines a longitudinal
powertrain model with seven scalar parameters whose values are unknown and
must be identified from data:

| Parameter                          | Symbol    | Unit  |
|------------------------------------|-----------|-------|
| Frontal area                       | A         | m²    |
| Drag coefficient                   | Cd        | –     |
| Rolling resistance coefficient     | Crr       | –     |
| Propulsion efficiency              | η_prop    | –     |
| Recuperation efficiency            | η_recup   | –     |
| HVAC coefficient (linear)          | c_HVAC    | kW/K  |
| Auxiliary load                     | P_aux     | kW    |

The forward model is

    F_total = m·g·Crr  +  ½·ρ·Cd·A·v²  +  m·a  +  m·g·θ
    P_mech  = F_total · v
    P_elec  = P_mech / η_prop      if P_mech ≥ 0
              P_mech · η_recup     otherwise
    P_total = P_elec + c_HVAC·|T − T_comfort|·1000 + P_aux·1000

with `electric_powerDemand` as the observed signal we fit against. This
matches the strategic-model decomposition validated in Beckers et al. (2021)
on Chinese and Norwegian fleets.

The deliverable is therefore an *identification* study, not a cleaning
exercise. Cleaning is in service of identification.

## Layout principles

1. **Separate raw, interim, processed, and reports.** Cleaning never modifies
   raw. Interim is a faithful Parquet copy. Processed is fully cleaned + feature-
   engineered. Reports are derived artifacts (plots, summary tables).
2. **Configs are data, not code.** Cleaning thresholds, physics priors, and
   HPC defaults live in YAML so a reviewer can audit assumptions without
   reading code, and so we can sweep over configurations.
3. **One mission = one Parquet = one SLURM task.** The dataset's natural unit
   of independence is the driving mission. Scaling is by array job, not by
   distributed dataframe.
4. **Pure-function pipeline stages.** Each cleaning step takes a `polars.DataFrame`
   and returns one. This makes them composable, testable, and easy to reason
   about in any order.
5. **Physical invariants pinned in tests.** The forward model has tests that
   pass for any reasonable parameter choice (e.g., "stationary at comfort
   temperature draws only the auxiliary power"). These survive parameter
   identification, unlike numeric snapshot tests.

## Pipeline stages

```
raw CSVs ──► ingest ──► interim Parquet ──► clean ──► processed Parquet ──► fit ──► parameters + reports
                                                       │
                                                       ├─► routes (depot detection, GTFS map-match)
                                                       └─► features (kinematics, mass, thermal)
```

Each arrow is a Snakemake rule. Each rule is one (or many) SLURM tasks on the cluster.

## Why Polars instead of pandas

Polars' Arrow-backed columnar layout writes Parquet without a copy, runs the
critical kernels in Rust, and supports streaming over datasets larger than
memory. For 10 GB of CSV with mostly numeric columns and time-series ops,
benchmarks consistently show 5–20× speedups over pandas with substantially
lower memory peaks. The migration cost is low: Polars' expression API is
similar enough that the few `pandas` idioms in the colleagues' script translate
in an afternoon. We pay this once.

## Why uv

`uv` produces deterministic environments from `pyproject.toml + uv.lock` in
seconds, installs without root, and works identically on a laptop and on an
HPC login node. Conda would also work but adds tooling weight that we don't
need, and is harder to bootstrap on locked-down clusters.

## Why Snakemake

Snakemake gives us a DAG, incremental reruns, first-class SLURM execution
via `snakemake-executor-plugin-slurm`, and per-rule resource specs. The same
workflow runs on a laptop with `--cores 8` and on the cluster with
`--executor slurm --jobs 100`.

## Contracts between modules

- `ztbus.io` returns `polars.DataFrame` with the schema documented in
  `configs/data/ztbus.yaml`. No cleaning here, ever.
- `ztbus.cleaning` consumes a raw DataFrame and a cleaning config, returns a
  cleaned DataFrame plus a per-mission QC dict.
- `ztbus.features` consumes a cleaned DataFrame, returns a feature-engineered
  DataFrame. No cleaning leaks in here either.
- `ztbus.physics.simulate_powertrain` consumes plain numpy arrays + typed
  parameter objects. It does not know about Polars or files.
- `ztbus.optim` consumes physics + processed data, returns identified
  parameters + uncertainty diagnostics.

These contracts let us swap implementations (e.g. try a different cleaning
policy or a different optimizer) without rippling changes through the codebase.
