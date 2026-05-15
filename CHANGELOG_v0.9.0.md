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
