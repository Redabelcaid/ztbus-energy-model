# Legacy

This directory archives prior work in full, untouched. None of it is on the
active import path of the current pipeline. It exists because (a) it contains
real intuitions worth preserving as references, and (b) reproducibility means
we can always reconstruct what the previous semester actually did.

Do not edit files in this directory. If a useful idea here needs to be carried
into the active codebase, port it explicitly into `src/ztbus/` with a comment
referencing the legacy file.

## Files

### `prepare_ztbus_v0.py` (formerly `prepare_ztbus.py`)

Previous semester's data-preparation script. It loads CSVs with pandas, applies
a flat cleaning pipeline, derives a few features, and writes one cleaned CSV +
one summary CSV + one preview CSV per mission.

What we kept as ideas in the new pipeline:
- The instinct that negative power should be preserved (it represents
  regenerative braking).
- Acceleration plausibility bounds in the few-m/s² range.
- Forward/back-fill for `itcs_*` columns (these update at stops, not every
  second).

What we did differently in the new pipeline (see `docs/architecture.md` for
rationale):
- Removed the hardcoded Windows path. Paths come from configs / env vars.
- Removed `warnings.filterwarnings("ignore")`. Surfacing warnings is the
  point of running tests.
- Replaced pandas with Polars; replaced CSV outputs with partitioned Parquet.
- Replaced `interpolate(limit_direction="both")` on every numeric column with
  signal-specific policies (in particular, GNSS coordinates and altitude get
  short-gap-only interpolation; the rest left as NaN with flags).
- Replaced `np.cumsum(P) * dt_median` with trapezoidal integration on the
  actual time vector.
- Moved cleaning thresholds to `configs/cleaning/v1.yaml`.
- Made each cleaning step a pure function with tests.

### `updated_energy_calculation_supervisor_original.py`

The supervisor's reference forward model with parameter placeholders left
as `?`. The cleaned-up, typed, tested version lives in
`src/ztbus/physics/powertrain.py`. Behavior matches where the original was
correct; explicit fixes (mass default, energy integration, HVAC unit) are
documented in the docstring of the new module.

### `firstAttempt_atCleaning.pdf`

Slide deck of the previous semester's cleaning rationale. Useful as a record
of what was considered. The thresholds and decisions in
`configs/cleaning/v1.yaml` are *informed* by it but are anchored to the ZTBus
paper's reported ranges and to what the parameter-identification problem
needs from each signal, rather than copied wholesale.
