"""Data loader for Phase 5 — consumes the cleaning pipeline outputs.

This module is the ONLY supported way to assemble JAX arrays for the
parameter-identification model. It exists for two reasons:

1. **Make bypassing the cleaning pipeline impossible by accident.** The
   colleagues' CMA-ES and IAGWO scripts re-derived speed, acceleration, grade,
   and mass from raw signals — duplicating (and breaking) the work in
   ``src/ztbus/cleaning/``. By centralising all I/O here, anyone writing a new
   optimizer / sampler just calls ``load_corpus(...)`` and gets the right
   thing.

2. **Make filtering choices auditable.** Every mask logs how many rows it
   drops. The final summary tells the methods section of the eventual paper
   exactly which samples contributed to the fit and which did not.

Architecture
------------
The loader uses Polars LazyFrames end-to-end, chaining filters as expressions
so nothing materialises until the final ``.collect()``. The full cleaned
corpus is ~48 M samples × ~10 columns × float64 = ~3.8 GB if loaded eagerly;
LazyFrames keep peak memory bounded.

Column conventions (from the cleaned-parquet schema)
----------------------------------------------------
::

    speed_smoothed_mps          → "speed_mps" in JAX dict
    acceleration_mps2           → "acceleration_mps2"
    mass_kg                     → "mass_kg"
    grade                       → "grade"
    temperature_ambient (°C)    → "temperature_K"  (converted)
    electric_powerDemand (W)    → "P_obs_W"        (likelihood target)

The forward kernel in ``optim/kernels.py`` expects exactly these JAX dict
keys. The user-facing names match the symbols in Hjelkrem 2021.

Filter cascade (default-on; each is an explicit parameter to override)
----------------------------------------------------------------------
1. ``fit_eligible_missions``: only load missions that passed the QC gates
   (optional caller-supplied set).
2. ``~in_depot``: drop depot-phase samples — ADR 0002 says traction params
   should be fit on motion samples only.
3. ``speed_smoothed_mps > speed_threshold_mps``: drop near-stationary
   samples where the regen physics is undefined (Hjelkrem's 15 km/h gate
   in spirit, but applied as a lower threshold rather than a regen kill).
4. ``|grade| <= grade_clip``: drop unphysical grade outliers (this is the
   provisional workaround until Issue 1 — windowed grade — lands).
5. All ``*_flag == False`` (i.e. clean samples only). The relevant flag
   columns are listed in :data:`_QUALITY_FLAGS_DROP_TRUE`.
6. ``gnss_course_valid == True`` (opposite polarity — keep when True).
7. Drop rows with nulls in any modelling column.
8. Optional random subsample for smoke-running.

The result is one flat JAX dict ready to feed into ``ztbus_model``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column conventions — single source of truth for cleaning ↔ JAX naming
# ---------------------------------------------------------------------------

# Parquet column → JAX dict key. Order matters for downstream consistency.
_PARQUET_TO_JAX_INPUT: dict[str, str] = {
    "speed_smoothed_mps": "speed_mps",
    "traction_tractionForce": "F_traction_N",
    "acceleration_mps2": "acceleration_mps2",
    "mass_kg": "mass_kg",
    "grade": "grade",
    "temperature_ambient": "temperature_K",  # convert below
}

# Likelihood target
_PARQUET_OBS_COLUMN: str = "electric_powerDemand"
_JAX_OBS_KEY: str = "P_obs_W"

# Boolean flags where True means "bad sample, drop it"
_QUALITY_FLAGS_DROP_TRUE: tuple[str, ...] = (
    "time_gap_flag",
    "speed_negative_flag",
    "speed_outlier_flag",
    "power_outlier_flag",
    "temperature_outlier_flag",
    "grade_outlier_flag",
)

# Boolean flags where True means "good sample, keep it"
_QUALITY_FLAGS_KEEP_TRUE: tuple[str, ...] = ("gnss_course_valid",)

# Depot mask (Boolean, True = in depot, default-drop)
_DEPOT_COLUMN: str = "in_depot"


# ---------------------------------------------------------------------------
# Audit dataclass — every filter step records what it dropped
# ---------------------------------------------------------------------------


@dataclass
class LoadAudit:
    """Row-level audit of every filter step. Use this in the methods section."""

    n_missions_total: int = 0
    n_missions_fit_eligible: int = 0
    n_samples_raw: int = 0
    drops: dict[str, int] = field(default_factory=dict)
    n_samples_final: int = 0
    n_samples_after_subsample: int = 0

    def log(self) -> None:
        """Pretty-print the audit trail to the logger."""
        logger.info("=== load_corpus audit ===")
        logger.info(
            "Missions: %d total, %d fit-eligible",
            self.n_missions_total,
            self.n_missions_fit_eligible,
        )
        logger.info("Samples raw: %d", self.n_samples_raw)
        running = self.n_samples_raw
        for filter_name, dropped in self.drops.items():
            running -= dropped
            logger.info("  - %-32s drops %10d  (running %d)", filter_name, dropped, running)
        logger.info("Samples final (pre-subsample): %d", self.n_samples_final)
        if self.n_samples_after_subsample != self.n_samples_final:
            logger.info("Samples after subsample: %d", self.n_samples_after_subsample)

    def to_dict(self) -> dict[str, int | dict[str, int]]:
        return {
            "n_missions_total": self.n_missions_total,
            "n_missions_fit_eligible": self.n_missions_fit_eligible,
            "n_samples_raw": self.n_samples_raw,
            "drops": dict(self.drops),
            "n_samples_final": self.n_samples_final,
            "n_samples_after_subsample": self.n_samples_after_subsample,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _discover_parquet_paths(
    processed_dir: Path,
    bus_ids: Iterable[str] | None,
    year_months: Iterable[str] | None,
) -> list[Path]:
    """Find parquet files matching the hive-partition filters.

    Hive layout: ``{processed_dir}/bus={bus_id}/year={YYYY}/month={MM}/*.parquet``.
    """
    if not processed_dir.exists():
        raise FileNotFoundError(f"processed_dir does not exist: {processed_dir}")

    bus_filter = set(bus_ids) if bus_ids else None
    ym_filter = set(year_months) if year_months else None

    paths: list[Path] = []
    for parquet in processed_dir.rglob("*.parquet"):
        # Parse the hive partition out of the path
        parts = {kv.split("=")[0]: kv.split("=")[1] for kv in parquet.parts if "=" in kv}
        if bus_filter is not None and parts.get("bus") not in bus_filter:
            continue
        if ym_filter is not None:
            year = parts.get("year")
            month = parts.get("month")
            if year is None or month is None:
                continue
            if f"{year}-{month}" not in ym_filter:
                continue
        paths.append(parquet)
    return sorted(paths)


def _add_windowed_grade(
    lf: pl.LazyFrame,
    *,
    half_window_samples: int = 5,
    min_ds_m: float = 10.0,
    partition_col: str | None = "__mission_path__",
) -> pl.LazyFrame:
    """Replace ``grade`` with a windowed dh/ds computation.

    The original cleaning pipeline computes ``grade`` as a naive per-sample
    finite difference, which blows up at low speed (`02_grade_diagnostic.png`
    shows spikes to 15{,}500 %). This helper recomputes grade over a sliding
    window of ``2 * half_window_samples + 1`` samples (default ±5 = 11 s at 1 Hz),
    and only emits a value where the cumulative travelled distance over the
    window is at least ``min_ds_m`` metres. Otherwise grade is null and the
    sample is dropped by the loader's null filter.

    The ``shift()`` operations are constrained to operate strictly within each
    mission (using ``over(partition_col)``) so we never leak altitude / distance
    values across mission boundaries when scanning a multi-file LazyFrame.
    The partition column is set by the caller; see ``load_corpus`` for how the
    ``__mission_path__`` column is injected.

    This is a temporary inline fix; once validated it should be promoted to
    ``cleaning/grade.py`` and the cleaning pipeline re-run.
    """
    w = half_window_samples
    h = pl.col("altitude_smoothed_m")
    s = pl.col("distance_m")
    if partition_col is not None:
        h_diff = h.shift(-w).over(partition_col) - h.shift(w).over(partition_col)
        s_diff = s.shift(-w).over(partition_col) - s.shift(w).over(partition_col)
    else:
        h_diff = h.shift(-w) - h.shift(w)
        s_diff = s.shift(-w) - s.shift(w)
    return (
        lf.with_columns(h_diff_=h_diff, s_diff_=s_diff)
        .with_columns(
            grade=pl.when(pl.col("s_diff_") >= min_ds_m)
            .then(pl.col("h_diff_") / pl.col("s_diff_"))
            .otherwise(None),
        )
        .drop(["h_diff_", "s_diff_"])
    )


def _detect_temperature_unit(lf: pl.LazyFrame) -> str:
    """Return 'kelvin' or 'celsius' by sniffing the value range.

    Bus-cabin ambient is roughly -20..+40 °C = 253..313 K. The two ranges
    don't overlap, so the heuristic ``max > 100`` is robust.
    """
    max_val = lf.select(pl.col("temperature_ambient").max()).collect().item()
    if max_val is None:
        return "celsius"  # fallback; will fail later if truly null
    return "kelvin" if max_val > 100.0 else "celsius"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_corpus(
    processed_dir: str | Path,
    *,
    bus_ids: Iterable[str] | None = ("183", "208"),
    year_months: Iterable[str] | None = None,
    fit_eligible_missions: Iterable[str] | None = None,
    exclude_depot: bool = True,
    grade_clip: float = 0.12,
    speed_threshold_mps: float = 0.5,
    require_clean_flags: bool = True,
    require_gnss_course_valid: bool = False,
    recompute_grade: bool = True,
    grade_half_window_samples: int = 5,
    grade_min_ds_m: float = 10.0,
    subsample: int | None = None,
    subsample_seed: int = 0,
    dtype: jnp.dtype = jnp.float64,
) -> tuple[dict[str, jnp.ndarray], LoadAudit]:
    """Load the cleaned corpus and produce JAX arrays for ``ztbus_model``.

    Parameters
    ----------
    processed_dir
        Root of the hive-partitioned cleaned parquets, e.g.
        ``/scratch/users/$USER/ztbus/processed``.
    bus_ids
        Bus IDs to include. Default: both ZTBus buses. Pass ``None`` for all.
    year_months
        Optional list of "YYYY-MM" strings for temporal filtering, e.g.
        ``["2021-06", "2021-07"]`` for a single summer.
    fit_eligible_missions
        Optional set of mission stem names (parquet filename without .parquet)
        that passed the QC gates. If ``None``, all discovered missions are
        loaded. Pass a set when you have the QC artefact available.
    exclude_depot
        Drop samples where ``in_depot == True``. Default ``True`` per ADR 0002.
        Override only if intentionally fitting depot-dominant params
        (e.g. P_aux from layover, c_HVAC from extreme ΔT).
    grade_clip
        Drop |grade| > this value. Default 0.12 (12%) — Zurich's steepest
        streets are ~10%, so this is a generous physical bound and drops
        the low-speed numerical-derivative outliers that Issue 1 will fix
        properly. Override to e.g. 1.0 to keep everything.
    speed_threshold_mps
        Drop ``speed_smoothed_mps <= this``. Default 0.5 m/s (~1.8 km/h) —
        Hjelkrem's regen physics is undefined near zero speed.
    require_clean_flags
        Drop rows where any of the cleaning ``*_flag`` columns is True.
    require_gnss_course_valid
        Drop rows where ``gnss_course_valid`` is False. Default ``False``
        because the optimizer doesn't use course directly, only speed.
    subsample
        If set, randomly subsample to this many rows after all filters.
        Useful for smoke-runs; leave ``None`` for production.
    subsample_seed
        RNG seed for the random subsample.
    dtype
        JAX dtype for the output arrays. Default ``float64`` matches the
        kernel parity tests; use ``float32`` for half-the-memory production
        runs once you've validated convergence.

    Returns
    -------
    arrays
        Dict with keys ``speed_mps``, ``acceleration_mps2``, ``mass_kg``,
        ``grade``, ``temperature_K``, ``P_obs_W`` — JAX arrays of equal
        length, ready to pass to ``ztbus_model(data=arrays, observed_power_W=
        arrays['P_obs_W'])``.
    audit
        :class:`LoadAudit` describing exactly how many samples each filter
        dropped. Log it or attach it to the output for reproducibility.

    Raises
    ------
    FileNotFoundError
        If ``processed_dir`` doesn't exist or no parquets match the filters.
    """
    processed_dir = Path(processed_dir)
    audit = LoadAudit()

    # ---- Discover & filter at file level --------------------------------
    paths = _discover_parquet_paths(processed_dir, bus_ids, year_months)
    audit.n_missions_total = len(paths)

    if fit_eligible_missions is not None:
        eligible = set(fit_eligible_missions)
        paths = [p for p in paths if p.stem in eligible]
    audit.n_missions_fit_eligible = len(paths)

    if not paths:
        raise FileNotFoundError(
            f"No parquets matched filters under {processed_dir} "
            f"(bus_ids={bus_ids}, year_months={year_months})"
        )

    # Scan each parquet separately and attach a __mission_path__ column so
    # downstream per-mission window operations don't leak across files.
    per_file_frames = [
        pl.scan_parquet(str(p)).with_columns(
            __mission_path__=pl.lit(str(p)),
        )
        for p in paths
    ]
    lf = pl.concat(per_file_frames, how="vertical_relaxed")

    # ---- Row count BEFORE any filter -----------------------------------
    audit.n_samples_raw = lf.select(pl.len()).collect().item()

    # ---- Temperature unit detection + conversion ------------------------
    temp_unit = _detect_temperature_unit(lf)
    if temp_unit == "celsius":
        lf = lf.with_columns(
            (pl.col("temperature_ambient") + 273.15).alias("temperature_K_internal"),
        )
    else:
        lf = lf.with_columns(
            pl.col("temperature_ambient").alias("temperature_K_internal"),
        )
    logger.info("Detected temperature unit: %s", temp_unit)

    # ---- Optionally recompute grade with windowed dh/ds -----------------
    if recompute_grade:
        lf = _add_windowed_grade(
            lf,
            half_window_samples=grade_half_window_samples,
            min_ds_m=grade_min_ds_m,
        )
        logger.info(
            "Recomputed grade with windowed dh/ds (half_window=%d, min_ds=%.1f m)",
            grade_half_window_samples,
            grade_min_ds_m,
        )

    # ---- Apply filters (each step audits the row count drop) ------------
    def _count_and_filter(lf_in: pl.LazyFrame, predicate: pl.Expr, label: str) -> pl.LazyFrame:
        before = lf_in.select(pl.len()).collect().item()
        lf_out = lf_in.filter(predicate)
        after = lf_out.select(pl.len()).collect().item()
        audit.drops[label] = before - after
        return lf_out

    if exclude_depot:
        lf = _count_and_filter(lf, ~pl.col(_DEPOT_COLUMN), "exclude_depot")

    lf = _count_and_filter(
        lf,
        pl.col("speed_smoothed_mps") > speed_threshold_mps,
        f"speed > {speed_threshold_mps} m/s",
    )

    lf = _count_and_filter(
        lf,
        pl.col("grade").abs() <= grade_clip,
        f"|grade| <= {grade_clip}",
    )

    if require_clean_flags:
        for flag in _QUALITY_FLAGS_DROP_TRUE:
            lf = _count_and_filter(lf, ~pl.col(flag), f"~{flag}")

    if require_gnss_course_valid:
        for flag in _QUALITY_FLAGS_KEEP_TRUE:
            lf = _count_and_filter(lf, pl.col(flag), flag)

    # Drop rows with nulls in any modelling column. Polars treats NaN
    # separately from null, so we filter on both.
    modelling_cols = [*_PARQUET_TO_JAX_INPUT.keys(), _PARQUET_OBS_COLUMN]
    modelling_cols = [
        c if c != "temperature_ambient" else "temperature_K_internal" for c in modelling_cols
    ]
    null_predicate = pl.all_horizontal([pl.col(c).is_not_null() for c in modelling_cols])
    lf = _count_and_filter(lf, null_predicate, "drop_nulls")

    # ---- Materialise the small subset we actually need ------------------
    select_cols = [*_PARQUET_TO_JAX_INPUT.keys(), _PARQUET_OBS_COLUMN]
    # Replace temperature_ambient with the converted column
    select_cols = [
        c if c != "temperature_ambient" else "temperature_K_internal" for c in select_cols
    ]
    df = lf.select(select_cols).collect()
    audit.n_samples_final = df.height

    # ---- Optional subsample ---------------------------------------------
    if subsample is not None and subsample < df.height:
        df = df.sample(n=subsample, seed=subsample_seed, shuffle=True)
    audit.n_samples_after_subsample = df.height

    # ---- Build JAX dict --------------------------------------------------
    arrays: dict[str, jnp.ndarray] = {}
    for parquet_col, jax_key in _PARQUET_TO_JAX_INPUT.items():
        src_col = "temperature_K_internal" if parquet_col == "temperature_ambient" else parquet_col
        arrays[jax_key] = jnp.asarray(df[src_col].to_numpy(), dtype=dtype)
    arrays[_JAX_OBS_KEY] = jnp.asarray(df[_PARQUET_OBS_COLUMN].to_numpy(), dtype=dtype)

    audit.log()
    return arrays, audit


__all__ = ["LoadAudit", "load_corpus"]
