"""Two small follow-up fixes after v3 smoke run:

1. Widen P_aux prior lower bound from 1.0 to 0.1 kW. The posterior was
   pegged at 1.0 in both v2 and v3 — data wants the value lower.

2. Fix the mission-boundary leak in the windowed grade. shift() on a
   concatenated LazyFrame crosses file boundaries; we partition by source
   file path so grade is computed strictly within each mission.

Apply on the cluster:
    uv run python dump/scripts/v3_followups.py
"""

from pathlib import Path

# ===========================================================================
# Fix 1: widen P_aux lower bound in model.py
# ===========================================================================

MODEL_PY = Path("src/ztbus/optim/model.py")
model_text = MODEL_PY.read_text()

old_p_aux = "_P_AUX_LO_KW: float = 1.0"
new_p_aux = "_P_AUX_LO_KW: float = 0.1"
assert old_p_aux in model_text, "Couldn't find _P_AUX_LO_KW anchor"
model_text = model_text.replace(old_p_aux, new_p_aux)

MODEL_PY.write_text(model_text)
print("✓ model.py: _P_AUX_LO_KW 1.0 → 0.1 kW")


# ===========================================================================
# Fix 2: per-mission windowed grade in data.py
# ===========================================================================

DATA_PY = Path("src/ztbus/optim/data.py")
data_text = DATA_PY.read_text()

# Replace _add_windowed_grade so shifts operate per source-file.
old_fn = '''def _add_windowed_grade(
    lf: pl.LazyFrame,
    *,
    half_window_samples: int = 5,
    min_ds_m: float = 10.0,
) -> pl.LazyFrame:
    """Replace ``grade`` with a windowed dh/ds computation.

    The original cleaning pipeline computes ``grade`` as a naive per-sample
    finite difference, which blows up at low speed (`02_grade_diagnostic.png`
    shows spikes to 15{,}500 %). This helper recomputes grade over a sliding
    window of ``2 * half_window_samples + 1`` samples (default ±5 = 11 s at 1 Hz),
    and only emits a value where the cumulative travelled distance over the
    window is at least ``min_ds_m`` metres. Otherwise grade is null and the
    sample is dropped by the loader's null filter.

    This is a temporary inline fix; once validated it should be promoted to
    ``cleaning/grade.py`` and the cleaning pipeline re-run.
    """
    w = half_window_samples
    return (
        lf.with_columns(
            h_diff_=(
                pl.col("altitude_smoothed_m").shift(-w)
                - pl.col("altitude_smoothed_m").shift(w)
            ),
            s_diff_=(
                pl.col("distance_m").shift(-w) - pl.col("distance_m").shift(w)
            ),
        )
        .with_columns(
            grade=pl.when(pl.col("s_diff_") >= min_ds_m)
            .then(pl.col("h_diff_") / pl.col("s_diff_"))
            .otherwise(None),
        )
        .drop(["h_diff_", "s_diff_"])
    )'''

new_fn = '''def _add_windowed_grade(
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
    )'''

assert old_fn in data_text, "Couldn't find _add_windowed_grade — was it modified?"
data_text = data_text.replace(old_fn, new_fn)

# Inject __mission_path__ column at scan time, so the per-file partitioning works.
old_scan = """    # Scan all paths as one LazyFrame
    lf = pl.scan_parquet([str(p) for p in paths])"""

new_scan = """    # Scan each parquet separately and attach a __mission_path__ column so
    # downstream per-mission window operations don't leak across files.
    per_file_frames = [
        pl.scan_parquet(str(p)).with_columns(
            __mission_path__=pl.lit(str(p)),
        )
        for p in paths
    ]
    lf = pl.concat(per_file_frames, how="vertical_relaxed")"""

assert old_scan in data_text, "Couldn't find scan_parquet anchor"
data_text = data_text.replace(old_scan, new_scan)

DATA_PY.write_text(data_text)
print("✓ data.py: _add_windowed_grade uses .over('__mission_path__')")
print("✓ data.py: scan injects __mission_path__ column per file")

# ===========================================================================
# Sanity verifications
# ===========================================================================

assert "_P_AUX_LO_KW: float = 0.1" in MODEL_PY.read_text()
assert "__mission_path__" in DATA_PY.read_text()
assert ".over(partition_col)" in DATA_PY.read_text()

print()
print("Both patches applied.")
print("Verify with: git diff src/ztbus/optim/model.py src/ztbus/optim/data.py")
