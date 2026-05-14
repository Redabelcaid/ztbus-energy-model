"""Add windowed-grade recomputation to data.py.

What this adds
--------------
A new function ``_compute_grade_windowed`` in data.py and a new parameter
``recompute_grade=True`` on ``load_corpus``. When enabled (default), the
loader replaces the parquet's ``grade`` column with a recomputed version
that uses a sliding window large enough to span ``min_ds_m`` metres of
travel distance. Samples where insufficient distance is covered get
``NaN`` and are dropped by the existing null-drop filter.

Why inline in data.py rather than in cleaning/grade.py?
-------------------------------------------------------
Iteration speed. Putting it here lets us re-run the smoke immediately
with the new grade, without the 22-minute full cleaning re-run. Once we
confirm the windowed grade fixes the bottleneck, we promote it back to
``cleaning/grade.py`` and re-run cleaning properly (Issue 1 in the
handoff).

Apply this on the cluster:
    cd ~/ztbus-energy-model
    uv run python dump/scripts/add_windowed_grade.py
"""

from pathlib import Path

DATA_PY = Path("src/ztbus/optim/data.py")
text = DATA_PY.read_text()

# ---------------------------------------------------------------------------
# 1. Add the windowed-grade helper near _detect_temperature_unit
# ---------------------------------------------------------------------------
helper_marker = "def _detect_temperature_unit(lf: pl.LazyFrame) -> str:"
assert helper_marker in text, "Couldn't find _detect_temperature_unit anchor"

helper_addition = '''def _add_windowed_grade(
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
    )


'''

text = text.replace(helper_marker, helper_addition + helper_marker)

# ---------------------------------------------------------------------------
# 2. Add the recompute_grade parameter to load_corpus signature
# ---------------------------------------------------------------------------
old_sig = """    require_clean_flags: bool = True,
    require_gnss_course_valid: bool = False,"""
new_sig = """    require_clean_flags: bool = True,
    require_gnss_course_valid: bool = False,
    recompute_grade: bool = True,
    grade_half_window_samples: int = 5,
    grade_min_ds_m: float = 10.0,"""

assert old_sig in text, "Couldn't find load_corpus signature anchor"
text = text.replace(old_sig, new_sig)

# ---------------------------------------------------------------------------
# 3. Apply recompute_grade after temperature unit detection, before filters
# ---------------------------------------------------------------------------
old_temp_block = """    logger.info("Detected temperature unit: %s", temp_unit)

    # ---- Apply filters (each step audits the row count drop) ------------"""
new_temp_block = """    logger.info("Detected temperature unit: %s", temp_unit)

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

    # ---- Apply filters (each step audits the row count drop) ------------"""

assert old_temp_block in text, "Couldn't find temperature/filters anchor"
text = text.replace(old_temp_block, new_temp_block)

DATA_PY.write_text(text)

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
final = DATA_PY.read_text()
assert "_add_windowed_grade" in final
assert "recompute_grade: bool = True" in final
assert "Recomputed grade with windowed dh/ds" in final
print("✓ _add_windowed_grade helper inserted")
print("✓ load_corpus gained recompute_grade / grade_half_window_samples / grade_min_ds_m")
print("✓ recomputation applied between temp detection and filter cascade")
print()
print("Default behaviour: grade is recomputed with half_window=5 samples (±5s),")
print("requiring at least 10 m of distance over the window. Samples with less")
print("distance get NaN and are dropped by the existing null filter.")
print()
print("Verify with: git diff src/ztbus/optim/data.py")
