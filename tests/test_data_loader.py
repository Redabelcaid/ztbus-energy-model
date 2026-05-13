"""Tests for the corpus data loader.

The loader hits the actual cleaned-parquet store, so these tests are marked
``hpc`` and skipped automatically when run off-cluster. They verify three
things any reviewer should be able to check:

1. The loader returns the JAX dict keys the model expects.
2. Every default-on filter actually drops rows compared to a no-filter call.
3. The audit accounting balances: raw_count − sum(drops) − subsample_drop = final.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from ztbus.optim.data import load_corpus

# ---------------------------------------------------------------------------
# Fixture: locate the processed-parquet root
# ---------------------------------------------------------------------------

_PROCESSED_ENV_VAR = "ZTBUS_PROCESSED_DIR"
_PROCESSED_DEFAULT = "/scratch/users/rbelcaid/ztbus/processed"


def _processed_dir_or_skip() -> Path:
    candidate = Path(os.environ.get(_PROCESSED_ENV_VAR, _PROCESSED_DEFAULT))
    if not candidate.exists():
        pytest.skip(
            f"Cleaned-parquet store not found at {candidate}. Set "
            f"{_PROCESSED_ENV_VAR} or run on the cluster."
        )
    return candidate


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.hpc
def test_load_corpus_returns_expected_keys() -> None:
    """The JAX dict must contain exactly the keys ztbus_model expects."""
    processed_dir = _processed_dir_or_skip()
    # Use a tiny subsample to keep the test under a few seconds
    arrays, _ = load_corpus(
        processed_dir,
        bus_ids=("183",),
        year_months=("2021-06",),
        subsample=1000,
    )
    expected_keys = {
        "speed_mps",
        "acceleration_mps2",
        "mass_kg",
        "grade",
        "temperature_K",
        "P_obs_W",
    }
    assert set(arrays.keys()) == expected_keys


@pytest.mark.hpc
def test_load_corpus_arrays_are_same_length() -> None:
    """All arrays in the dict must be parallel — same length, same order."""
    processed_dir = _processed_dir_or_skip()
    arrays, _ = load_corpus(processed_dir, subsample=2000)
    lengths = {k: v.shape[0] for k, v in arrays.items()}
    assert len(set(lengths.values())) == 1, f"Inconsistent lengths: {lengths}"


@pytest.mark.hpc
def test_temperature_is_in_kelvin() -> None:
    """The loader must auto-detect Celsius and convert. Output should be in K."""
    processed_dir = _processed_dir_or_skip()
    arrays, _ = load_corpus(processed_dir, subsample=2000)
    T = arrays["temperature_K"]
    # Bus ambient: cold day -20°C = 253 K; hot day +40°C = 313 K. Anything
    # in 100..400 K is plausible; <100 means we missed the conversion.
    assert float(T.min()) > 100.0
    assert float(T.max()) < 400.0


@pytest.mark.hpc
def test_default_filters_actually_drop_rows() -> None:
    """Every default-on filter should leave fewer rows than no-filter."""
    processed_dir = _processed_dir_or_skip()
    _, audit_unfiltered = load_corpus(
        processed_dir,
        bus_ids=("183",),
        year_months=("2021-06",),
        exclude_depot=False,
        grade_clip=10.0,  # effectively disabled
        speed_threshold_mps=-1.0,  # effectively disabled
        require_clean_flags=False,
        subsample=None,
    )
    _, audit_filtered = load_corpus(
        processed_dir,
        bus_ids=("183",),
        year_months=("2021-06",),
        subsample=None,
        # All defaults active
    )
    assert audit_filtered.n_samples_final < audit_unfiltered.n_samples_final, (
        f"Filters didn't drop anything: unfiltered={audit_unfiltered.n_samples_final}, "
        f"filtered={audit_filtered.n_samples_final}"
    )


@pytest.mark.hpc
def test_audit_accounting_balances() -> None:
    """raw_count − sum(drops) == final."""
    processed_dir = _processed_dir_or_skip()
    _, audit = load_corpus(
        processed_dir,
        bus_ids=("183",),
        year_months=("2021-06",),
        subsample=None,
    )
    accounted = audit.n_samples_raw - sum(audit.drops.values())
    assert accounted == audit.n_samples_final, (
        f"Audit imbalance: raw={audit.n_samples_raw} - drops={sum(audit.drops.values())} "
        f"= {accounted}, but final={audit.n_samples_final}"
    )


@pytest.mark.hpc
def test_subsample_caps_final_size() -> None:
    """Subsampling must clip to the requested size."""
    processed_dir = _processed_dir_or_skip()
    arrays, audit = load_corpus(
        processed_dir,
        bus_ids=("183",),
        year_months=("2021-06",),
        subsample=500,
    )
    assert arrays["speed_mps"].shape[0] <= 500
    assert audit.n_samples_after_subsample <= 500


@pytest.mark.hpc
def test_dtype_is_float64_by_default() -> None:
    processed_dir = _processed_dir_or_skip()
    arrays, _ = load_corpus(processed_dir, subsample=500)
    assert arrays["speed_mps"].dtype == jnp.float64
