"""NUTS sampler runner for Phase 5 — runs Bayesian inference on the model.

This module is the thin layer between :mod:`ztbus.optim.model` (a NumPyro
generative program) and the user-facing CLI / SLURM jobs. It does three
things:

1. ``nuts_fit`` — wraps ``numpyro.infer.MCMC(NUTS(...))`` with sensible
   defaults and verifies the data dict shape before launching the chain.
2. ``diagnostics`` — computes R-hat, ESS, divergences, and emits a one-line
   summary suitable for log scanning.
3. ``posterior_to_dataframe`` — converts NumPyro's dict-of-arrays output to
   a flat polars DataFrame for plotting and parquet persistence.

Why not put this in model.py?
-----------------------------
``model.py`` is the *spec* of the inverse problem; ``samplers.py`` is one
*method* for solving it (NUTS specifically). Keeping them separate lets us
add ``emcee_fit`` and ``cmaes_fit`` later as drop-in alternatives without
touching the model.

Why NUTS for this problem?
--------------------------
The forward kernel is differentiable end-to-end (JAX autodiff), the
parameter space is continuous and 7-dimensional, and we want posteriors
not point estimates. That combination is the natural habitat for HMC, and
NUTS (Hoffman & Gelman 2014) is HMC with the trajectory length adapted
automatically. The default sampler in every modern probabilistic
programming framework.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpyro
import polars as pl
from numpyro.infer import MCMC, NUTS

from ztbus.optim.model import ztbus_model

if TYPE_CHECKING:
    from collections.abc import Mapping

    from jax import Array

logger = logging.getLogger(__name__)


@dataclass
class FitResult:
    """Output of a NUTS fit. All fields are JAX/numpy arrays plus diagnostics."""

    samples: dict[str, jnp.ndarray]
    """Posterior samples, one entry per latent (A, Cd, ..., sigma_W).
    Each array has shape (num_chains, num_samples) — *not* concatenated."""

    diagnostics: dict[str, float | int]
    """Summary scalars: r_hat_max, ess_bulk_min, num_divergent, wall_seconds."""

    config: dict[str, int | str | float]
    """Reproducibility metadata: chain count, warmup, sample count, RNG seed."""

    def log_summary(self) -> None:
        d = self.diagnostics
        logger.info(
            "Posterior: rhat_max=%.4f  ess_bulk_min=%.0f  divergent=%d  wall=%.1fs",
            d["r_hat_max"],
            d["ess_bulk_min"],
            d["num_divergent"],
            d["wall_seconds"],
        )
        if d["r_hat_max"] > 1.01:
            logger.warning(
                "R-hat = %.4f > 1.01 — chains have not converged; increase num_warmup",
                d["r_hat_max"],
            )
        if d["num_divergent"] > 0:
            logger.warning(
                "%d divergent transitions — model may have funnel/identifiability "
                "issues; inspect pair plots",
                d["num_divergent"],
            )


def nuts_fit(
    data: Mapping[str, Array],
    observed_power_W: Array,
    *,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 4,
    chain_method: str = "parallel",
    rng_seed: int = 0,
    target_accept_prob: float = 0.85,
    max_tree_depth: int = 10,
    progress_bar: bool = True,
    sample_weights: Array | None = None,
) -> FitResult:
    """Run NUTS on the ZTBus probabilistic model.

    Parameters
    ----------
    data
        Output of :func:`ztbus.optim.data.load_corpus` (the JAX dict), or
        any dict with the keys ``ztbus_model`` expects.
    observed_power_W
        Measured power in W. Same length as ``data["speed_mps"]`` etc.
        Typically ``data["P_obs_W"]`` itself.
    num_warmup, num_samples
        Standard MCMC controls. Warmup adapts the step size + mass matrix
        and is discarded; samples are kept.
    num_chains
        Number of independent chains. 4 is the convergence-diagnostics
        standard (we need at least 2 for R-hat).
    chain_method
        One of ``"parallel"`` (chains on different devices, requires
        ``num_chains`` ≤ device count), ``"sequential"`` (chains run one
        after another), or ``"vectorized"`` (chains as a single batched
        vmap call). On a single-GPU node, ``"parallel"`` requires
        ``num_chains ≤ jax.local_device_count()``; otherwise use
        ``"vectorized"``.
    rng_seed
        PRNG key for the run. Vary across array tasks for independence.
    target_accept_prob
        NUTS adapts the step size to hit this acceptance rate.
        Default 0.85 is conservative; raise to 0.95 if divergences appear.
    max_tree_depth
        NUTS doubles trajectory length up to 2^max_tree_depth leapfrog steps.
        Increase to 12 if trajectories regularly hit the limit (indicates
        very stiff posterior geometry).
    progress_bar
        Show MCMC progress bar (off in SLURM batch jobs).
    sample_weights
        Optional per-sample likelihood weights, passed through to the model.

    Returns
    -------
    FitResult
        Posterior samples + diagnostic scalars + config metadata.
    """
    # ---- Shape validation -------------------------------------------------
    n_obs = observed_power_W.shape[0]
    for key, arr in data.items():
        if arr.shape[0] != n_obs:
            raise ValueError(
                f"Shape mismatch: observed_power_W has {n_obs} samples but "
                f"data[{key!r}] has {arr.shape[0]}"
            )
    logger.info(
        "Starting NUTS: n_samples_obs=%d, chains=%d, warmup=%d, samples=%d",
        n_obs,
        num_chains,
        num_warmup,
        num_samples,
    )

    # ---- Build kernel and MCMC machine ------------------------------------
    kernel = NUTS(
        ztbus_model,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=progress_bar,
    )

    # ---- Run, timing the whole thing --------------------------------------
    rng_key = jax.random.PRNGKey(rng_seed)
    t0 = time.time()
    mcmc.run(
        rng_key,
        data=data,
        observed_power_W=observed_power_W,
        sample_weights=sample_weights,
    )
    wall_seconds = time.time() - t0

    # ---- Pull samples (group_by_chain=True keeps the chain dimension) -----
    samples = mcmc.get_samples(group_by_chain=True)

    # ---- Diagnostics ------------------------------------------------------
    diag = _compute_diagnostics(mcmc, samples, wall_seconds)

    result = FitResult(
        samples=samples,
        diagnostics=diag,
        config={
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "num_chains": num_chains,
            "chain_method": chain_method,
            "rng_seed": rng_seed,
            "target_accept_prob": target_accept_prob,
        },
    )
    result.log_summary()
    return result


def _compute_diagnostics(
    mcmc: MCMC,
    samples: dict[str, jnp.ndarray],
    wall_seconds: float,
) -> dict[str, float | int]:
    """Compute R-hat (worst across params), ESS bulk (worst), divergences."""
    summary = numpyro.diagnostics.summary(samples, group_by_chain=True)
    r_hat_max = max(float(summary[p]["r_hat"]) for p in summary)
    ess_bulk_min = min(float(summary[p]["n_eff"]) for p in summary)
    # mcmc.get_extra_fields() returns per-sample diagnostic arrays; count
    # divergent transitions across the whole run.
    extra = mcmc.get_extra_fields()
    num_divergent = int(extra["diverging"].sum()) if "diverging" in extra else 0
    return {
        "r_hat_max": r_hat_max,
        "ess_bulk_min": ess_bulk_min,
        "num_divergent": num_divergent,
        "wall_seconds": wall_seconds,
    }


# ---------------------------------------------------------------------------
# Posterior → DataFrame for plotting / persistence
# ---------------------------------------------------------------------------


def posterior_to_dataframe(samples: dict[str, jnp.ndarray]) -> pl.DataFrame:
    """Flatten NumPyro samples to a long-form polars DataFrame.

    Each row is one (chain, draw, parameter) triple — convenient for
    plotting libraries (seaborn, plotly express) and for parquet persistence.
    """
    rows: list[dict[str, float | int | str]] = []
    for param_name, arr in samples.items():
        # arr shape: (num_chains, num_samples)
        for chain_idx in range(arr.shape[0]):
            for draw_idx in range(arr.shape[1]):
                rows.append(
                    {
                        "parameter": param_name,
                        "chain": chain_idx,
                        "draw": draw_idx,
                        "value": float(arr[chain_idx, draw_idx]),
                    }
                )
    return pl.DataFrame(rows)


def posterior_summary(samples: dict[str, jnp.ndarray]) -> pl.DataFrame:
    """Compact one-row-per-parameter summary: mean, sd, 2.5/50/97.5 percentiles, R-hat."""
    summary = numpyro.diagnostics.summary(samples, group_by_chain=True)
    rows: list[dict[str, float | str]] = []
    for param_name, stats in summary.items():
        rows.append(
            {
                "parameter": param_name,
                "mean": float(stats["mean"]),
                "sd": float(stats["std"]),
                "ci_lo_95": float(stats["5.0%"]) if "5.0%" in stats else float(stats["2.5%"]),
                "median": float(stats["50.0%"]) if "50.0%" in stats else float(stats["median"]),
                "ci_hi_95": float(stats["95.0%"]) if "95.0%" in stats else float(stats["97.5%"]),
                "r_hat": float(stats["r_hat"]),
                "ess_bulk": float(stats["n_eff"]),
            }
        )
    return pl.DataFrame(rows)


__all__ = ["FitResult", "nuts_fit", "posterior_summary", "posterior_to_dataframe"]
