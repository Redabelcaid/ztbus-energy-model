"""Phase 5 — Bayesian parameter identification.

Public API:

  PARAM_NAMES         names of the 7 parameters, in canonical order
  NUM_PARAMS          7
  forward             eager JAX forward model (P_total in watts)
  forward_jit         JIT-compiled forward model — use this in tight loops
  forward_vmap        vmapped over candidate parameter vectors
  forward_vmap_jit    vmap + jit composed
"""

from ztbus.optim.kernels import (
    NUM_PARAMS,
    PARAM_NAMES,
    forward,
    forward_jit,
    forward_vmap,
    forward_vmap_jit,
)

__all__ = [
    "NUM_PARAMS",
    "PARAM_NAMES",
    "forward",
    "forward_jit",
    "forward_vmap",
    "forward_vmap_jit",
]
