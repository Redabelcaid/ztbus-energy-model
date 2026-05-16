"""Strong informative priors for identifiability — model.py patch.

Background
----------
The 4.14M-sample production run on V100 revealed two real identifiability
problems that single-month smokes had hidden:

1. Cd and A enter the model only as the product Cd*A. The likelihood is
   constant along the ridge Cd*A = const. With weak data the priors kept
   things in physical range; with strong data the joint posterior collapsed
   along the ridge until it hit both lower walls (A=7.0, Cd=0.50).

2. P_aux is not identifiable from driving samples alone. With ~in_depot
   excluding parked phases (where P_aux dominates the energy balance), P_aux
   becomes a small constant offset on driving samples and walks to its prior
   lower wall (0.1 kW).

This is a Bayesian-inference finding, not a bug. The right response is to
anchor the unidentifiable parameters at their externally-known values via
strong informative priors. We do NOT silently hardcode anything — every
parameter is still inferred by NUTS, and every prior choice is documented
in the model docstring and in the eventual methods section.

Changes
-------

A:    TruncatedNormal(8.4, 0.3, low=7.0, high=9.5)
  ->  TruncatedNormal(8.4, 0.1, low=8.0, high=8.8)
  Justification: Widmer 2023 Table 2 documents A = 8.4 m^2 for the HESS
  lighTram 19. Manufacturer geometry. ±0.1 m^2 accommodates panel paint /
  measurement uncertainty.

Cd:   Uniform(0.50, 1.10)
  ->  Uniform(0.30, 1.10)
  Justification: with A now anchored, Cd is independently identifiable. The
  data's preferred Cd*A ~ 3.5 m^2 implies Cd ~ 0.42 at A = 8.4. Old lower
  bound of 0.50 would peg; new lower bound of 0.30 gives NUTS room to land
  on the data-preferred value.

P_aux:  Uniform(0.1, 30.0) kW
   ->  TruncatedNormal(4.0, 1.0, low=1.0, high=10.0) kW
  Justification: Widmer reports 3-5 kW non-HVAC auxiliary draw for these
  buses. ADR 0002 (depot-samples fit for P_aux) deferred; pending that, an
  informative prior anchored at the literature value is the correct
  intermediate step.

Apply on the cluster:
    uv run python dump/scripts/apply_strong_priors.py
"""

from pathlib import Path

p = Path("src/ztbus/optim/model.py")
text = p.read_text()


def replace_or_die(old: str, new: str, label: str) -> None:
    """In-place string replacement that fails loud if the anchor is missing."""
    global text
    assert old in text, f"Anchor not found for {label!r} -- model.py may have changed"
    text = text.replace(old, new, 1)
    print(f"  ok  {label}")


# --- 1. Tighten A prior ---------------------------------------------------
replace_or_die(
    "_A_PRIOR_SD: float = 0.3",
    "_A_PRIOR_SD: float = 0.1",
    "A_PRIOR_SD 0.3 -> 0.1",
)
replace_or_die(
    "_A_LO: float = 7.0\n_A_HI: float = 9.5",
    "_A_LO: float = 8.0\n_A_HI: float = 8.8",
    "A bounds [7.0, 9.5] -> [8.0, 8.8]",
)

# --- 2. Widen Cd lower bound ----------------------------------------------
replace_or_die(
    "_CD_LO: float = 0.50",
    "_CD_LO: float = 0.30",
    "Cd lower bound 0.50 -> 0.30",
)

# --- 3. Switch P_aux from Uniform to TruncatedNormal ----------------------
#
# We replace the entire Auxiliary-power constants block to keep things
# coherent. Try the most-recent shape first; if not present, the file
# has been modified in unexpected ways and we should bail.
old_aux_block_v1 = """# Auxiliary power
_P_AUX_LO_KW: float = 0.1
_P_AUX_HI_KW: float = 30.0"""
old_aux_block_v2 = """# Auxiliary power
_P_AUX_LO_KW: float = 1.0
_P_AUX_HI_KW: float = 30.0"""

new_aux_block = """# Auxiliary power (informative prior — see module docstring for justification)
_P_AUX_PRIOR_MEAN_KW: float = 4.0
_P_AUX_PRIOR_SD_KW: float = 1.0
_P_AUX_LO_KW: float = 1.0
_P_AUX_HI_KW: float = 10.0"""

if old_aux_block_v1 in text:
    text = text.replace(old_aux_block_v1, new_aux_block, 1)
    print("  ok  P_aux constants (v0.8 form -> strong-prior form)")
elif old_aux_block_v2 in text:
    text = text.replace(old_aux_block_v2, new_aux_block, 1)
    print("  ok  P_aux constants (intermediate form -> strong-prior form)")
else:
    raise SystemExit("Could not find P_aux constants block")

# --- 4. Swap the P_aux sample line from Uniform to TruncatedNormal ---------
replace_or_die(
    'P_aux = numpyro.sample("P_aux", dist.Uniform(_P_AUX_LO_KW, _P_AUX_HI_KW))',
    'P_aux = numpyro.sample(\n        "P_aux",\n        dist.TruncatedNormal(\n            _P_AUX_PRIOR_MEAN_KW, _P_AUX_PRIOR_SD_KW,\n            low=_P_AUX_LO_KW, high=_P_AUX_HI_KW,\n        ),\n    )',
    "P_aux sample(Uniform) -> sample(TruncatedNormal)",
)

p.write_text(text)

# --- 5. Verify the patched file parses and contains the new values --------
final = p.read_text()
assert "_A_PRIOR_SD: float = 0.1" in final
assert "_A_LO: float = 8.0" in final
assert "_CD_LO: float = 0.30" in final
assert "_P_AUX_PRIOR_MEAN_KW: float = 4.0" in final
assert "TruncatedNormal(\n            _P_AUX_PRIOR_MEAN_KW" in final
print()
print("All changes applied. Verify with: git diff src/ztbus/optim/model.py")
print()
print("After verification, smoke-test on July 2021 full month (no subsample):")
print("    uv run python dump/scripts/smoke_one_month.py "
      "--year-month 2021-07 --subsample -1")
