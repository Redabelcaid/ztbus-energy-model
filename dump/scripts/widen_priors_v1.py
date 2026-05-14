"""Widen the two priors that were pegged at their upper bound in the first
smoke run (eta_recup at 0.85, Cd at 0.85). Apply this on the cluster:

    cd ~/ztbus-energy-model
    uv run python dump/scripts/widen_priors_v1.py

This script is a one-off patch — once applied, the changes live in model.py
forever and this file can be deleted. We keep it in dump/scripts/ as a
change record (what was widened, when, why).

Justification for the new bounds:

- eta_recup: posterior pegged at 0.85. Bounded above by 1.0 by physics
  (can't recover more than 100% of braking energy). Set new upper bound at
  0.95 to leave a safety margin away from the hard physical wall but allow
  the data to find higher values if it wants. Lower bound widened to 0.30
  (some trolley buses regen poorly when grid is unavailable + battery full).

- Cd: posterior pegged at 0.85. Bus aerodynamics with pantograph hardware
  can plausibly exceed Hjelkrem's 0.70 default. Set new upper bound at 1.10
  (a brick has Cd ~1.0; a bus with roof-mounted equipment can approach this).
"""

from pathlib import Path

MODEL_PY = Path("src/ztbus/optim/model.py")
text = MODEL_PY.read_text()

# Cd: 0.50–0.85  →  0.50–1.10
text = text.replace(
    "_CD_LO: float = 0.50\n_CD_HI: float = 0.85",
    "_CD_LO: float = 0.50\n_CD_HI: float = 1.10",
)

# eta_recup: 0.40–0.85  →  0.30–0.95
text = text.replace(
    "_ETA_RECUP_LO: float = 0.40\n_ETA_RECUP_HI: float = 0.85",
    "_ETA_RECUP_LO: float = 0.30\n_ETA_RECUP_HI: float = 0.95",
)

MODEL_PY.write_text(text)

# Verify the substitutions actually landed (defensive — fails loud if not)
new_text = MODEL_PY.read_text()
assert "_CD_HI: float = 1.10" in new_text, "Cd patch failed"
assert "_ETA_RECUP_LO: float = 0.30" in new_text, "eta_recup LO patch failed"
assert "_ETA_RECUP_HI: float = 0.95" in new_text, "eta_recup HI patch failed"
print("✓ Cd:        [0.50, 0.85]  →  [0.50, 1.10]")
print("✓ eta_recup: [0.40, 0.85]  →  [0.30, 0.95]")
print("\nPatched. Verify with: git diff src/ztbus/optim/model.py")
