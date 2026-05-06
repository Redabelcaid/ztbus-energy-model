# `dump/` — colleague workspace

This is a deliberate sandbox. Drop whatever helps you explore: notebooks,
one-off scripts, data snippets, screenshots, scratch notes. No quality bar.

## Suggested subdirs (use whichever fits, ignore the rest)

- `notebooks/` — Jupyter notebooks, exploratory analysis
- `scripts/`   — one-off Python/bash scripts that aren't pipeline-grade yet
- `data/`      — small intermediate files; do NOT commit anything > a few MB
- `figures/`   — screenshots, plots, diagrams
- `notes/`     — markdown notes, meeting takeaways, anything written

## Ground rules

1. **Anything here is unstable by definition.** Nothing in `src/`, `tests/`,
   `configs/`, or `scripts/` may import from this folder.
2. **No large data files.** Use `/scratch/users/$USER/ztbus/` for those and
   reference the path here. Git is for code and small artifacts.
3. **If something here graduates** into proper pipeline code, port it
   explicitly into `src/ztbus/` with a real test, then delete it from `dump/`.
4. **Date-prefix your files** when convenient (e.g. `2026-05-06_eda_routes.ipynb`)
   so the folder doesn't become a random pile.
5. **The cleaning policy in `configs/cleaning/v1.yaml` and the physics
   constants in `configs/physics/` are the contract.** If you want to try
   different thresholds, copy them to `configs/local/` (gitignored) or write
   a script here that reads them — don't edit the committed configs without
   discussion.

## Examples of things that belong here

- "I tried clustering by time-of-day and got these plots"
- "Quick sanity check on bus 208's December 2022 missions"
- "Notes from the supervisor meeting on 2026-05-13"
- "Random script to grep through stop names for 'Hauptbahnhof'"

## Examples of things that do NOT belong here

- Modifications to `src/ztbus/` modules — those go through the normal flow
- Any file that another script in `src/` or `scripts/` needs to import
- Large datasets — use scratch
