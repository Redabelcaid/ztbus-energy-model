# ZTBus energy model — convenience targets.
# Run `make help` to see what's available.

.DEFAULT_GOAL := help
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

UV ?= uv
PYTHON_VERSION ?= 3.11

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
.PHONY: install
install:  ## Install the environment with all extras (uses uv)
	$(UV) sync --all-extras
	$(UV) run pre-commit install

.PHONY: lock
lock:  ## Refresh the uv lockfile (deliberate dependency bumps only)
	$(UV) lock --upgrade

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------
.PHONY: lint
lint:  ## Lint with ruff
	$(UV) run ruff check src tests scripts

.PHONY: format
format:  ## Auto-format with ruff
	$(UV) run ruff format src tests scripts
	$(UV) run ruff check --fix src tests scripts

.PHONY: typecheck
typecheck:  ## Static type-check with mypy
	$(UV) run mypy src

.PHONY: test
test:  ## Run the fast test suite
	$(UV) run pytest -q -m "not slow and not hpc"

.PHONY: test-all
test-all:  ## Run all tests including slow ones
	$(UV) run pytest -q

.PHONY: check
check: lint typecheck test  ## Run all quality gates locally

# ---------------------------------------------------------------------------
# Pipeline (local; for HPC use the slurm/*.sbatch scripts or `snakemake --executor slurm`)
# ---------------------------------------------------------------------------
.PHONY: ingest
ingest:  ## Convert raw ZTBus CSVs to mission-partitioned parquet
	$(UV) run ztbus ingest

.PHONY: clean-data
clean-data:  ## Run the cleaning pipeline on all interim missions
	$(UV) run ztbus clean

.PHONY: pipeline
pipeline:  ## Run the full Snakemake pipeline locally
	$(UV) run snakemake --cores 8 --use-conda=false

.PHONY: pipeline-hpc
pipeline-hpc:  ## Run the full pipeline on SLURM (configure configs/hpc/slurm.yaml first)
	$(UV) run snakemake --executor slurm --jobs 100 --workflow-profile configs/hpc/

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
.PHONY: clean-cache
clean-cache:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} +

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
