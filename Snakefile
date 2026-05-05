# Snakemake pipeline for the ZTBus energy-model project.
#
# Local run:
#   uv run snakemake --cores 8
#
# HPC (SLURM) run:
#   uv run snakemake --executor slurm --jobs 100 --workflow-profile configs/hpc/

import os
from pathlib import Path

configfile: "configs/default.yaml"

RAW_DIR       = Path(os.environ.get("ZTBUS_RAW_DIR",       "data/raw"))
INTERIM_DIR   = Path(os.environ.get("ZTBUS_INTERIM_DIR",   "data/interim"))
PROCESSED_DIR = Path(os.environ.get("ZTBUS_PROCESSED_DIR", "data/processed"))
REPORTS_DIR   = Path(os.environ.get("ZTBUS_REPORTS_DIR",   "data/reports"))

MISSIONS = sorted(p.stem for p in RAW_DIR.glob("B*.csv"))

# ---------------------------------------------------------------------------
# Top-level target
# ---------------------------------------------------------------------------
rule all:
    input:
        REPORTS_DIR / "ingest_summary.parquet",
        REPORTS_DIR / "dataset_profile.parquet",
        PROCESSED_DIR / "_qc_summary.parquet",
        # Phase 5 will append:  REPORTS_DIR / "fitted_parameters.parquet",

# ---------------------------------------------------------------------------
# Phase 1: ingest (CSV → Parquet, mission-partitioned)
# ---------------------------------------------------------------------------
rule ingest_one:
    input:
        csv = RAW_DIR / "{mission}.csv",
    output:
        marker = INTERIM_DIR / ".markers" / "{mission}.done",
    log:
        "logs/ingest/{mission}.log",
    resources:
        mem_mb = 4000,
        runtime = 30,
    threads: 2
    shell:
        """
        mkdir -p $(dirname {output.marker})
        uv run python -m scripts.ingest_one --csv {input.csv} --interim-dir {INTERIM_DIR} \
            && touch {output.marker} 2> {log}
        """

rule ingest_summary:
    input:
        expand(INTERIM_DIR / ".markers" / "{mission}.done", mission=MISSIONS),
    output:
        REPORTS_DIR / "ingest_summary.parquet",
    log:
        "logs/ingest_summary.log",
    threads: 1
    shell:
        """
        mkdir -p $(dirname {output})
        uv run python -m scripts.ingest_summary --interim-dir {INTERIM_DIR} --out {output} 2> {log}
        """

# ---------------------------------------------------------------------------
# Phase 1.5: dataset-level EDA profile (one row per mission)
# ---------------------------------------------------------------------------
rule dataset_profile:
    input:
        REPORTS_DIR / "ingest_summary.parquet",
    output:
        REPORTS_DIR / "dataset_profile.parquet",
    log:
        "logs/dataset_profile.log",
    threads: 4
    resources:
        mem_mb = 8000,
        runtime = 30,
    shell:
        """
        uv run ztbus profile --interim-dir {INTERIM_DIR} --out {output} 2> {log}
        """

# ---------------------------------------------------------------------------
# Phase 2: cleaning + features (per mission)
# ---------------------------------------------------------------------------
rule clean_all:
    input:
        REPORTS_DIR / "ingest_summary.parquet",
    output:
        PROCESSED_DIR / "_qc_summary.parquet",
    log:
        "logs/clean_all.log",
    threads: 4
    resources:
        mem_mb = 8000,
        runtime = 120,
    shell:
        """
        uv run ztbus clean \
            --interim-dir {INTERIM_DIR} \
            --processed-dir {PROCESSED_DIR} \
            --config configs/cleaning/v1.yaml 2> {log}
        """

# Phase 5 (parameter identification) rules will be added here once the
# `ztbus.optim` module lands.
