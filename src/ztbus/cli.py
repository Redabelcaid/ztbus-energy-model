"""Command-line interface: ``ztbus ingest``, ``ztbus clean``, ``ztbus fit``, ...

Subcommands are thin wrappers around functions in the package; the heavy
lifting lives in the modules so that everything is also callable from
notebooks and from Snakemake rules without going through the CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console

app = typer.Typer(
    name="ztbus",
    help="ZTBus energy-model identification CLI.",
    no_args_is_help=True,
    add_completion=False,
)

_console = Console()


def _configure_logging(level: str) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> - {message}",
    )


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------
@app.command()
def ingest(
    raw_dir: Path = typer.Option(
        None,
        "--raw-dir",
        envvar="ZTBUS_RAW_DIR",
        help="Directory containing the original ZTBus B*.csv files.",
    ),
    interim_dir: Path = typer.Option(
        Path("data/interim"),
        "--interim-dir",
        envvar="ZTBUS_INTERIM_DIR",
        help="Output directory for partitioned parquet.",
    ),
    max_workers: int = typer.Option(
        1,
        "--max-workers",
        help="Number of parallel workers for local ingest. On HPC use slurm/ingest.sbatch.",
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Re-ingest even if parquet exists."),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Convert raw ZTBus CSVs to mission-partitioned Parquet."""
    _configure_logging(log_level)

    if raw_dir is None:
        typer.echo("ERROR: --raw-dir not given and ZTBUS_RAW_DIR not set.", err=True)
        raise typer.Exit(2)

    # Defer the heavy import so `--help` is fast
    from ztbus.io import (
        discover_missions,
    )

    missions = discover_missions(raw_dir)
    logger.info("Ingesting {} missions → {}", len(missions), interim_dir)

    if max_workers <= 1:
        for path in missions:
            _ingest_one(path, interim_dir, overwrite)
        logger.success("Ingest complete: {} missions", len(missions))
        return

    from concurrent.futures import ProcessPoolExecutor, as_completed
    from functools import partial

    fn = partial(_ingest_one, interim_dir=interim_dir, overwrite=overwrite)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn, p): p for p in missions}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:  # pragma: no cover
                logger.error("Failed: {} ({})", futures[fut].name, exc)
    logger.success("Ingest complete: {} missions", len(missions))


def _ingest_one(path: Path, interim_dir: Path, overwrite: bool) -> Path:
    """Process a single mission file (top-level for ProcessPoolExecutor pickleability)."""
    from ztbus.io import parse_mission_filename, read_mission_csv, write_mission_parquet

    meta = parse_mission_filename(path)
    df = read_mission_csv(path)
    return write_mission_parquet(
        df,
        root=interim_dir,
        bus=meta["bus"],
        start_utc=meta["start_utc"],
        mission_id=meta["mission_id"],
        overwrite=overwrite,
    )


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------
@app.command()
def clean(
    interim_dir: Path = typer.Option(Path("data/interim"), envvar="ZTBUS_INTERIM_DIR"),
    processed_dir: Path = typer.Option(Path("data/processed"), envvar="ZTBUS_PROCESSED_DIR"),
    config: Path = typer.Option(Path("configs/cleaning/v1.yaml"), help="Cleaning policy YAML"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Apply the cleaning + feature pipeline to all interim missions."""
    _configure_logging(log_level)

    import polars as pl

    from ztbus.cleaning import CleaningConfig, clean_mission
    from ztbus.cleaning.grade import derive_grade
    from ztbus.features import add_energy, add_kinematics, add_mass
    from ztbus.qc import run_gates
    from ztbus.routes import detect_depot_phases

    cfg = CleaningConfig.from_yaml(config)
    files = sorted(interim_dir.rglob("*.parquet"))
    if not files:
        logger.error("No interim parquet files under {}", interim_dir)
        raise typer.Exit(1)

    qc_rows: list[dict] = []
    for f in files:
        try:
            bus = int(next(part.split("=")[1] for part in f.parts if part.startswith("bus=")))
        except StopIteration:
            bus = -1
        out_path = processed_dir / f.relative_to(interim_dir)
        if out_path.exists() and not overwrite:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)

        df = pl.read_parquet(f)
        df, qc = clean_mission(df, mission_id=f.stem, bus=bus, cfg=cfg)
        if qc.rejected:
            qc_rows.append(qc.as_dict())
            continue

        # Feature stage: depends on cleaned signals
        df = add_kinematics(df)
        df = derive_grade(df, cfg.grade)
        df = add_mass(df)
        df = add_energy(df)
        df, depot = detect_depot_phases(df)

        # Run QC gates
        gates = run_gates(df)
        gate_dict = {f"gate_{r.name}": r.passed for r in gates}
        gate_values = {f"gate_{r.name}_value": r.value for r in gates}

        df.write_parquet(out_path, compression="zstd", compression_level=3, statistics=True)
        qc_rows.append(
            {
                **qc.as_dict(),
                **gate_dict,
                **gate_values,
                "depot_n_start": depot.n_rows_start_depot,
                "depot_n_end": depot.n_rows_end_depot,
                "depot_at_start": depot.detected_depot_at_start,
                "depot_at_end": depot.detected_depot_at_end,
            }
        )

    qc_df = pl.DataFrame(qc_rows)
    qc_out = processed_dir / "_qc_summary.parquet"
    qc_df.write_parquet(qc_out)
    logger.success("Cleaning complete: {} missions → {}", len(qc_rows), processed_dir)
    logger.info("QC summary: {}", qc_out)


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------
@app.command()
def profile(
    interim_dir: Path = typer.Option(Path("data/interim"), envvar="ZTBUS_INTERIM_DIR"),
    out: Path = typer.Option(Path("data/reports/dataset_profile.parquet")),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Profile every interim mission: per-mission stats for EDA."""
    _configure_logging(log_level)
    from ztbus.eda import profile_corpus

    profile_corpus(interim_dir, out_path=out)


# ---------------------------------------------------------------------------
# fit (placeholder; implemented in phase 5)
# ---------------------------------------------------------------------------
@app.command()
def fit(
    processed_dir: Path = typer.Option(Path("data/processed"), envvar="ZTBUS_PROCESSED_DIR"),
    physics_config: Path = typer.Option(Path("configs/physics/hess_lightram_19.yaml")),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Identify powertrain parameters from cleaned data. Phase 5 — stub."""
    _configure_logging(log_level)
    logger.warning("fit: not yet implemented — see ROADMAP in README.md")
    raise typer.Exit(0)


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------
@app.command()
def version() -> None:
    """Print package version."""
    from ztbus import __version__

    _console.print(f"ztbus-energy-model [bold]{__version__}[/bold]")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
