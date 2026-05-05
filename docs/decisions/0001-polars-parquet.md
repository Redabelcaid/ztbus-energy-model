# ADR 0001 — Use Polars + Parquet instead of pandas + CSV

## Status

Accepted, 2026-05-05.

## Context

The ZTBus dataset is ~10 GB of CSV across 1409 mission files, stored as
schema-stable text. The previous semester's `prepare_ztbus.py` used pandas
with `read_csv`, in-memory transforms, and CSV outputs. On a developer
laptop a single pass over the corpus took on the order of 30 minutes and
peaked above available RAM for some missions.

## Decision

Replace pandas with Polars for I/O and transforms; replace CSV outputs with
Parquet (zstd, mission-partitioned).

## Consequences

Positive:

- Read+write throughput improves by an order of magnitude.
- Memory peaks fit comfortably on laptop and on a single cluster CPU.
- Parquet is ~10× smaller on disk and supports predicate pushdown.
- Schema is enforced at read time (`schema_overrides`), surfacing data
  corruption immediately.

Negative:

- Polars expression API is unfamiliar to people coming from pandas; learning
  curve of a few hours.
- A few pandas-only libraries do not accept Polars frames; conversions to
  pandas via `df.to_pandas()` remain available where strictly needed.

## Alternatives considered

- **Dask DataFrame**: distributed but heavy; we don't need it because each
  mission is independent and fits in memory.
- **DuckDB-only**: fine for SQL-style queries but we want imperative
  transforms in Python; we use DuckDB for ad-hoc analysis only.
- **pyspark**: rejected — overkill at 10 GB and operationally heavy on HPC.
