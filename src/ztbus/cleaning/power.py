"""Power signal cleaning.

Per the ZTBus paper, the motors deliver up to ±320 kW and auxiliaries add
~20-30 kW, so values beyond about ±450 kW indicate CAN-bus spikes rather than
real operation. We FLAG these, never silently clip — the cleaned signal
still has them, and the parameter-fit objective downweights flagged samples.

Negative power is preserved: it represents either regenerative braking back
into the battery or, on these trolley buses, energy returned to the overhead
line during grid-available phases.
"""

from __future__ import annotations

import polars as pl

from ztbus.cleaning.config import PowerConfig

RAW_COL = "electric_powerDemand"


def clean_power(df: pl.DataFrame, cfg: PowerConfig) -> pl.DataFrame:
    """Flag implausible power values; do NOT modify the signal itself."""
    if RAW_COL not in df.columns:
        return df

    df = df.with_columns(
        (
            (pl.col(RAW_COL) < cfg.hard_lower_W)
            | (pl.col(RAW_COL) > cfg.hard_upper_W)
        ).alias("power_outlier_flag"),
    )
    return df
