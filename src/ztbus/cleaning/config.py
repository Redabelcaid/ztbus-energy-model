"""Typed configuration for the cleaning pipeline.

The YAML in ``configs/cleaning/*.yaml`` is loaded into these models so that:

* every cleaning step gets a typed, validated config (no string typos);
* defaults are documented in code, not buried in YAML;
* swapping cleaning policies (v1, v2, ...) is a config change, not a code change.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class TimestampConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reject_on_duplicate_timestamps: bool = False
    reject_on_non_monotonic_time: bool = True
    max_internal_gap_s: float = 10.0


class SmoothingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: str = "rolling_median_then_mean"
    window_seconds: float = 3.0
    use_smoothed_for_derivative: bool = True


class SpeedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    small_negative_threshold_mps: float = -0.2
    hard_negative_action: str = "flag"
    upper_plausibility_mps: float = 25.0
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)


class PowerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hard_lower_W: float = -450_000.0
    hard_upper_W: float = 500_000.0
    on_violation: str = "flag"
    preserve_negative_for_regen: bool = True


class AccelerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plausibility_bounds_mps2: tuple[float, float] = (-3.0, 3.0)
    on_violation: str = "flag"


class AltitudeSmoothingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: str = "rolling_median"
    window_seconds: float = 30.0


class AltitudeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = "gnss_altitude"
    short_gap_max_s: float = 10.0
    long_gap_action: str = "leave_nan"
    smoothing: AltitudeSmoothingConfig = Field(default_factory=AltitudeSmoothingConfig)


class GradeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    derivation: str = "dh_over_ds_from_smoothed"
    smooth_after_derive: bool = True
    smoothing_window_seconds: float = 30.0
    plausibility_bounds: tuple[float, float] = (-0.12, 0.12)
    on_violation: str = "flag"


class TemperatureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit_in_dataset: str = "K"
    store_as: str = "K"
    short_gap_max_s: float = 60.0
    plausibility_bounds_K: tuple[float, float] = (243.0, 323.0)


class PassengersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = "itcs_numberOfPassengers"
    na_handling: str = "forward_fill_then_zero"
    hard_lower: int = 0
    hard_upper: int = 200


class BinarySignalsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resampling: str = "previous_neighbor"
    na_handling: str = "leave_nan"


class CourseHandlingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bus_183: str = "hold_last_value_when_stationary"
    bus_208: str = "zero_when_stationary"


class GNSSConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    short_gap_max_s: float = 5.0
    long_gap_action: str = "leave_nan"
    course_handling: CourseHandlingConfig = Field(default_factory=CourseHandlingConfig)
    use_for_grade: bool = True


class DepotDetectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    method: str = "stationary_at_known_depot_polygon"


class DerivedSignalsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    energy_integration_method: str = "trapezoid"
    distance_integration_method: str = "trapezoid"
    cumulative_quantities: list[str] = Field(default_factory=lambda: ["energy_kWh", "distance_m"])


class CleaningConfig(BaseModel):
    """Top-level cleaning policy."""

    model_config = ConfigDict(extra="forbid")

    policy_version: str = "v1"
    timestamps: TimestampConfig = Field(default_factory=TimestampConfig)
    speed: SpeedConfig = Field(default_factory=SpeedConfig)
    power: PowerConfig = Field(default_factory=PowerConfig)
    acceleration: AccelerationConfig = Field(default_factory=AccelerationConfig)
    altitude: AltitudeConfig = Field(default_factory=AltitudeConfig)
    grade: GradeConfig = Field(default_factory=GradeConfig)
    temperature: TemperatureConfig = Field(default_factory=TemperatureConfig)
    passengers: PassengersConfig = Field(default_factory=PassengersConfig)
    binary_signals: BinarySignalsConfig = Field(default_factory=BinarySignalsConfig)
    gnss_coordinates: GNSSConfig = Field(default_factory=GNSSConfig)
    depot_detection: DepotDetectionConfig = Field(default_factory=DepotDetectionConfig)
    derived_signals: DerivedSignalsConfig = Field(default_factory=DerivedSignalsConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> CleaningConfig:
        cfg = OmegaConf.to_container(OmegaConf.load(Path(path)), resolve=True)
        # YAML uses `gnss_coordinates`; ensure pydantic gets the dict
        return cls.model_validate(cfg)
