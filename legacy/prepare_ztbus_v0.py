from pathlib import Path
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


DATASET_DIR = Path(
    r"C:\Users\Joao Castro\Documents\Uni\Master\Semester 2\Optimisation for Computer Science\Week 8\ZTBus_samples"
)

OUTPUT_DIR = DATASET_DIR / "cleaned_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_ztbus_csv(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["time_iso"])
    df["time_iso"] = pd.to_datetime(df["time_iso"], utc=True, errors="coerce")
    df = df.dropna(subset=["time_iso"])
    df = df.sort_values("time_iso").set_index("time_iso")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove duplicate timestamps
    df = df[~df.index.duplicated(keep="first")]

    # Drop columns that are fully constant
    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if const_cols:
        df = df.drop(columns=const_cols)

    # Replace "-" with NaN in text columns
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].replace("-", np.nan)

    # Handle high-missing columns
    for col in df.columns:
        missing_ratio = df[col].isna().mean()

        if missing_ratio > 0.95:
            # Keep column, but only if it may still be useful later
            if col.startswith("itcs_"):
                df[col] = df[col].ffill().bfill()
        elif df[col].dtype.kind in "biufc":
            df[col] = df[col].interpolate(limit_direction="both")
        else:
            df[col] = df[col].ffill().bfill()

    # Binary columns
    binary_cols = [c for c in df.columns if c.startswith("status_")]
    for col in binary_cols:
        df[col] = df[col].fillna(0).astype(int)

    # Clip physically implausible values
    if "electric_powerDemand" in df.columns:
        df["electric_powerDemand"] = df["electric_powerDemand"].clip(-320000, 320000)

    speed_cols = [c for c in df.columns if "Speed" in c or "speed" in c.lower()]
    for col in speed_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    if "traction_tractionForce" in df.columns:
        df["traction_tractionForce"] = df["traction_tractionForce"].clip(-50000, 50000)

    if "traction_brakePressure" in df.columns:
        df["traction_brakePressure"] = df["traction_brakePressure"].clip(lower=0)

    if "temperature_ambient" in df.columns:
        df["temperature_ambient"] = df["temperature_ambient"].clip(250, 330)

    if "itcs_numberOfPassengers" in df.columns:
        df["itcs_numberOfPassengers"] = df["itcs_numberOfPassengers"].clip(lower=0, upper=200)

    # GNSS interpolation
    gnss_cols = [c for c in df.columns if c.startswith("gnss_")]
    for col in gnss_cols:
        if col in df.columns and df[col].dtype.kind in "biufc":
            df[col] = df[col].interpolate(limit_direction="both")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Time step in seconds
    if "time_unix" in df.columns:
        dt = df["time_unix"].diff().fillna(1)
        dt = dt.replace(0, 1)
    else:
        dt = pd.Series(1, index=df.index)

    df["dt_s"] = dt

    # Acceleration
    if "odometry_vehicleSpeed" in df.columns:
        df["accel_mps2"] = df["odometry_vehicleSpeed"].diff() / df["dt_s"]
        df["accel_mps2"] = df["accel_mps2"].fillna(0).clip(-3, 3)

    # Distance from speed
    if "odometry_vehicleSpeed" in df.columns:
        df["distance_step_m"] = df["odometry_vehicleSpeed"] * df["dt_s"]
        df["distance_cum_m"] = df["distance_step_m"].cumsum()

    # Grade estimate from GNSS altitude
    if "gnss_altitude" in df.columns and "distance_cum_m" in df.columns:
        delta_h = df["gnss_altitude"].diff().fillna(0)
        delta_s = df["distance_cum_m"].diff().replace(0, np.nan)
        df["road_grade"] = (delta_h / delta_s).replace([np.inf, -np.inf], np.nan)
        df["road_grade"] = df["road_grade"].fillna(0).clip(-0.2, 0.2)

    # Ambient temperature in Celsius
    if "temperature_ambient" in df.columns:
        df["temperature_ambient_C"] = df["temperature_ambient"] - 273.15

    # Passenger-based mass estimate
    curb_mass_kg = 19000
    avg_passenger_mass_kg = 80

    if "itcs_numberOfPassengers" in df.columns:
        passengers = df["itcs_numberOfPassengers"].fillna(0)
    else:
        passengers = 0

    df["estimated_vehicle_mass_kg"] = curb_mass_kg + passengers * avg_passenger_mass_kg

    # Instantaneous energy increments
    if "electric_powerDemand" in df.columns:
        df["energy_step_J"] = df["electric_powerDemand"] * df["dt_s"]
        df["energy_cum_J"] = df["energy_step_J"].cumsum()
        df["energy_cum_kWh"] = df["energy_cum_J"] / 3.6e6

    # Specific energy consumption
    if "distance_step_m" in df.columns and "energy_step_J" in df.columns:
        dist_km = df["distance_step_m"] / 1000
        energy_kWh = df["energy_step_J"] / 3.6e6
        df["specific_energy_kWh_per_km"] = (energy_kWh / dist_km.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )

    # HVAC proxy
    if "status_doorIsOpen" in df.columns and "temperature_ambient_C" in df.columns:
        cabin_setpoint_C = 20
        df["deltaT_abs_C"] = (cabin_setpoint_C - df["temperature_ambient_C"]).abs()
        df["hvac_proxy"] = df["deltaT_abs_C"] * (1 + df["status_doorIsOpen"])

    return df


def summarize_dataframe(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    summary = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing_count": [df[c].isna().sum() for c in df.columns],
        "missing_pct": [100 * df[c].isna().mean() for c in df.columns],
        "n_unique": [df[c].nunique(dropna=True) for c in df.columns],
    })
    summary.insert(0, "source_file", source_name)
    return summary


def process_one_file(csv_path: Path):
    print(f"Processing: {csv_path.name}")

    df_raw = load_ztbus_csv(csv_path)
    df_clean = clean_dataset(df_raw)
    df_final = engineer_features(df_clean)

    stem = csv_path.stem

    cleaned_file = OUTPUT_DIR / f"{stem}_cleaned.csv"
    summary_file = OUTPUT_DIR / f"{stem}_summary.csv"
    preview_file = OUTPUT_DIR / f"{stem}_preview.csv"

    df_final.to_csv(cleaned_file)
    summarize_dataframe(df_final, csv_path.name).to_csv(summary_file, index=False)
    df_final.head(1000).to_csv(preview_file)

    print(f"  saved: {cleaned_file.name}")
    print(f"  saved: {summary_file.name}")
    print(f"  saved: {preview_file.name}")


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Folder not found: {DATASET_DIR}")

    csv_files = sorted(DATASET_DIR.glob("B*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No bus CSV files found in: {DATASET_DIR}")

    print(f"Found {len(csv_files)} CSV file(s) in:")
    print(DATASET_DIR)
    print()

    for csv_file in csv_files:
        process_one_file(csv_file)

    print("\nDone.")
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()