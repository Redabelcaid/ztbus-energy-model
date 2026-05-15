import argparse
import multiprocessing as mp
import os

import numpy as np
import pandas as pd


TIME_COL = "time_iso"
SPEED_COL = "odometry_vehicleSpeed"
POWER_COL = "electric_powerDemand"
ALTITUDE_COL = "gnss_altitude"
TEMP_COL = "temperature_ambient"
PASSENGER_COL = "itcs_numberOfPassengers"
TRIP_GAP_THRESHOLD_SEC = 5 * 60

CURB_MASS_KG = 19000.0
PASSENGER_MASS_KG = 68.0
DEFAULT_PASSENGERS = 0.0

PARAMETER_SPECS = [
    {"name": "Crr", "default": 0.010, "lb": 0.006, "ub": 0.020},
    {"name": "Cd", "default": 0.700, "lb": 0.600, "ub": 0.800},
    {"name": "P_aux_kW", "default": 2.000, "lb": 2.000, "ub": 7.000},
    {"name": "eta_bus", "default": 0.820, "lb": 0.630, "ub": 0.900},
    {"name": "eta_battery", "default": 0.900, "lb": 0.630, "ub": 0.900},
    {"name": "eta_recup", "default": 0.820, "lb": 0.640, "ub": 0.820},
]

A_FRONT = 8.4
RHO_AIR = 1.2
G = 9.81
MIN_RECUP_SPEED_KMH = 15.0
GRADE_CLIP = 0.08


class TeeLogger:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._file = None

    def __enter__(self):
        self._file = open(self.file_path, "w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._file is not None:
            self._file.close()

    def print(self, *args, **kwargs) -> None:
        print(*args, **kwargs)
        if self._file is not None:
            print(*args, **kwargs, file=self._file)
            self._file.flush()


def list_input_files(input_path: str) -> list[str]:
    if os.path.isdir(input_path):
        cleaned_files = sorted(
            os.path.join(root, name)
            for root, _, files in os.walk(input_path)
            for name in files
            if name.lower().endswith("_cleaned.csv")
        )
        csv_files = cleaned_files or sorted(
            os.path.join(root, name)
            for root, _, files in os.walk(input_path)
            for name in files
            if (
                name.lower().endswith(".csv")
                and not name.lower().endswith("_summary.csv")
                and not name.lower().endswith("_preview.csv")
            )
        )
        parquet_files = sorted(
            os.path.join(root, name)
            for root, _, files in os.walk(input_path)
            for name in files
            if name.lower().endswith(".parquet")
        )
        input_files = cleaned_files or csv_files or parquet_files
    else:
        input_files = [input_path]

    if not input_files:
        raise FileNotFoundError(f"No CSV or parquet files found in {input_path!r}")

    return input_files


def read_input_file(file_path: str) -> pd.DataFrame:
    extension = os.path.splitext(file_path)[1].lower()
    if extension == ".csv":
        return pd.read_csv(file_path)
    if extension == ".parquet":
        return pd.read_parquet(file_path)
    raise ValueError(f"Unsupported input file type: {file_path}")


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def numeric_column(
    df: pd.DataFrame, candidates: list[str], default: float | None = None
) -> pd.Series:
    column = first_existing_column(df, candidates)
    if column is None:
        if default is None:
            raise ValueError(f"Missing one of required columns: {candidates}")
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def prepare_model_df(raw_df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    df = raw_df.copy()
    df["source_file"] = source_file

    if "time_s" in df.columns:
        df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    elif TIME_COL in df.columns:
        df["time"] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
        df["time_s"] = (df["time"] - df["time"].min()).dt.total_seconds()
    elif "time_unix" in df.columns:
        df["time_s"] = pd.to_numeric(df["time_unix"], errors="coerce")
        df["time_s"] = df["time_s"] - df["time_s"].min()
    elif "time" in df.columns:
        numeric_time = pd.to_numeric(df["time"], errors="coerce")
        if numeric_time.notna().sum() >= len(df) * 0.5:
            df["time_s"] = numeric_time - numeric_time.min()
        else:
            parsed_time = pd.to_datetime(df["time"], errors="coerce", utc=True)
            df["time_s"] = (parsed_time - parsed_time.min()).dt.total_seconds()
    elif isinstance(df.index, pd.DatetimeIndex):
        parsed_time = pd.to_datetime(df.index, errors="coerce", utc=True)
        df["time_s"] = (parsed_time - parsed_time.min()).total_seconds()
    else:
        raise ValueError("Missing time column. Expected time_s, time_iso, or time_unix.")

    df = df.dropna(subset=["time_s"]).sort_values("time_s").reset_index(drop=True)
    df["dt"] = df["time_s"].diff()
    df["dt"] = df["dt"].where(df["dt"] > 0.0, np.nan)

    if "trip_id" not in df.columns:
        df["time_gap"] = df["time_s"].diff()
        df["trip_number"] = (df["time_gap"] > TRIP_GAP_THRESHOLD_SEC).cumsum().astype(int)
        df["trip_id"] = df["source_file"].astype(str) + "_trip_" + df["trip_number"].astype(str)

    df["time_s"] = df.groupby("trip_id")["time_s"].transform(lambda x: x - x.min())
    df["dt"] = df.groupby("trip_id")["time_s"].diff()
    df["speed_m_per_s"] = numeric_column(df, ["speed_m_per_s", SPEED_COL])

    if first_existing_column(df, ["acceleration_m_s2", "accel_mps2"]) is not None:
        df["acceleration_m_s2"] = numeric_column(df, ["acceleration_m_s2", "accel_mps2"])
    else:
        df["acceleration_m_s2"] = df.groupby("trip_id")["speed_m_per_s"].diff() / df["dt"]

    if first_existing_column(df, ["distance_m", "distance_cum_m"]) is not None:
        df["distance_m"] = numeric_column(df, ["distance_m", "distance_cum_m"])
    else:
        df["distance_step_m"] = df["speed_m_per_s"] * df["dt"].fillna(0.0)
        df["distance_m"] = df.groupby("trip_id")["distance_step_m"].cumsum()

    df["elevation_m"] = numeric_column(df, ["elevation_m", ALTITUDE_COL], default=0.0)

    if first_existing_column(df, ["temperature_C", "temperature_ambient_C"]) is not None:
        df["temperature_C"] = numeric_column(df, ["temperature_C", "temperature_ambient_C"])
    elif TEMP_COL in df.columns:
        temp = pd.to_numeric(df[TEMP_COL], errors="coerce")
        df["temperature_C"] = np.where(temp > 100.0, temp - 273.15, temp)
    else:
        df["temperature_C"] = 21.0

    if "mass_kg" in df.columns:
        df["mass_kg"] = pd.to_numeric(df["mass_kg"], errors="coerce")
    elif "estimated_vehicle_mass_kg" in df.columns:
        df["mass_kg"] = pd.to_numeric(df["estimated_vehicle_mass_kg"], errors="coerce")
    else:
        passengers = numeric_column(df, [PASSENGER_COL, "passengers"], default=DEFAULT_PASSENGERS)
        df["mass_kg"] = CURB_MASS_KG + PASSENGER_MASS_KG * passengers.fillna(DEFAULT_PASSENGERS)

    if "power_kW" in df.columns:
        df["power_kW"] = pd.to_numeric(df["power_kW"], errors="coerce")
    else:
        df["power_kW"] = numeric_column(df, [POWER_COL, "power_measured_W"]) / 1000.0

    keep = [
        "trip_id",
        "time_s",
        "dt",
        "distance_m",
        "speed_m_per_s",
        "acceleration_m_s2",
        "elevation_m",
        "temperature_C",
        "mass_kg",
        "power_kW",
        "source_file",
    ]
    df = df[[column for column in keep if column in df.columns]].copy()
    df = df.dropna(
        subset=[
            "trip_id",
            "time_s",
            "distance_m",
            "speed_m_per_s",
            "acceleration_m_s2",
            "elevation_m",
            "mass_kg",
            "power_kW",
        ]
    )
    df = df[df["dt"].fillna(0.0) > 0.0].copy()
    return df.sort_values(["trip_id", "time_s"]).reset_index(drop=True)


def bounds_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.array([spec["lb"] for spec in PARAMETER_SPECS], dtype=float)
    upper = np.array([spec["ub"] for spec in PARAMETER_SPECS], dtype=float)
    default = np.array([spec["default"] for spec in PARAMETER_SPECS], dtype=float)
    return lower, upper, default


def theta_to_params(theta: np.ndarray) -> dict[str, float]:
    params = {spec["name"]: float(theta[index]) for index, spec in enumerate(PARAMETER_SPECS)}
    params["A_front"] = A_FRONT
    params["rho_air"] = RHO_AIR
    params["g"] = G
    params["min_recup_speed_kmh"] = MIN_RECUP_SPEED_KMH
    return params


def parameter_table(theta: np.ndarray) -> pd.DataFrame:
    rows = []
    for index, spec in enumerate(PARAMETER_SPECS):
        rows.append(
            {
                "parameter": spec["name"],
                "default": spec["default"],
                "lower_bound": spec["lb"],
                "upper_bound": spec["ub"],
                "best_estimate": float(theta[index]),
            }
        )
    return pd.DataFrame(rows)


def finite_array(values: pd.Series | np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    series = pd.Series(array).replace([np.inf, -np.inf], np.nan)
    series = series.interpolate(limit_direction="both").fillna(fill_value)
    return series.to_numpy(dtype=float)


def infer_dt(time_s: np.ndarray) -> np.ndarray:
    t = finite_array(time_s)
    if len(t) == 0:
        return np.array([], dtype=float)
    if len(t) == 1:
        return np.array([1.0], dtype=float)
    dt = np.diff(t, prepend=t[0])
    dt[0] = dt[1] if len(dt) > 1 else 1.0
    dt = np.where(np.isfinite(dt), dt, 1.0)
    dt = np.where(dt > 0.0, dt, 1.0)
    return np.clip(dt, 1e-3, 10.0)


def gradient_1d(values: np.ndarray) -> np.ndarray:
    x = finite_array(values)
    grad = np.zeros_like(x)
    n = len(x)
    if n <= 1:
        return grad
    grad[0] = x[1] - x[0]
    grad[-1] = x[-1] - x[-2]
    if n > 2:
        grad[1:-1] = (x[2:] - x[:-2]) / 2.0
    return grad


def predict_arrays(df: pd.DataFrame, theta: np.ndarray) -> dict[str, np.ndarray]:
    params = theta_to_params(theta)
    v = finite_array(df["speed_m_per_s"])
    a = finite_array(df["acceleration_m_s2"])
    distance = finite_array(df["distance_m"])
    elevation = finite_array(df["elevation_m"])
    mass = finite_array(df["mass_kg"], fill_value=CURB_MASS_KG)
    time_s = finite_array(df["time_s"])
    dt = infer_dt(time_s)

    d_dist = gradient_1d(distance)
    d_elev = gradient_1d(elevation)
    grade = np.zeros_like(distance)
    valid = np.abs(d_dist) > 1e-6
    grade[valid] = d_elev[valid] / d_dist[valid]
    grade = np.clip(np.where(np.isfinite(grade), grade, 0.0), -GRADE_CLIP, GRADE_CLIP)
    slope_angle = np.arctan(grade)

    F_roll = mass * params["g"] * params["Crr"] * np.cos(slope_angle)
    F_aero = 0.5 * params["rho_air"] * v**2 * params["A_front"] * params["Cd"]
    F_inertia = mass * a
    F_grade = mass * params["g"] * np.sin(slope_angle)
    F_total = F_roll + F_aero + F_inertia + F_grade

    P_ldm = v * F_total
    P_prop = np.where(P_ldm >= 0.0, P_ldm / max(params["eta_bus"], 1e-6), P_ldm)
    min_recup_speed = params["min_recup_speed_kmh"] / 3.6
    recup_mask = (P_prop < 0.0) & (v >= min_recup_speed)
    low_speed_mask = (P_prop < 0.0) & (v < min_recup_speed)
    P_prop = np.where(recup_mask, P_prop * params["eta_recup"], P_prop)
    P_prop = np.where(low_speed_mask, 0.0, P_prop)

    P_aux = np.full(len(df), params["P_aux_kW"] * 1000.0, dtype=float)
    P_model = P_prop + P_aux / max(params["eta_battery"], 1e-6)

    return {
        "dt": dt,
        "power_model_W": P_model,
        "power_measured_W": finite_array(df["power_kW"]) * 1000.0,
    }


def file_loss(
    theta: np.ndarray,
    file_path: str,
    input_root: str,
    power_weight: float,
    energy_weight: float,
    verbose: bool = False,
) -> tuple[float, int]:
    source_file = (
        os.path.relpath(file_path, input_root)
        if os.path.isdir(input_root)
        else os.path.basename(file_path)
    )
    try:
        raw = read_input_file(file_path)
        df = prepare_model_df(raw, source_file)
    except Exception as exc:
        if verbose:
            print(f"Skipping {source_file}: {exc}", flush=True)
        return 0.0, 0
    if df.empty:
        return 0.0, 0

    weighted_loss_sum = 0.0
    n_trips = 0
    for _, trip in df.groupby("trip_id", sort=False):
        arrays = predict_arrays(trip, theta)
        y = arrays["power_measured_W"]
        yhat = arrays["power_model_W"]
        dt = arrays["dt"]
        mse_power_w = float(np.mean((yhat - y) ** 2))
        measured_energy = float(np.sum((y / 1000.0) * dt / 3600.0))
        predicted_energy = float(np.sum((yhat / 1000.0) * dt / 3600.0))
        energy_rel = ((predicted_energy - measured_energy) / max(abs(measured_energy), 1e-6)) ** 2
        loss = power_weight * mse_power_w + energy_weight * energy_rel * 1e8
        if np.isfinite(loss):
            weighted_loss_sum += loss
            n_trips += 1

    return weighted_loss_sum, n_trips


def month_group_key(file_path: str, input_root: str) -> str:
    rel_path = (
        os.path.relpath(file_path, input_root)
        if os.path.isdir(input_root)
        else os.path.basename(file_path)
    )
    parts = rel_path.split(os.sep)
    for index, part in enumerate(parts):
        if part.startswith("month="):
            return os.path.join(*parts[: index + 1])
    return os.path.dirname(rel_path) or "."


def make_balanced_month_chunks(
    file_paths: list[str], input_root: str, workers: int
) -> list[list[str]]:
    grouped: dict[str, list[str]] = {}
    for file_path in file_paths:
        grouped.setdefault(month_group_key(file_path, input_root), []).append(file_path)

    n_chunks = max(1, min(int(workers), len(grouped)))
    chunks: list[list[str]] = [[] for _ in range(n_chunks)]
    chunk_sizes = [0 for _ in range(n_chunks)]

    for _, group_files in sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True):
        target = int(np.argmin(chunk_sizes))
        chunks[target].extend(group_files)
        chunk_sizes[target] += len(group_files)

    return [chunk for chunk in chunks if chunk]


def chunk_loss(args: tuple[np.ndarray, list[str], str, float, float]) -> tuple[float, int]:
    theta, file_chunk, input_root, power_weight, energy_weight = args
    total_loss = 0.0
    total_count = 0
    for file_path in file_chunk:
        loss_sum, count = file_loss(
            theta, file_path, input_root, power_weight, energy_weight, verbose=False
        )
        total_loss += loss_sum
        total_count += count
    return total_loss, total_count


def objective_from_theta(
    theta: np.ndarray,
    file_paths: list[str],
    input_root: str,
    power_weight: float,
    energy_weight: float,
    pool=None,
    file_chunks: list[list[str]] | None = None,
) -> float:
    total_loss = 0.0
    total_count = 0
    if pool is None or file_chunks is None:
        for file_path in file_paths:
            loss_sum, count = file_loss(
                theta, file_path, input_root, power_weight, energy_weight, verbose=True
            )
            total_loss += loss_sum
            total_count += count
    else:
        work_items = [
            (theta, file_chunk, input_root, power_weight, energy_weight)
            for file_chunk in file_chunks
        ]
        for loss_sum, count in pool.map(chunk_loss, work_items):
            total_loss += loss_sum
            total_count += count
    if total_count == 0:
        return 1e30
    return float(total_loss / total_count)


def bounded_cmaes_optimize_streaming(
    file_paths: list[str],
    input_root: str,
    iterations: int,
    population_size: int | None,
    sigma: float,
    seed: int,
    power_weight: float,
    energy_weight: float,
    workers: int,
    logger: TeeLogger | None,
) -> tuple[np.ndarray, float, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    lower, upper, default = bounds_arrays()
    span = upper - lower
    dim = len(default)

    if population_size is None:
        population_size = 4 + int(3 * np.log(dim))
    population_size = max(4, int(population_size))
    mu = population_size // 2

    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mu_eff = 1.0 / np.sum(weights**2)

    c_sigma = (mu_eff + 2.0) / (dim + mu_eff + 5.0)
    d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (dim + 1.0)) - 1.0) + c_sigma
    c_c = (4.0 + mu_eff / dim) / (dim + 4.0 + 2.0 * mu_eff / dim)
    c1 = 2.0 / ((dim + 1.3) ** 2 + mu_eff)
    c_mu = min(
        1.0 - c1,
        2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((dim + 2.0) ** 2 + mu_eff),
    )
    chi_n = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim**2))

    mean_scaled = (default - lower) / span
    sigma_scaled = float(sigma)
    covariance = np.eye(dim)
    p_sigma = np.zeros(dim)
    p_c = np.zeros(dim)

    workers = max(1, int(workers))
    file_chunks = make_balanced_month_chunks(file_paths, input_root, workers)
    pool = mp.Pool(processes=workers) if workers > 1 else None

    best_theta = np.clip(default.copy(), lower, upper)
    history = []
    log_print = logger.print if logger is not None else print
    log_print(
        f"Using {workers} worker process(es) across {len(file_chunks)} balanced month/file chunk(s)."
    )

    try:
        best_loss = objective_from_theta(
            best_theta,
            file_paths,
            input_root,
            power_weight,
            energy_weight,
            pool=pool,
            file_chunks=file_chunks if pool is not None else None,
        )

        for iteration in range(1, iterations + 1):
            covariance = (covariance + covariance.T) / 2.0
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            sqrt_covariance = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
            inv_sqrt_covariance = (
                eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
            )

            z_samples = rng.normal(size=(population_size, dim))
            y_samples = z_samples @ sqrt_covariance.T
            x_scaled = mean_scaled + sigma_scaled * y_samples
            x_scaled = np.clip(x_scaled, 0.0, 1.0)
            theta_samples = lower + x_scaled * span
            losses = np.array(
                [
                    objective_from_theta(
                        theta,
                        file_paths,
                        input_root,
                        power_weight,
                        energy_weight,
                        pool=pool,
                        file_chunks=file_chunks if pool is not None else None,
                    )
                    for theta in theta_samples
                ],
                dtype=float,
            )

            order = np.argsort(losses)
            theta_samples = theta_samples[order]
            x_scaled = x_scaled[order]
            y_samples = y_samples[order]
            losses = losses[order]

            if losses[0] < best_loss:
                best_loss = float(losses[0])
                best_theta = theta_samples[0].copy()

            old_mean_scaled = mean_scaled.copy()
            selected_x = x_scaled[:mu]
            mean_scaled = np.sum(selected_x * weights[:, None], axis=0)
            y_w = (mean_scaled - old_mean_scaled) / max(sigma_scaled, 1e-12)

            p_sigma = (1.0 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * (
                inv_sqrt_covariance @ y_w
            )
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma_threshold = (1.4 + 2.0 / (dim + 1.0)) * chi_n
            h_sigma = float(
                norm_p_sigma / np.sqrt(1.0 - (1.0 - c_sigma) ** (2.0 * iteration))
                < h_sigma_threshold
            )

            p_c = (1.0 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2.0 - c_c) * mu_eff) * y_w
            rank_mu = np.zeros((dim, dim))
            for weight, y in zip(weights, y_samples[:mu]):
                rank_mu += weight * np.outer(y, y)

            covariance = (
                (1.0 - c1 - c_mu) * covariance
                + c1 * (np.outer(p_c, p_c) + (1.0 - h_sigma) * c_c * (2.0 - c_c) * covariance)
                + c_mu * rank_mu
            )
            sigma_scaled *= np.exp((c_sigma / d_sigma) * (norm_p_sigma / chi_n - 1.0))
            sigma_scaled = float(np.clip(sigma_scaled, 1e-4, 2.0))

            row = {"iteration": iteration, "best_objective": best_loss, "sigma": sigma_scaled}
            row.update(theta_to_params(best_theta))
            history.append(row)

            if iteration == 1 or iteration % 10 == 0 or iteration == iterations:
                params = theta_to_params(best_theta)
                log_print(
                    f"Iteration {iteration:4d} | "
                    f"objective={best_loss:.6f} | "
                    f"Crr={params['Crr']:.6f} | "
                    f"Cd={params['Cd']:.6f} | "
                    f"P_aux_kW={params['P_aux_kW']:.6f} | "
                    f"eta_bus={params['eta_bus']:.6f} | "
                    f"eta_battery={params['eta_battery']:.6f} | "
                    f"eta_recup={params['eta_recup']:.6f}"
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return best_theta, best_loss, pd.DataFrame(history)


def evaluate_files(theta: np.ndarray, file_paths: list[str], input_root: str) -> pd.DataFrame:
    rows = []
    for file_path in file_paths:
        source_file = (
            os.path.relpath(file_path, input_root)
            if os.path.isdir(input_root)
            else os.path.basename(file_path)
        )
        try:
            raw = read_input_file(file_path)
            df = prepare_model_df(raw, source_file)
        except Exception as exc:
            rows.append(
                {
                    "trip_id": "",
                    "source_file": source_file,
                    "n_samples": 0,
                    "measured_energy_kWh_from_power": np.nan,
                    "predicted_energy_kWh_from_power": np.nan,
                    "trip_energy_error_kWh": np.nan,
                    "trip_energy_error_pct": np.nan,
                    "sample_mse_kW2": np.nan,
                    "sample_rmse_kW": np.nan,
                    "status": f"skipped: {exc}",
                }
            )
            continue
        if df.empty:
            continue

        for trip_id, trip in df.groupby("trip_id", sort=False):
            arrays = predict_arrays(trip, theta)
            y = arrays["power_measured_W"]
            yhat = arrays["power_model_W"]
            dt = arrays["dt"]
            measured_energy = float(np.sum((y / 1000.0) * dt / 3600.0))
            predicted_energy = float(np.sum((yhat / 1000.0) * dt / 3600.0))
            mse_kW2 = float(np.mean(((yhat - y) / 1000.0) ** 2))
            rows.append(
                {
                    "trip_id": trip_id,
                    "source_file": source_file,
                    "n_samples": len(trip),
                    "measured_energy_kWh_from_power": measured_energy,
                    "predicted_energy_kWh_from_power": predicted_energy,
                    "trip_energy_error_kWh": predicted_energy - measured_energy,
                    "trip_energy_error_pct": 100.0
                    * (predicted_energy - measured_energy)
                    / max(abs(measured_energy), 1e-6),
                    "sample_mse_kW2": mse_kW2,
                    "sample_rmse_kW": float(np.sqrt(mse_kW2)),
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit one global bus energy parameter set with parallel streaming CMA-ES."
    )
    parser.add_argument(
        "data_path", help="Top-level CSV/parquet dataset directory or one input file."
    )
    parser.add_argument("--iterations", type=int, default=120, help="Number of CMA-ES iterations.")
    parser.add_argument(
        "--population-size", type=int, default=8, help="Number of candidates per iteration."
    )
    parser.add_argument(
        "--sigma", type=float, default=0.25, help="Initial CMA-ES step size in normalized bounds."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--power-weight", type=float, default=0.65, help="Weight for sample power MSE in watts."
    )
    parser.add_argument(
        "--energy-weight", type=float, default=0.35, help="Weight for trip energy relative error."
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Worker processes for parallel month/file chunks."
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("output", "cmaes_streaming_parallel"),
        help="Output directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with TeeLogger(os.path.join(args.output_dir, "printed_output.txt")) as logger:
        file_paths = list_input_files(args.data_path)
        logger.print(f"Discovered {len(file_paths)} input file(s).")
        logger.print(
            "Parallel streaming objective is enabled: each worker loads its assigned files one at a time."
        )

        best_theta, best_objective, convergence_df = bounded_cmaes_optimize_streaming(
            file_paths=file_paths,
            input_root=args.data_path,
            iterations=args.iterations,
            population_size=args.population_size,
            sigma=args.sigma,
            seed=args.seed,
            power_weight=args.power_weight,
            energy_weight=args.energy_weight,
            workers=args.workers,
            logger=logger,
        )

        params = theta_to_params(best_theta)
        logger.print("\nEstimated global parameters:")
        for spec in PARAMETER_SPECS:
            logger.print(f"{spec['name']}: {params[spec['name']]:.6f}")
        logger.print(f"\nFinal CMA-ES objective: {best_objective:.6f}")

        metrics_df = evaluate_files(best_theta, file_paths, args.data_path)
        logger.print("\nEvaluation summary:")
        logger.print(metrics_df.describe(include="all"))

        parameter_table(best_theta).to_csv(
            os.path.join(args.output_dir, "cmaes_global_best_parameters.csv"), index=False
        )
        convergence_df.to_csv(
            os.path.join(args.output_dir, "cmaes_global_convergence.csv"), index=False
        )
        metrics_df.to_csv(
            os.path.join(args.output_dir, "cmaes_global_trip_metrics.csv"), index=False
        )
