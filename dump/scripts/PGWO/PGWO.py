from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

DATASET_DIR = Path("data")
OUT_DIR = Path("output_PGWO")
OUT_DIR.mkdir(exist_ok=True, parents=True)
BUS_IDS = ("183", "208")

PARAMETER_SPECS = [
    {"name": "Crr", "label": "Rolling resistance coefficient", "unit": "-", "default": 0.010, "lb": 0.006, "ub": 0.020},
    {"name": "Cd", "label": "Aerodynamic drag coefficient", "unit": "-", "default": 0.700, "lb": 0.600, "ub": 0.800},
    {"name": "P_aux_kW", "label": "Auxiliary power", "unit": "kW", "default": 2.000, "lb": 2.000, "ub": 7.000},
    {"name": "eta_bus", "label": "Bus propulsion efficiency", "unit": "-", "default": 0.820, "lb": 0.630, "ub": 0.900},
    {"name": "eta_battery", "label": "Battery efficiency", "unit": "-", "default": 0.900, "lb": 0.630, "ub": 0.900},
    {"name": "eta_recup", "label": "Recuperation efficiency", "unit": "-", "default": 0.820, "lb": 0.640, "ub": 0.820},
]

A_FRONT = 8.4
RHO_AIR = 1.2
G = 9.81
MIN_RECUP_SPEED_KMH = 15.0
PASSENGER_MASS_KG = 68.0
BASE_MASS_KG = 19000.0
MAX_PASSENGERS = 160.0
MASS_MIN_KG = BASE_MASS_KG
MASS_MAX_KG = BASE_MASS_KG + MAX_PASSENGERS * PASSENGER_MASS_KG

MEDIAN_WINDOW = 5
SMOOTH_WINDOW = 15
SPEED_GRADE_WINDOW = 31
DISTANCE_MIN_STEP_M = 1.0
POWER_CLIP_W = 350000.0
GRADE_CLIP = 0.08
ACCEL_CLIP = 2.5

PLOT_MAX_POINTS = 2200
PLOT_QUANTILE_LOW = 0.01
PLOT_QUANTILE_HIGH = 0.99
PLOT_RESIDUAL_KW_LIMIT = 200.0
PLOT_FORCE_LIMIT_N = 25000.0
PLOT_GRADE_LIMIT_PCT = 8.0
PLOT_POWER_LIMIT_KW = 300.0
GAUSSIAN_SIGMA = 22.0
TREND_DEGREE = 3

RAW_ALPHA = 0.10
RAW_LINEWIDTH = 0.55
SMOOTH_LINEWIDTH = 1.6
REG_LINEWIDTH = 3.0


def bounds_arrays():
    lb = np.array([spec['lb'] for spec in PARAMETER_SPECS], dtype=float)
    ub = np.array([spec['ub'] for spec in PARAMETER_SPECS], dtype=float)
    x0 = np.array([spec['default'] for spec in PARAMETER_SPECS], dtype=float)
    return lb, ub, x0


def theta_to_params(theta):
    params = {spec['name']: float(theta[i]) for i, spec in enumerate(PARAMETER_SPECS)}
    params['A_front'] = A_FRONT
    params['rho_air'] = RHO_AIR
    params['g'] = G
    params['min_recup_speed_kmh'] = MIN_RECUP_SPEED_KMH
    return params


def infer_dt(time_s: np.ndarray) -> np.ndarray:
    t = np.asarray(time_s, dtype=float)
    if len(t) == 0:
        return np.array([], dtype=float)
    if len(t) == 1:
        return np.array([1.0], dtype=float)
    dt = np.diff(t, prepend=t[0])
    dt[0] = dt[1] if len(dt) > 1 else 1.0
    dt = np.where(~np.isfinite(dt), 1.0, dt)
    dt = np.where(dt <= 0, 1.0, dt)
    return np.clip(dt, 1e-3, 10.0)


def centered_rolling_mean(arr, window):
    s = pd.Series(np.asarray(arr, dtype=float))
    return s.rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=float)


def centered_median(arr, window):
    s = pd.Series(np.asarray(arr, dtype=float))
    return s.rolling(window=window, center=True, min_periods=1).median().to_numpy(dtype=float)


def robust_clip_series(arr, z=4.0):
    arr = np.asarray(arr, dtype=float)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    med = np.nanmedian(arr)
    mad = np.nanmedian(np.abs(arr - med))
    if not np.isfinite(med) or not np.isfinite(mad) or mad < 1e-9:
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = 1.4826 * mad
    lo = med - z * sigma
    hi = med + z * sigma
    arr = np.clip(arr, lo, hi)
    return np.nan_to_num(arr, nan=med, posinf=hi, neginf=lo)


def finite_series(x, fill_value=0.0):
    s = pd.Series(np.asarray(x, dtype=float))
    s = s.replace([np.inf, -np.inf], np.nan)
    s = s.interpolate(limit_direction='both').fillna(fill_value)
    return s.to_numpy(dtype=float)


def safe_gradient(values, coords=None):
    v = finite_series(values)
    n = len(v)
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([0.0], dtype=float)
    if coords is None:
        c = np.arange(n, dtype=float)
    else:
        c = finite_series(coords)
    grad = np.zeros(n, dtype=float)
    for i in range(n):
        if i == 0:
            dx = c[1] - c[0]
            grad[i] = 0.0 if abs(dx) < 1e-9 else (v[1] - v[0]) / dx
        elif i == n - 1:
            dx = c[-1] - c[-2]
            grad[i] = 0.0 if abs(dx) < 1e-9 else (v[-1] - v[-2]) / dx
        else:
            dx = c[i + 1] - c[i - 1]
            grad[i] = 0.0 if abs(dx) < 1e-9 else (v[i + 1] - v[i - 1]) / dx
    return np.where(np.isfinite(grad), grad, 0.0)


def compute_grade(distance_m, elevation_m):
    d = finite_series(distance_m)
    h = finite_series(elevation_m)
    d_dist = safe_gradient(d)
    d_elev = safe_gradient(h)
    safe_dist = np.where(np.abs(d_dist) < DISTANCE_MIN_STEP_M, np.nan, d_dist)
    grade = d_elev / safe_dist
    grade = pd.Series(grade).replace([np.inf, -np.inf], np.nan).interpolate(limit_direction='both').fillna(0.0).to_numpy(dtype=float)
    grade = centered_median(grade, MEDIAN_WINDOW)
    grade = centered_rolling_mean(grade, SPEED_GRADE_WINDOW)
    grade = np.clip(grade, -GRADE_CLIP, GRADE_CLIP)
    alpha = np.arctan(grade)
    return grade, alpha


def clean_plot_df(df, required_cols):
    out = df.copy()
    for c in required_cols:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=required_cols)
    return out


def plot_filter_series(series, q_low=PLOT_QUANTILE_LOW, q_high=PLOT_QUANTILE_HIGH, hard_clip=None):
    s = pd.Series(pd.to_numeric(series, errors='coerce')).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return s
    lo = s.quantile(q_low)
    hi = s.quantile(q_high)
    s = s.clip(lower=lo, upper=hi)
    if hard_clip is not None:
        s = s.clip(lower=-abs(hard_clip), upper=abs(hard_clip))
    return s


def make_plot_sample(df, max_points=PLOT_MAX_POINTS):
    if df is None or len(df) == 0:
        return df
    if len(df) <= max_points:
        return df.copy().reset_index(drop=True)
    idx = np.linspace(0, len(df) - 1, max_points, dtype=int)
    return df.iloc[idx].copy().reset_index(drop=True)


def extract_dataset_month(labels):
    s = pd.Series(labels, dtype='object').astype(str)
    date_token = s.str.extract(r'(\d{4}-\d{2}-\d{2})', expand=False)
    dt = pd.to_datetime(date_token, errors='coerce')
    month = dt.dt.to_period('M').astype(str)
    fallback = s.str.extract(r'(\d{4}-\d{2})', expand=False)
    month = month.where(~month.isna(), fallback)
    return month.fillna('unknown')


def gaussian_kernel1d(sigma, radius=None):
    sigma = max(float(sigma), 1e-6)
    if radius is None:
        radius = max(3, int(4 * sigma + 0.5))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def gaussian_smooth(y, sigma=GAUSSIAN_SIGMA):
    y = finite_series(y)
    if len(y) <= 3:
        return y.copy()
    kernel = gaussian_kernel1d(sigma)
    pad = len(kernel) // 2
    padded = np.pad(y, pad_width=pad, mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed[:len(y)]


def polynomial_trend(x, y, degree=TREND_DEGREE):
    x = finite_series(x)
    y = finite_series(y)
    if len(x) == 0:
        return np.array([], dtype=float)
    deg = int(min(max(1, degree), max(1, len(x) - 1)))
    if np.allclose(x, x[0]):
        return np.full(len(x), np.nanmean(y), dtype=float)
    coeffs = np.polyfit(x, y, deg=deg)
    return np.polyval(coeffs, x)


def first_present_column(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_raw_dataset(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        'time_s': ['time_unix', 'time_s', 'timestamp_unix', 'timestamp'],
        'speed_m_per_s': ['odometry_vehicleSpeed', 'speed_m_per_s', 'vehicle_speed_mps', 'speed_mps'],
        'acceleration_m_s2': ['accel_mps2', 'acceleration_m_s2', 'vehicle_acceleration_mps2'],
        'elevation_m': ['gnss_altitude', 'elevation_m', 'altitude_m'],
        'distance_m': ['distance_cum_m', 'distance_m', 'odometry_distance_m'],
        'temperature_C': ['temperature_ambient_C', 'temperature_C', 'ambient_temperature_C'],
        'door_open': ['status_doorIsOpen', 'door_open', 'door_is_open'],
        'power_measured_W': ['electric_powerDemand', 'power_measured_W', 'power_demand_W', 'traction_power_W'],
        'mass_kg': ['estimated_vehicle_mass_kg', 'mass_kg'],
        'passengers': ['itcs_numberOfPassengers', 'number_of_passengers', 'passengers'],
    }
    out = pd.DataFrame(index=df.index)
    required = ['speed_m_per_s', 'elevation_m', 'power_measured_W']
    for target in required:
        source = first_present_column(df, aliases[target])
        if source is None:
            return pd.DataFrame()
        out[target] = pd.to_numeric(df[source], errors='coerce')

    time_source = first_present_column(df, aliases['time_s'])
    if time_source is None:
        out['time_s'] = np.arange(len(df), dtype=float)
    else:
        out['time_s'] = pd.to_numeric(df[time_source], errors='coerce')

    dist_source = first_present_column(df, aliases['distance_m'])
    if dist_source is None:
        dt = infer_dt(finite_series(out['time_s']))
        out['distance_m'] = np.cumsum(np.maximum(finite_series(out['speed_m_per_s']), 0.0) * dt)
    else:
        out['distance_m'] = pd.to_numeric(df[dist_source], errors='coerce')

    accel_source = first_present_column(df, aliases['acceleration_m_s2'])
    if accel_source is None:
        out['acceleration_m_s2'] = safe_gradient(finite_series(out['speed_m_per_s']), finite_series(out['time_s']))
    else:
        out['acceleration_m_s2'] = pd.to_numeric(df[accel_source], errors='coerce')

    temp_source = first_present_column(df, aliases['temperature_C'])
    out['temperature_C'] = pd.to_numeric(df[temp_source], errors='coerce') if temp_source else 15.0

    door_source = first_present_column(df, aliases['door_open'])
    out['door_open'] = pd.to_numeric(df[door_source], errors='coerce') if door_source else 0.0

    mass_source = first_present_column(df, aliases['mass_kg'])
    if mass_source is not None:
        out['mass_kg'] = pd.to_numeric(df[mass_source], errors='coerce')
    else:
        pax_source = first_present_column(df, aliases['passengers'])
        if pax_source is not None:
            passengers = pd.to_numeric(df[pax_source], errors='coerce').fillna(0).clip(0, MAX_PASSENGERS)
            out['mass_kg'] = BASE_MASS_KG + PASSENGER_MASS_KG * passengers
        else:
            out['mass_kg'] = BASE_MASS_KG + PASSENGER_MASS_KG * 10.0

    return out


def preprocess_dataset(use: pd.DataFrame) -> pd.DataFrame:
    use = use.replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(use) < 100:
        return pd.DataFrame()
    use = use.sort_values('time_s').reset_index(drop=True)
    use['time_s'] = pd.to_numeric(use['time_s'], errors='coerce')
    use['time_s'] = use['time_s'] - use['time_s'].iloc[0]
    use['time_s'] = finite_series(use['time_s'])
    use['mass_kg'] = pd.to_numeric(use['mass_kg'], errors='coerce').clip(MASS_MIN_KG, MASS_MAX_KG)
    use['speed_m_per_s'] = centered_median(use['speed_m_per_s'], MEDIAN_WINDOW)
    use['speed_m_per_s'] = centered_rolling_mean(use['speed_m_per_s'], SMOOTH_WINDOW)
    use['speed_m_per_s'] = np.clip(finite_series(use['speed_m_per_s']), 0.0, 35.0)
    use['power_measured_W'] = robust_clip_series(use['power_measured_W'])
    use['power_measured_W'] = centered_rolling_mean(use['power_measured_W'], SMOOTH_WINDOW)
    use['power_measured_W'] = np.clip(finite_series(use['power_measured_W']), -POWER_CLIP_W, POWER_CLIP_W)
    use['temperature_C'] = centered_rolling_mean(use['temperature_C'], SMOOTH_WINDOW)
    use['temperature_C'] = finite_series(use['temperature_C'])
    use['door_open'] = centered_median(use['door_open'], MEDIAN_WINDOW)
    use['door_open'] = (finite_series(use['door_open']) >= 0.5).astype(float)
    use['distance_m'] = centered_rolling_mean(use['distance_m'], SMOOTH_WINDOW)
    use['distance_m'] = finite_series(use['distance_m'])
    use['distance_m'] = np.maximum.accumulate(use['distance_m'])
    use['elevation_m'] = centered_rolling_mean(use['elevation_m'], SPEED_GRADE_WINDOW)
    use['elevation_m'] = finite_series(use['elevation_m'])
    speed = use['speed_m_per_s'].to_numpy(dtype=float)
    accel = safe_gradient(speed, use['time_s'].to_numpy(dtype=float))
    accel = centered_rolling_mean(accel, SMOOTH_WINDOW)
    accel = np.clip(finite_series(accel), -ACCEL_CLIP, ACCEL_CLIP)
    use['acceleration_m_s2'] = accel
    grade, alpha = compute_grade(use['distance_m'].to_numpy(dtype=float), use['elevation_m'].to_numpy(dtype=float))
    use['grade'] = grade
    use['slope_angle_rad'] = alpha
    return use.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def load_bus_dataset(bus_id: str, base_dir: Path):
    bus_folder = base_dir / f"bus={bus_id}"
    files = sorted(bus_folder.rglob("*.parquet"))
    datasets = []
    for f in files:
        try:
            raw = pd.read_parquet(f)
        except Exception:
            continue
        use = normalize_raw_dataset(raw)
        if len(use) == 0:
            continue
        use = preprocess_dataset(use)
        if len(use) >= 100:
            datasets.append((f"{bus_id}_{f.stem}", use))
    return datasets


def vehicle_power_profile(dft: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = dft.copy()
    v = finite_series(df['speed_m_per_s'])
    a = finite_series(df['acceleration_m_s2'])
    alpha = finite_series(df['slope_angle_rad'])
    m = finite_series(df['mass_kg'], fill_value=BASE_MASS_KG)
    t = finite_series(df['time_s'])
    dt = infer_dt(t)
    F_roll = m * params['g'] * params['Crr'] * np.cos(alpha)
    F_aero = 0.5 * params['rho_air'] * v**2 * params['A_front'] * params['Cd']
    F_inertia = m * a
    F_grade = m * params['g'] * np.sin(alpha)
    F_total = F_roll + F_aero + F_inertia + F_grade
    P_ldm = v * F_total
    P_after_bus = np.where(P_ldm >= 0.0, P_ldm / max(params['eta_bus'], 1e-6), P_ldm)
    min_recup_speed = params['min_recup_speed_kmh'] / 3.6
    recup_mask = (P_after_bus < 0.0) & (v >= min_recup_speed)
    low_speed_mask = (P_after_bus < 0.0) & (v < min_recup_speed)
    P_after_bus = np.where(recup_mask, P_after_bus * params['eta_recup'], P_after_bus)
    P_after_bus = np.where(low_speed_mask, 0.0, P_after_bus)
    P_aux = np.full(len(df), params['P_aux_kW'] * 1000.0, dtype=float)
    P_total = P_after_bus + P_aux / max(params['eta_battery'], 1e-6)
    measured_power = finite_series(df['power_measured_W'])
    energy_step_model_kWh = (P_total / 1000.0) * dt / 3600.0
    energy_step_measured_kWh = (measured_power / 1000.0) * dt / 3600.0
    df['F_roll_N'] = F_roll
    df['F_aero_N'] = F_aero
    df['F_inertia_N'] = F_inertia
    df['F_grade_N'] = F_grade
    df['F_total_N'] = F_total
    df['P_ldm_W'] = P_ldm
    df['P_prop_W'] = P_after_bus
    df['P_aux_W'] = P_aux
    df['P_model_W'] = P_total
    df['power_measured_W'] = measured_power
    df['power_error_W'] = df['P_model_W'] - df['power_measured_W']
    df['energy_step_model_kWh'] = energy_step_model_kWh
    df['energy_step_measured_kWh'] = energy_step_measured_kWh
    df['energy_cum_model_kWh'] = np.cumsum(energy_step_model_kWh)
    df['energy_cum_measured_kWh'] = np.cumsum(energy_step_measured_kWh)
    return df


def objective_from_params(theta, datasets):
    params = theta_to_params(theta)
    losses = []
    for _, df in datasets:
        pred = vehicle_power_profile(df, params)
        y = finite_series(pred['power_measured_W'])
        yhat = finite_series(pred['P_model_W'])
        e_meas = float(pred['energy_cum_measured_kWh'].iloc[-1]) if len(pred) else 0.0
        e_pred = float(pred['energy_cum_model_kWh'].iloc[-1]) if len(pred) else 0.0
        mse_power = np.mean((yhat - y) ** 2)
        energy_rel = ((e_pred - e_meas) / max(abs(e_meas), 1e-6)) ** 2
        loss = 0.65 * mse_power + 0.35 * (energy_rel * 1e8)
        if np.isfinite(loss):
            losses.append(loss)
    if not losses:
        return 1e20
    return float(np.mean(losses))


def pgwo_optimize(datasets, n_wolves=24, n_iter=80, seed=42, progress_callback=None, progress_every=5):
    rng = np.random.default_rng(seed)
    lb, ub, x0 = bounds_arrays()
    dim = len(lb)
    wolves = rng.uniform(lb, ub, size=(n_wolves, dim))
    wolves[0] = x0.copy()
    span = np.maximum(ub - lb, 1e-12)
    velocities = rng.uniform(-0.10 * span, 0.10 * span, size=(n_wolves, dim))
    fitness = np.array([objective_from_params(w, datasets) for w in wolves], dtype=float)
    pbest = wolves.copy()
    pbest_fitness = fitness.copy()
    history = []
    for it in range(n_iter):
        order = np.argsort(fitness)
        wolves = wolves[order]
        fitness = fitness[order]
        alpha, beta, delta = wolves[0].copy(), wolves[1].copy(), wolves[2].copy()
        history.append((it, float(fitness[0])))
        a = 2.0 - 2.0 * it / max(n_iter - 1, 1)
        inertia = 0.90 - 0.60 * it / max(n_iter - 1, 1)
        c1 = 1.50
        c2 = 1.50
        c3 = 0.75
        new_wolves = wolves.copy()
        for i in range(n_wolves):
            X = wolves[i].copy()
            r1 = rng.random(dim); r2 = rng.random(dim)
            A1 = 2 * a * r1 - a; C1 = 2 * r2
            X1 = alpha - A1 * np.abs(C1 * alpha - X)
            r1 = rng.random(dim); r2 = rng.random(dim)
            A2 = 2 * a * r1 - a; C2 = 2 * r2
            X2 = beta - A2 * np.abs(C2 * beta - X)
            r1 = rng.random(dim); r2 = rng.random(dim)
            A3 = 2 * a * r1 - a; C3 = 2 * r2
            X3 = delta - A3 * np.abs(C3 * delta - X)
            gwo_target = (X1 + X2 + X3) / 3.0
            rp = rng.random(dim)
            ra = rng.random(dim)
            rg = rng.random(dim)
            velocities[i] = (
                inertia * velocities[i]
                + c1 * rp * (pbest[i] - X)
                + c2 * ra * (alpha - X)
                + c3 * rg * (gwo_target - X)
            )
            velocities[i] = np.clip(velocities[i], -0.25 * span, 0.25 * span)
            new_wolves[i] = np.clip(X + velocities[i], lb, ub)
        new_fitness = np.array([objective_from_params(w, datasets) for w in new_wolves], dtype=float)
        wolves = new_wolves
        fitness = new_fitness
        improved = fitness < pbest_fitness
        pbest[improved] = wolves[improved]
        pbest_fitness[improved] = fitness[improved]
        if progress_callback is not None and ((it + 1) % max(progress_every, 1) == 0 or it == n_iter - 1):
            progress_callback(it + 1, n_iter, float(np.min(fitness)))
    order = np.argsort(fitness)
    wolves = wolves[order]
    fitness = fitness[order]
    return wolves[0], float(fitness[0]), pd.DataFrame(history, columns=['iteration', 'best_objective'])


def evaluate_all_samples(best_theta, datasets, bus_id):
    params = theta_to_params(best_theta)
    rows = []
    all_preds = []
    for name, df in datasets:
        pred = vehicle_power_profile(df, params)
        pred = pred.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        if len(pred) == 0:
            continue
        y = finite_series(pred['power_measured_W'])
        yhat = finite_series(pred['P_model_W'])
        e_meas = float(pred['energy_cum_measured_kWh'].iloc[-1])
        e_pred = float(pred['energy_cum_model_kWh'].iloc[-1])
        rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
        mae = float(np.mean(np.abs(yhat - y)))
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1 - np.sum((y - yhat) ** 2) / ss_tot) if ss_tot > 1e-9 else np.nan
        energy_err = e_pred - e_meas
        energy_err_pct = 100.0 * energy_err / max(abs(e_meas), 1e-6)
        rows.append([
            bus_id, name, len(pred), float(np.mean(pred['speed_m_per_s']) * 3.6), float(np.mean(np.abs(pred['acceleration_m_s2']))),
            float(np.mean(np.abs(pred['grade'])) * 100.0), float(np.mean(pred['mass_kg']) / 1000.0),
            float(np.mean(pred['P_aux_W']) / 1000.0), float(np.mean(pred['P_prop_W']) / 1000.0),
            rmse, mae, r2, e_meas, e_pred, energy_err, energy_err_pct,
        ])
        pred['file'] = name
        pred['bus_id'] = bus_id
        all_preds.append(pred)
    metrics_df = pd.DataFrame(rows, columns=[
        'bus_id', 'file', 'n_samples', 'avg_speed_kmh', 'avg_abs_accel_mps2', 'avg_abs_grade_pct', 'avg_mass_t',
        'avg_aux_kW', 'avg_prop_kW', 'rmse_W', 'mae_W', 'r2', 'energy_measured_kWh', 'energy_model_kWh',
        'energy_error_kWh', 'energy_error_pct'
    ])
    all_df = pd.concat(all_preds, ignore_index=True) if len(all_preds) > 0 else pd.DataFrame()
    metrics_df = metrics_df.replace([np.inf, -np.inf], np.nan)
    return metrics_df, all_df, params


def save_parameter_table(best_theta, bus_id):
    table = []
    for i, spec in enumerate(PARAMETER_SPECS):
        table.append([bus_id, spec['name'], spec['label'], spec['unit'], spec['default'], spec['lb'], spec['ub'], float(best_theta[i])])
    df = pd.DataFrame(table, columns=['bus_id', 'parameter', 'description', 'unit', 'default', 'lower_bound', 'upper_bound', 'best_estimate'])
    return df


def plot_global_energy_bars(metrics_df):
    df = clean_plot_df(metrics_df, ['energy_measured_kWh', 'energy_model_kWh'])
    if len(df) == 0:
        return
    bins = max(8, min(24, int(np.sqrt(len(df)) * 2)))
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    axs[0].hist(df['energy_measured_kWh'], bins=bins, alpha=0.65, color='tab:blue', label='Measured energy')
    axs[0].hist(df['energy_model_kWh'], bins=bins, alpha=0.65, color='tab:orange', label='Model energy')
    axs[0].set_ylabel('Count')
    axs[0].set_title('Measured vs model energy distribution (all samples, summarized)')
    axs[0].grid(alpha=0.25, axis='y')
    axs[0].legend()

    monthly = df.copy()
    monthly['month'] = extract_dataset_month(monthly['file'])
    monthly['month_sort'] = pd.to_datetime(monthly['month'] + '-01', errors='coerce')
    monthly = monthly.dropna(subset=['month_sort'])
    if len(monthly) > 0:
        summary = monthly.groupby('month', dropna=False).agg(
            measured_kWh=('energy_measured_kWh', 'mean'),
            model_kWh=('energy_model_kWh', 'mean'),
        ).reset_index()
        summary['month_sort'] = pd.to_datetime(summary['month'] + '-01', errors='coerce')
        summary = summary.sort_values('month_sort').reset_index(drop=True)
        x = np.arange(len(summary), dtype=float)
        axs[1].plot(x, summary['measured_kWh'], color='tab:blue', lw=1.6, alpha=0.65)
        axs[1].plot(x, summary['model_kWh'], color='tab:orange', lw=1.6, alpha=0.65)
        axs[1].plot(x, polynomial_trend(x, summary['measured_kWh'], degree=2), color='tab:blue', lw=2.8, ls='--', label='Measured trend')
        axs[1].plot(x, polynomial_trend(x, summary['model_kWh'], degree=2), color='tab:orange', lw=2.8, ls='--', label='Model trend')
        axs[1].set_xticks(x)
        axs[1].set_xticklabels(summary['month'], rotation=25, ha='right')
    axs[1].set_ylabel('Energy (kWh)')
    axs[1].set_xlabel('Month')
    axs[1].set_title('Monthly summary trendlines of measured and model energy')
    axs[1].grid(alpha=0.25)
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'energy_measured_vs_model_summary_histogram.png', dpi=220)
    plt.close(fig)


def plot_force_components_all_samples(all_df):
    if all_df is None or len(all_df) == 0:
        return
    df = clean_plot_df(all_df, ['F_roll_N', 'F_aero_N', 'F_inertia_N', 'F_grade_N'])
    if len(df) == 0:
        return
    vals = [df[c].abs().mean() for c in ['F_roll_N', 'F_aero_N', 'F_inertia_N', 'F_grade_N']]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(['Rolling', 'Aerodynamic', 'Inertia', 'Grade'], vals,
           color=['#4c78a8', '#f58518', '#54a24b', '#e45756'])
    ax.set_ylabel('Mean absolute force (N)')
    ax.set_title('Average Force Contribution Across All Samples')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'force_components_all_samples.png', dpi=220)
    plt.close(fig)


def plot_energy_param_scatter(metrics_df, params_df):
    df = metrics_df.copy()
    for col in ['energy_measured_kWh', 'energy_model_kWh', 'energy_error_pct', 'avg_speed_kmh', 'avg_abs_grade_pct', 'avg_mass_t']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    if len(df) == 0:
        return

    fig, axs = plt.subplots(1, 3, figsize=(16, 5.0))
    views = [
        ('avg_speed_kmh', 'Avg speed (km/h)', 'tab:blue', 'Energy vs speed summary'),
        ('avg_abs_grade_pct', 'Avg abs grade (%)', 'tab:orange', 'Energy vs grade summary'),
        ('avg_mass_t', 'Avg mass (t)', 'tab:green', 'Energy vs mass summary'),
    ]
    for ax, (xcol, xlbl, color, ttl) in zip(axs, views):
        sub = df[[xcol, 'energy_measured_kWh', 'energy_model_kWh']].dropna().sort_values(xcol).reset_index(drop=True)
        if len(sub) == 0:
            continue
        x = sub[xcol].to_numpy(dtype=float)
        y_meas = sub['energy_measured_kWh'].to_numpy(dtype=float)
        y_model = sub['energy_model_kWh'].to_numpy(dtype=float)
        ax.plot(x, y_meas, color=color, alpha=0.28, lw=1.1)
        ax.plot(x, y_model, color='tab:red', alpha=0.28, lw=1.1)
        ax.plot(x, polynomial_trend(x, y_meas, degree=2), color=color, lw=2.7, ls='--', label='Measured trend')
        ax.plot(x, polynomial_trend(x, y_model, degree=2), color='tab:red', lw=2.7, ls='--', label='Model trend')
        ax.set_xlabel(xlbl)
        ax.set_ylabel('Energy (kWh)')
        ax.set_title(ttl)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'energy_vs_physics_summary_trendlines.png', dpi=220)
    plt.close(fig)

    if params_df is None or len(params_df) == 0:
        return
    pivot = params_df.pivot(index='parameter', columns='bus_id', values='best_estimate')
    if len(pivot) == 0:
        return
    x = np.arange(len(pivot), dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, bus_id in enumerate(pivot.columns):
        vals = pd.to_numeric(pivot[bus_id], errors='coerce').to_numpy(dtype=float)
        ax.plot(x, vals, color=f'C{i}', alpha=0.45, lw=1.4)
        ax.plot(x, polynomial_trend(x, vals, degree=2), color=f'C{i}', lw=2.7, ls='--', label=f'Bus {bus_id} trend')
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=25, ha='right')
    ax.set_title('PGWO parameter profile trendlines')
    ax.set_ylabel('Estimated value')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'gwo_parameter_estimates_trendlines.png', dpi=220)
    plt.close(fig)


def plot_performance_trendlines_monthly(metrics_df, bus_order):
    df = metrics_df.copy()
    if len(df) == 0:
        return
    for c in ['rmse_W', 'mae_W', 'r2', 'energy_error_pct']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['month'] = extract_dataset_month(df['file'])
    df['month_sort'] = pd.to_datetime(df['month'] + '-01', errors='coerce')
    df = df.dropna(subset=['month_sort'])
    if len(df) == 0:
        return
    monthly = df.groupby(['bus_id', 'month'], dropna=False).agg(
        rmse_W=('rmse_W', 'mean'),
        mae_W=('mae_W', 'mean'),
        r2=('r2', 'mean'),
        abs_energy_error_pct=('energy_error_pct', lambda x: np.nanmean(np.abs(pd.to_numeric(x, errors='coerce')))),
    ).reset_index()
    monthly['month_sort'] = pd.to_datetime(monthly['month'] + '-01', errors='coerce')
    monthly = monthly.sort_values(['bus_id', 'month_sort']).reset_index(drop=True)

    metrics = ['rmse_W', 'mae_W', 'r2', 'abs_energy_error_pct']
    titles = ['RMSE trend', 'MAE trend', 'R² trend', 'Abs energy error (%) trend']
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    axs = axs.flatten()
    for ax, col, ttl in zip(axs, metrics, titles):
        for i, bus_id in enumerate(bus_order):
            sub = monthly[monthly['bus_id'] == bus_id].copy()
            if len(sub) == 0:
                continue
            x = np.arange(len(sub), dtype=float)
            y = pd.to_numeric(sub[col], errors='coerce').to_numpy(dtype=float)
            ax.plot(x, y, color=f'C{i}', alpha=0.35, lw=1.2)
            ax.plot(x, polynomial_trend(x, y, degree=2), color=f'C{i}', lw=2.6, ls='--', label=f'Bus {bus_id}')
            ax.set_xticks(x)
            ax.set_xticklabels(sub['month'], rotation=25, ha='right', fontsize=8)
        ax.set_title(ttl)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle('Monthly performance trendlines by bus', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'monthly_performance_trendlines.png', dpi=220)
    plt.close(fig)


def plot_performance_distribution_histograms(metrics_df, bus_order):
    df = metrics_df.copy()
    if len(df) == 0:
        return
    metric_cols = ['rmse_W', 'mae_W', 'r2', 'energy_error_pct']
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    axs = axs.flatten()
    titles = ['RMSE distribution', 'MAE distribution', 'R² distribution', 'Energy error (%) distribution']
    for ax, col, ttl in zip(axs, metric_cols, titles):
        for i, bus_id in enumerate(bus_order):
            vals = pd.to_numeric(df.loc[df['bus_id'] == bus_id, col], errors='coerce').to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            bins = max(6, min(20, int(np.sqrt(len(vals)) * 2)))
            ax.hist(vals, bins=bins, alpha=0.45, color=f'C{i}', label=f'Bus {bus_id}')
        ax.set_title(ttl)
        ax.grid(alpha=0.25, axis='y')
        ax.legend(fontsize=8)
    fig.suptitle('Performance distributions (summarized)', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'performance_distributions_histogram.png', dpi=220)
    plt.close(fig)


def bootstrap_mean_ci(values, n_boot=1500, seed=42):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(np.mean(sample))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def cohen_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = ((len(x) - 1) * vx + (len(y) - 1) * vy) / max(len(x) + len(y) - 2, 1)
    if pooled <= 1e-12:
        return np.nan
    return float((np.mean(y) - np.mean(x)) / np.sqrt(pooled))


def build_statistical_benchmark(all_metrics_df, bus_order):
    metric_cols = ['rmse_W', 'mae_W', 'r2', 'energy_error_kWh', 'energy_error_pct']
    per_bus_rows = []
    for bus_id in bus_order:
        sub = all_metrics_df[all_metrics_df['bus_id'] == bus_id].copy()
        for m in metric_cols:
            vals = pd.to_numeric(sub[m], errors='coerce').to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            ci_lo, ci_hi = bootstrap_mean_ci(vals)
            per_bus_rows.append([
                bus_id, m, len(vals), float(np.mean(vals)) if len(vals) else np.nan, float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                float(np.median(vals)) if len(vals) else np.nan, float(np.percentile(vals, 10)) if len(vals) else np.nan,
                float(np.percentile(vals, 90)) if len(vals) else np.nan, ci_lo, ci_hi
            ])
    per_bus_df = pd.DataFrame(per_bus_rows, columns=[
        'bus_id', 'metric', 'n', 'mean', 'std', 'median', 'p10', 'p90', 'mean_ci95_low', 'mean_ci95_high'
    ])

    comp_rows = []
    if len(bus_order) >= 2:
        b1, b2 = bus_order[0], bus_order[1]
        for m in metric_cols:
            x = pd.to_numeric(all_metrics_df.loc[all_metrics_df['bus_id'] == b1, m], errors='coerce').to_numpy(dtype=float)
            y = pd.to_numeric(all_metrics_df.loc[all_metrics_df['bus_id'] == b2, m], errors='coerce').to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            y = y[np.isfinite(y)]
            mean_diff = float(np.mean(y) - np.mean(x)) if len(x) and len(y) else np.nan
            comp_rows.append([m, b1, b2, mean_diff, cohen_d(x, y)])
    comp_df = pd.DataFrame(comp_rows, columns=['metric', 'reference_bus', 'comparison_bus', 'mean_diff', 'cohen_d'])
    return per_bus_df, comp_df


def plot_bus_performance_comparison(all_metrics_df, bus_order):
    metric_cols = ['rmse_W', 'mae_W', 'r2', 'energy_error_pct']
    titles = ['RMSE (W)', 'MAE (W)', 'R²', 'Energy error (%)']
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    axs = axs.flatten()
    for i, (m, ttl) in enumerate(zip(metric_cols, titles)):
        samples = []
        labels = []
        for bus_id in bus_order:
            vals = pd.to_numeric(all_metrics_df.loc[all_metrics_df['bus_id'] == bus_id, m], errors='coerce').to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                samples.append(vals)
                labels.append(f'Bus {bus_id}')
        if samples:
            axs[i].boxplot(samples, labels=labels, showmeans=True)
        axs[i].set_title(ttl)
        axs[i].grid(alpha=0.25)
    fig.suptitle('PGWO performance comparison across buses', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'bus_performance_comparison_boxplots.png', dpi=220)
    plt.close(fig)


def plot_bus_parameter_comparison(param_df):
    pivot = param_df.pivot(index='parameter', columns='bus_id', values='best_estimate')
    if len(pivot.columns) < 2:
        return
    x = np.arange(len(pivot))
    width = 0.35
    buses = list(pivot.columns)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, pivot[buses[0]], width, label=f'Bus {buses[0]}', color='tab:blue', alpha=0.85)
    ax.bar(x + width / 2, pivot[buses[1]], width, label=f'Bus {buses[1]}', color='tab:orange', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=25, ha='right')
    ax.set_ylabel('Estimated parameter value')
    ax.set_title('Estimated parameter comparison (PGWO)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'bus_parameter_estimates_comparison.png', dpi=220)
    plt.close(fig)


def plot_pgwo_vs_actual_parameter_histogram(param_df, bus_order):
    if param_df is None or len(param_df) == 0:
        return
    pivot = param_df.pivot(index='parameter', columns='bus_id', values='best_estimate')
    if len(pivot) == 0:
        return
    buses = [b for b in bus_order if b in pivot.columns]
    if len(buses) == 0:
        return
    actual_map = {spec['name']: float(spec['default']) for spec in PARAMETER_SPECS}
    params = list(pivot.index)
    x = np.arange(len(params))
    width = 0.18 if len(buses) >= 2 else 0.28

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, bus_id in enumerate(buses[:2]):
        est_vals = pd.to_numeric(pivot[bus_id], errors='coerce').to_numpy(dtype=float)
        actual_vals = np.array([actual_map.get(p, np.nan) for p in params], dtype=float)
        offset = (-1.5 + 2 * i) * width
        ax.bar(x + offset, est_vals, width=width, alpha=0.85, label=f'Bus {bus_id} PGWO', color=f'C{i}')
        ax.bar(
            x + offset + width,
            actual_vals,
            width=width,
            alpha=0.65,
            label=f'Bus {bus_id} Actual',
            color=f'C{i}',
            hatch='//',
            edgecolor='black',
            linewidth=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(params, rotation=25, ha='right')
    ax.set_ylabel('Parameter value')
    ax.set_title('PGWO parameters vs actual bus parameters')
    ax.grid(alpha=0.25, axis='y')
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'pgwo_vs_actual_parameters_histogram.png', dpi=220)
    plt.close(fig)


def plot_energy_histogram_all_datasets(metrics_df):
    df = clean_plot_df(metrics_df, ['energy_measured_kWh', 'energy_model_kWh'])
    if len(df) == 0:
        return
    summary = df.groupby('bus_id', dropna=False).agg(
        measured_mean_kWh=('energy_measured_kWh', 'mean'),
        model_mean_kWh=('energy_model_kWh', 'mean'),
        measured_median_kWh=('energy_measured_kWh', 'median'),
        model_median_kWh=('energy_model_kWh', 'median'),
    ).reset_index()
    x = np.arange(len(summary), dtype=float)
    width = 0.32

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width / 2, summary['measured_mean_kWh'], width=width, label='Measured mean', color='tab:blue', alpha=0.85)
    ax.bar(x + width / 2, summary['model_mean_kWh'], width=width, label='Model mean', color='tab:orange', alpha=0.85)
    ax.plot(x, summary['measured_median_kWh'], color='tab:blue', lw=2.0, ls='--', marker='o', label='Measured median trend')
    ax.plot(x, summary['model_median_kWh'], color='tab:orange', lw=2.0, ls='--', marker='o', label='Model median trend')
    ax.set_ylabel('Energy (kWh)')
    ax.set_title('Measured vs model energy summary by bus')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Bus {b}' for b in summary['bus_id']])
    ax.grid(alpha=0.22, axis='y')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'energy_bars_summary_by_bus.png', dpi=220)
    plt.close(fig)


def plot_bus_physical_averages(all_df, bus_order):
    if all_df is None or len(all_df) == 0:
        return
    required = ['bus_id', 'file', 'speed_m_per_s', 'temperature_C', 'F_roll_N', 'F_aero_N', 'F_inertia_N', 'F_grade_N', 'F_total_N']
    df = all_df.copy()
    for c in required:
        if c not in df.columns:
            return
    numeric_cols = ['speed_m_per_s', 'temperature_C', 'F_roll_N', 'F_aero_N', 'F_inertia_N', 'F_grade_N', 'F_total_N']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['bus_id', 'file', 'speed_m_per_s', 'temperature_C'])
    if len(df) == 0:
        return

    df['month'] = extract_dataset_month(df['file'])
    df['month_sort'] = pd.to_datetime(df['month'] + '-01', errors='coerce')
    df = df.dropna(subset=['month_sort'])
    if len(df) == 0:
        return

    agg = df.groupby(['bus_id', 'month'], dropna=False).agg(
        avg_speed_kmh=('speed_m_per_s', lambda x: float(np.nanmean(pd.to_numeric(x, errors='coerce')) * 3.6)),
        avg_temperature_C=('temperature_C', lambda x: float(np.nanmean(pd.to_numeric(x, errors='coerce')))),
        avg_abs_roll_kN=('F_roll_N', lambda x: float(np.nanmean(np.abs(pd.to_numeric(x, errors='coerce'))) / 1000.0)),
        avg_abs_aero_kN=('F_aero_N', lambda x: float(np.nanmean(np.abs(pd.to_numeric(x, errors='coerce'))) / 1000.0)),
        avg_abs_inertia_kN=('F_inertia_N', lambda x: float(np.nanmean(np.abs(pd.to_numeric(x, errors='coerce'))) / 1000.0)),
        avg_abs_grade_kN=('F_grade_N', lambda x: float(np.nanmean(np.abs(pd.to_numeric(x, errors='coerce'))) / 1000.0)),
        avg_abs_total_kN=('F_total_N', lambda x: float(np.nanmean(np.abs(pd.to_numeric(x, errors='coerce'))) / 1000.0)),
    ).reset_index()
    agg['month_sort'] = pd.to_datetime(agg['month'] + '-01', errors='coerce')

    for bus_id in bus_order:
        sub = agg[agg['bus_id'] == bus_id].copy().sort_values('month_sort').reset_index(drop=True)
        if len(sub) == 0:
            continue
        x = np.arange(len(sub), dtype=float)
        labels = sub['month'].astype(str).tolist()
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        axs[0].plot(x, sub['avg_speed_kmh'], color='tab:orange', alpha=0.45, lw=1.2)
        axs[0].plot(x, polynomial_trend(x, sub['avg_speed_kmh'], degree=2), color='tab:orange', lw=2.8, ls='--', label='Speed trend')
        axs[0].plot(x, sub['avg_temperature_C'], color='tab:red', alpha=0.45, lw=1.2)
        axs[0].plot(x, polynomial_trend(x, sub['avg_temperature_C'], degree=2), color='tab:red', lw=2.8, ls='--', label='Temperature trend')
        axs[0].set_ylabel('Speed (km/h) / Temp (C)')
        axs[0].set_title(f'Bus {bus_id}: monthly physics summaries (entire dataset)')
        axs[0].grid(alpha=0.25, axis='y')
        axs[0].legend(fontsize=8)

        for col, color, label in [
            ('avg_abs_total_kN', 'black', 'Total'),
            ('avg_abs_roll_kN', 'tab:blue', 'Rolling'),
            ('avg_abs_aero_kN', 'tab:orange', 'Aerodynamic'),
            ('avg_abs_inertia_kN', 'tab:green', 'Inertia'),
            ('avg_abs_grade_kN', 'tab:red', 'Grade'),
        ]:
            axs[1].plot(x, sub[col], color=color, alpha=0.30, lw=1.2)
            axs[1].plot(x, polynomial_trend(x, sub[col], degree=2), color=color, lw=2.5, ls='--', label=f'{label} trend')
        axs[1].set_ylabel('Avg |Force| (kN)')
        axs[1].set_title(f'Bus {bus_id}: monthly force component trendlines')
        axs[1].grid(alpha=0.25)
        axs[1].legend(ncol=3, fontsize=8)

        force_cols = ['avg_abs_roll_kN', 'avg_abs_aero_kN', 'avg_abs_inertia_kN', 'avg_abs_grade_kN']
        stacked = np.vstack([pd.to_numeric(sub[c], errors='coerce').to_numpy(dtype=float) for c in force_cols])
        y_force_mean = np.nanmean(stacked, axis=0)
        axs[2].plot(x, y_force_mean, color='tab:purple', alpha=0.45, lw=1.2)
        axs[2].plot(x, polynomial_trend(x, y_force_mean, degree=2), color='tab:purple', lw=2.8, ls='--', label='Mean force trend')
        axs[2].plot(x, sub['avg_abs_total_kN'], color='black', alpha=0.45, lw=1.2)
        axs[2].plot(x, polynomial_trend(x, sub['avg_abs_total_kN'], degree=2), color='black', lw=2.8, ls='--', label='Total force trend')
        axs[2].set_ylabel('Force (kN)')
        axs[2].set_title(f'Bus {bus_id}: compact force summary trendlines')
        axs[2].grid(alpha=0.25)
        axs[2].legend(fontsize=8)

        axs[2].set_xticks(x)
        axs[2].set_xticklabels(labels, rotation=25, fontsize=8, ha='right')
        axs[2].set_xlabel('Month')
        fig.tight_layout()
        fig.savefig(OUT_DIR / f'bus_{bus_id}_physical_parameter_averages.png', dpi=220)
        plt.close(fig)


def plot_bus_convergence(convergence_df):
    return


def plot_energy_scatter_by_bus(all_df):
    if all_df is None or len(all_df) == 0:
        return
    df = all_df.copy()
    df['power_measured_W'] = pd.to_numeric(df['power_measured_W'], errors='coerce')
    df['P_model_W'] = pd.to_numeric(df['P_model_W'], errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['power_measured_W', 'P_model_W', 'bus_id'])
    if len(df) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    for bus_id, sub in df.groupby('bus_id'):
        sub = make_plot_sample(sub, max_points=2500)
        ax.scatter(sub['power_measured_W'] / 1000.0, sub['P_model_W'] / 1000.0, alpha=0.25, s=8, label=f'Bus {bus_id}')
    lim = np.nanmax(np.abs(np.concatenate([(df['power_measured_W'] / 1000.0).to_numpy(), (df['P_model_W'] / 1000.0).to_numpy()])))
    lim = float(max(lim, 10.0))
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1.3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('Measured power (kW)')
    ax.set_ylabel('Model power (kW)')
    ax.set_title('Measured vs model power by bus')
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'bus_power_scatter_comparison.png', dpi=220)
    plt.close(fig)


def add_time_trend(ax, t, y, color, label):
    ys = gaussian_smooth(y, sigma=GAUSSIAN_SIGMA)
    yr = polynomial_trend(t, y, degree=TREND_DEGREE)
    ax.plot(t, ys, color=color, lw=SMOOTH_LINEWIDTH, alpha=0.6, label=f'{label} smooth')
    ax.plot(t, yr, color=color, lw=REG_LINEWIDTH, ls='--', alpha=0.95, label=f'{label} regression')


def plot_dataset_diagnostics(sample_name, pred_df):
    if pred_df is None or len(pred_df) == 0:
        return
    df = pred_df.copy()
    numeric_cols = [
        'time_s', 'power_measured_W', 'P_model_W', 'power_error_W', 'P_prop_W', 'P_aux_W',
        'F_roll_N', 'F_aero_N', 'F_inertia_N', 'F_grade_N', 'speed_m_per_s', 'grade'
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['time_s', 'power_measured_W', 'P_model_W'])
    if len(df) == 0:
        return

    plot_df = make_plot_sample(df, max_points=PLOT_MAX_POINTS)
    t = finite_series(plot_df['time_s'] / 60.0)

    measured_kw = plot_filter_series(plot_df['power_measured_W'] / 1000.0, hard_clip=PLOT_POWER_LIMIT_KW).to_numpy()
    model_kw = plot_filter_series(plot_df['P_model_W'] / 1000.0, hard_clip=PLOT_POWER_LIMIT_KW).to_numpy()
    residual_kw = plot_filter_series(plot_df['power_error_W'] / 1000.0, hard_clip=PLOT_RESIDUAL_KW_LIMIT).to_numpy()

    prop_kw = plot_filter_series(plot_df['P_prop_W'] / 1000.0, hard_clip=PLOT_POWER_LIMIT_KW).to_numpy()
    aux_kw = plot_filter_series(plot_df['P_aux_W'] / 1000.0, hard_clip=20.0).to_numpy()

    roll_n = plot_filter_series(plot_df['F_roll_N'], hard_clip=PLOT_FORCE_LIMIT_N).to_numpy()
    aero_n = plot_filter_series(plot_df['F_aero_N'], hard_clip=PLOT_FORCE_LIMIT_N).to_numpy()
    inertia_n = plot_filter_series(plot_df['F_inertia_N'], hard_clip=PLOT_FORCE_LIMIT_N).to_numpy()
    grade_n = plot_filter_series(plot_df['F_grade_N'], hard_clip=PLOT_FORCE_LIMIT_N).to_numpy()

    speed_kmh = plot_filter_series(plot_df['speed_m_per_s'] * 3.6, hard_clip=90.0).to_numpy()
    grade_pct = plot_filter_series(plot_df['grade'] * 100.0, hard_clip=PLOT_GRADE_LIMIT_PCT).to_numpy()

    tag = sample_name.replace('_cleaned', '')[:80]

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axs[0].plot(t, measured_kw, color='tab:blue', alpha=RAW_ALPHA, lw=RAW_LINEWIDTH)
    axs[0].plot(t, model_kw, color='tab:red', alpha=RAW_ALPHA, lw=RAW_LINEWIDTH)
    add_time_trend(axs[0], t, measured_kw, 'tab:blue', 'Measured')
    add_time_trend(axs[0], t, model_kw, 'tab:red', 'Model')
    axs[0].set_ylabel('Power (kW)')
    axs[0].set_title(f'Power profile (trendlines): {tag}')
    axs[0].legend(ncol=2, fontsize=9)

    axs[1].plot(t, residual_kw, color='tab:cyan', alpha=RAW_ALPHA, lw=RAW_LINEWIDTH)
    add_time_trend(axs[1], t, residual_kw, 'tab:cyan', 'Residual')
    axs[1].axhline(0, color='black', lw=1)
    axs[1].set_xlabel('Time (min)')
    axs[1].set_ylabel('Residual (kW)')
    axs[1].set_title('Residual over time')
    axs[1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f'{tag}_power_profile_trends.png', dpi=220)
    plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axs[0].plot(t, prop_kw, color='tab:purple', alpha=RAW_ALPHA, lw=RAW_LINEWIDTH)
    axs[0].plot(t, aux_kw, color='tab:green', alpha=RAW_ALPHA, lw=RAW_LINEWIDTH)
    axs[0].plot(t, model_kw, color='tab:red', alpha=RAW_ALPHA, lw=RAW_LINEWIDTH)
    add_time_trend(axs[0], t, prop_kw, 'tab:purple', 'Propulsion')
    add_time_trend(axs[0], t, aux_kw, 'tab:green', 'Auxiliary')
    add_time_trend(axs[0], t, model_kw, 'tab:red', 'Total')
    axs[0].set_ylabel('Power (kW)')
    axs[0].set_title('Power components (trendlines)')
    axs[0].legend(ncol=2, fontsize=9)

    axs[1].plot(t, roll_n, alpha=RAW_ALPHA, lw=RAW_LINEWIDTH, color='tab:blue')
    axs[1].plot(t, aero_n, alpha=RAW_ALPHA, lw=RAW_LINEWIDTH, color='tab:orange')
    axs[1].plot(t, inertia_n, alpha=RAW_ALPHA, lw=RAW_LINEWIDTH, color='tab:green')
    axs[1].plot(t, grade_n, alpha=RAW_ALPHA, lw=RAW_LINEWIDTH, color='tab:red')
    add_time_trend(axs[1], t, roll_n, 'tab:blue', 'Rolling')
    add_time_trend(axs[1], t, aero_n, 'tab:orange', 'Aero')
    add_time_trend(axs[1], t, inertia_n, 'tab:green', 'Inertia')
    add_time_trend(axs[1], t, grade_n, 'tab:red', 'Grade')
    axs[1].set_xlabel('Time (min)')
    axs[1].set_ylabel('Force (N)')
    axs[1].set_title('Force components (trendlines)')
    axs[1].legend(ncol=2, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f'{tag}_components_trends.png', dpi=220)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(t, speed_kmh, color='tab:orange', alpha=RAW_ALPHA, lw=RAW_LINEWIDTH)
    add_time_trend(ax1, t, speed_kmh, 'tab:orange', 'Speed')
    ax1.set_ylabel('Speed (km/h)', color='tab:orange')
    ax1.tick_params(axis='y', labelcolor='tab:orange')

    ax2 = ax1.twinx()
    ax2.plot(t, grade_pct, color='tab:brown', alpha=RAW_ALPHA, lw=RAW_LINEWIDTH)
    add_time_trend(ax2, t, grade_pct, 'tab:brown', 'Grade')
    ax2.set_ylabel('Grade (%)', color='tab:brown')
    ax2.tick_params(axis='y', labelcolor='tab:brown')

    ax1.set_xlabel('Time (min)')
    ax1.set_title(f'Physics inputs (trendlines): {tag}')
    ax1.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f'{tag}_speed_grade_trends.png', dpi=220)
    plt.close(fig)


def main():
    run_t0 = time.perf_counter()
    def log_progress(message: str):
        elapsed_s = time.perf_counter() - run_t0
        print(f"[+{elapsed_s:8.1f}s] {message}", flush=True)

    log_progress("Starting calibration run")
    all_bus_metrics = []
    all_bus_params = []
    all_bus_convergence = []
    all_bus_predictions = []
    bus_order = []

    for i, bus_id in enumerate(BUS_IDS):
        log_progress(f"Loading parquet files for bus {bus_id} ({i + 1}/{len(BUS_IDS)})")
        datasets = load_bus_dataset(bus_id, DATASET_DIR)
        if not datasets:
            log_progress(f"Bus {bus_id}: no usable files found, skipping")
            continue
        log_progress(f"Bus {bus_id}: loaded {len(datasets)} preprocessed samples")
        bus_order.append(bus_id)
        log_progress(f"Bus {bus_id}: starting optimization (wolves=24, iterations=80)")
        best_theta, best_obj, history_df = pgwo_optimize(
            datasets,
            n_wolves=24,
            n_iter=80,
            seed=42 + i,
            progress_callback=lambda cur, total, best: log_progress(
                f"Bus {bus_id}: optimization {cur}/{total}, best objective={best:.3e}"
            ),
            progress_every=5,
        )
        log_progress(f"Bus {bus_id}: optimization finished, evaluating all samples")
        metrics_df, pred_all_df, best_params = evaluate_all_samples(best_theta, datasets, bus_id=bus_id)
        param_df = save_parameter_table(best_theta, bus_id=bus_id)

        history_df['bus_id'] = bus_id
        all_bus_convergence.append(history_df)
        all_bus_metrics.append(metrics_df)
        all_bus_params.append(param_df)
        if pred_all_df is not None and len(pred_all_df) > 0:
            all_bus_predictions.append(pred_all_df)

        bus_summary = pd.DataFrame({
            'bus_id': [bus_id],
            'best_objective': [best_obj],
            'n_sample_files': [len(metrics_df)],
            'mean_rmse_W': [pd.to_numeric(metrics_df['rmse_W'], errors='coerce').mean()],
            'mean_mae_W': [pd.to_numeric(metrics_df['mae_W'], errors='coerce').mean()],
            'mean_r2': [pd.to_numeric(metrics_df['r2'], errors='coerce').mean()],
            'mean_abs_energy_error_kWh': [pd.to_numeric(metrics_df['energy_error_kWh'], errors='coerce').abs().mean()],
            'mean_abs_energy_error_pct': [pd.to_numeric(metrics_df['energy_error_pct'], errors='coerce').abs().mean()],
        })
        bus_summary.to_csv(OUT_DIR / f'pgwo_bus_{bus_id}_summary.csv', index=False)
        pd.DataFrame({'parameter': list(best_params.keys()), 'value': list(best_params.values())}).to_csv(OUT_DIR / f'pgwo_bus_{bus_id}_best_parameters.csv', index=False)
        metrics_df.to_csv(OUT_DIR / f'pgwo_bus_{bus_id}_sample_metrics.csv', index=False)
        log_progress(f"Bus {bus_id}: outputs saved")

    if not all_bus_metrics:
        raise FileNotFoundError(f'No usable parquet files found in {DATASET_DIR / "bus=183"} and {DATASET_DIR / "bus=208"}')

    log_progress("Building global summaries and plots")
    metrics_all_df = pd.concat(all_bus_metrics, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    params_all_df = pd.concat(all_bus_params, ignore_index=True)
    convergence_all_df = pd.concat(all_bus_convergence, ignore_index=True)
    predictions_all_df = pd.concat(all_bus_predictions, ignore_index=True) if len(all_bus_predictions) > 0 else pd.DataFrame()
    stats_per_bus_df, stats_comp_df = build_statistical_benchmark(metrics_all_df, bus_order)

    params_pivot = params_all_df.pivot(index=['parameter', 'description', 'unit', 'default', 'lower_bound', 'upper_bound'], columns='bus_id', values='best_estimate').reset_index()
    if len(bus_order) >= 2 and bus_order[0] in params_pivot.columns and bus_order[1] in params_pivot.columns:
        params_pivot['delta_bus2_minus_bus1'] = params_pivot[bus_order[1]] - params_pivot[bus_order[0]]

    metrics_all_df.to_csv(OUT_DIR / 'pgwo_two_bus_sample_metrics.csv', index=False)
    params_all_df.to_csv(OUT_DIR / 'pgwo_parameter_estimates_long.csv', index=False)
    params_pivot.to_csv(OUT_DIR / 'pgwo_parameter_estimates_comparison_table.csv', index=False)
    convergence_all_df.to_csv(OUT_DIR / 'pgwo_two_bus_convergence.csv', index=False)
    stats_per_bus_df.to_csv(OUT_DIR / 'pgwo_statistical_benchmark_per_bus.csv', index=False)
    stats_comp_df.to_csv(OUT_DIR / 'pgwo_statistical_benchmark_comparison.csv', index=False)

    overall_summary = metrics_all_df.groupby('bus_id', dropna=False).agg(
        n_files=('file', 'count'),
        mean_rmse_W=('rmse_W', 'mean'),
        mean_mae_W=('mae_W', 'mean'),
        mean_r2=('r2', 'mean'),
        mean_abs_energy_error_kWh=('energy_error_kWh', lambda x: np.nanmean(np.abs(pd.to_numeric(x, errors='coerce')))),
        mean_abs_energy_error_pct=('energy_error_pct', lambda x: np.nanmean(np.abs(pd.to_numeric(x, errors='coerce'))))
    ).reset_index()
    overall_summary.to_csv(OUT_DIR / 'pgwo_two_bus_summary.csv', index=False)

    plot_global_energy_bars(metrics_all_df)
    plot_energy_histogram_all_datasets(metrics_all_df)
    plot_energy_param_scatter(metrics_all_df, params_all_df)
    plot_bus_performance_comparison(metrics_all_df, bus_order)
    plot_performance_trendlines_monthly(metrics_all_df, bus_order)
    plot_performance_distribution_histograms(metrics_all_df, bus_order)
    plot_bus_parameter_comparison(params_all_df)
    plot_pgwo_vs_actual_parameter_histogram(params_all_df, bus_order)
    plot_bus_physical_averages(predictions_all_df, bus_order)
    log_progress("Run completed")


if __name__ == '__main__':
    main()
