import numpy as np
import pandas  as pd
from pathlib import Path

import matplotlib.pyplot as plt

NOISE_COMPONENT = 0.1

def get_project_path() -> Path:
    return Path(__file__).parent.parent


def get_data_folder(folder_name: str):
    data_folder = Path(get_project_path(), folder_name)
    data_folder.mkdir(exist_ok=True, parents=True)
    return data_folder


def generate_synthetic_time_series(n: int = 100,
                                   start: str = "2025-01-01",
                                   freq: str = "D",
                                   kind: str = "stationary_noise") -> pd.DataFrame:
    """
    Generate synthetic time series for teaching purposes.

    Parameters
    ----------
    n : int
        Number of data points.
    start : str
        Start date as a string (e.g., "2020-01-01 00:00:00").
    freq : str
        Frequency string compatible with pandas date_range
        (e.g., "D" for days, "H" for hours).
    kind : str, optional
        Type of time series to generate. Options:
        - "stationary_noise"                       : White noise, no trend, no seasonality, no cycles.
        - "stationary_cycles"                      : Stationary with damped cycles (AR-like).
        - "trend_linear_up"                        : Non-stationary with upward linear trend.
        - "trend_linear_down"                      : Non-stationary with downward linear trend.
        - "trend_exponential_up"                   : Non-stationary with exponential upward trend.
        - "trend_exponential_down"                 : Non-stationary with exponential downward trend.
        - "non_stationary_seasonal"                : Non-stationary with seasonality (sine wave + noise).
        - "structural_shift"                       : Stationary noise with mean shift at midpoint.
        - "trend_linear_up_seasonal"               : Linear upward trend + seasonality + noise.
        - "noise_var_decreasing"                   : White noise with variance decreasing over time.
        - "noise_var_increasing"                   : White noise with variance increasing over time.
        - "trend_linear_up_amp_increasing"         : Linear upward trend + seasonality with increasing amplitude.
        - "autocorr_shift"                         : Change in AR(1) coefficient at midpoint (covariance nonstationarity).
        - "non_stationary_seasonal_var_increasing" : Seasonality + variance increasing.
        - "stationary_cycles_var_decreasing"       : Damped cycles + variance decreasing.
    """
    # Time index
    datetime_index = pd.date_range(start=start, periods=n, freq=freq)
    t = np.arange(n)
    m_sync = 20  # common period for both series
    phi = 0.0  # common phase

    # Base noise
    noise = np.random.normal(0, NOISE_COMPONENT, n)
    eps = np.random.normal(0, 2, n)

    # simple heuristic for seasonal period based on freq
    f = str(freq).upper()
    if f.startswith("H"):
        # hourly data -> daily seasonality
        m = 24
    elif f.startswith("T") or f.startswith("MIN"):
        # minutely -> hourly seasonality
        m = 60
    elif f.startswith("S"):
        # secondly -> minutely seasonality
        m = 60
    elif f.startswith("D"):
        # daily -> weekly seasonality
        m = 7
    else:
        m = 12

    if kind == "stationary_noise":
        values = noise

    elif kind == "stationary_cycles":
        freq_base = 1 / 20
        freq_variation = 1 / 60 * np.sin(2 * np.pi * t / n)
        instantaneous_freq = freq_base + freq_variation
        phase = 2 * np.pi * np.cumsum(instantaneous_freq)
        # Damped cosine with slowly varying frequency
        values = np.cos(phase) * np.exp(-t / 400) + noise * 0.5

    elif kind == "non_stationary_seasonal":
        # Pure seasonality + noise
        values = np.sin(2 * np.pi * t / m) + noise

    elif kind == "trend_linear_up":
        values = 0.1 * t + noise

    elif kind == "trend_linear_down":
        values = -0.1 * t + noise

    elif kind == "trend_exponential_up":
        values = np.exp(0.03 * t) + noise

    elif kind == "trend_exponential_down":
        values = np.exp(-0.03 * t) + noise

    elif kind == "structural_shift":
        # White noise with mean shift at the midpoint
        shift_point = n // 2
        values = noise.copy()
        values[shift_point:] += 2.0

    elif kind == "trend_linear_up_seasonal":
        # Linear upward trend + seasonality + noise
        trend = 0.1 * t                         # slope controls trend strength
        seasonal = np.sin(2 * np.pi * t / m)    # fixed-amplitude seasonal component
        values = trend + seasonal + noise

    elif kind == "noise_var_decreasing":
        # Variance decreases over time (linear schedule on std)
        # std goes from high -> low
        std_schedule = np.linspace(20, 0.5, n)  # tweak endpoints as needed
        values = (NOISE_COMPONENT * std_schedule) * eps

    elif kind == "noise_var_increasing":
        # Variance increases over time (linear schedule on std)
        # std goes from low -> high
        std_schedule = np.linspace(0.5, 20, n)  # tweak endpoints as needed
        values = (NOISE_COMPONENT * std_schedule) * eps

    elif kind == "trend_linear_up_amp_increasing":
        # Linear upward trend
        trend = 0.1 * t
        # Amplitude of seasonal component increases over time
        amplitude = np.linspace(0.5, 3.0, n)
        seasonal = amplitude * np.sin(2 * np.pi * t / m)
        values = trend + seasonal + NOISE_COMPONENT * eps

    elif kind == "autocorr_shift":
        # AR(1) with a change in phi at the midpoint (covariance nonstationarity).
        # Before midpoint: strong positive autocorrelation (phi1=0.7)
        # After midpoint: negative autocorrelation (phi2=-0.3)
        shift_point = n // 2
        phi1, phi2 = 0.7, -0.3

        # Choose innovation std so that the unconditional variance remains base_sigma^2
        # For AR(1): Var(y) = sigma_e^2 / (1 - phi^2) => sigma_e = base_sigma * sqrt(1 - phi^2)
        sigma_e1 = NOISE_COMPONENT * np.sqrt(1.0 - phi1 ** 2)
        sigma_e2 = NOISE_COMPONENT * np.sqrt(1.0 - phi2 ** 2)

        e1 = np.random.normal(0, sigma_e1, shift_point)
        e2 = np.random.normal(0, sigma_e2, n - shift_point)

        y = np.zeros(n)
        # Initialize from the stationary distribution of the first regime
        y[0] = np.random.normal(0, NOISE_COMPONENT)

        # First regime
        for i in range(1, shift_point):
            y[i] = phi1 * y[i - 1] + e1[i]

        # Second regime
        for i in range(shift_point, n):
            prev = y[i - 1] if i > 0 else 0.0
            y[i] = phi2 * prev + e2[i - shift_point]

        values = y

    elif kind == "trend_piecewise":
        # Piecewise linear trend: slope up to midpoint, then slope down
        shift_point = n // 2
        slope1, slope2 = 0.12, -0.08  # tweak as needed
        trend = np.empty(n, dtype=float)
        trend[:shift_point] = slope1 * np.arange(shift_point)
        # Continue from the last level to avoid jumps at the kink
        base_level = trend[shift_point - 1]
        trend[shift_point:] = base_level + slope2 * np.arange(n - shift_point)
        values = trend + NOISE_COMPONENT * eps

    elif kind == "non_stationary_seasonal_var_increasing":
        std_up = np.linspace(0.3, 1.5, n)  # increasing variance
        seasonal = np.cos(2 * np.pi * t / m_sync + phi)  # SAME base wave as below
        values = seasonal + (NOISE_COMPONENT * std_up) * eps

    elif kind == "stationary_cycles_var_decreasing":
        std_down = np.linspace(1.5, 0.3, n)  # decreasing variance
        cycles = np.cos(2 * np.pi * t / m_sync + phi) * np.exp(-t / n)  # SAME wave + mild damping
        values = cycles + (NOISE_COMPONENT * std_down) * eps

    else:
        raise ValueError(f"Unknown kind {kind}. Choose from predefined options!")

    return pd.DataFrame({"datetime": datetime_index, "value": values})


def generate_dataset(n: int = 100,
                     start: str = "2025-01-01",
                     freq: str = "D"):
    cases = ["stationary_noise", "non_stationary_seasonal", "stationary_cycles",
             "trend_linear_up", "trend_linear_down", "trend_exponential_up", "structural_shift",
             "trend_linear_up_seasonal",
             "noise_var_decreasing", "noise_var_increasing",
             "trend_linear_up_amp_increasing",
             "autocorr_shift",
             "trend_piecewise",
             "non_stationary_seasonal_var_increasing",
             "stationary_cycles_var_decreasing"]

    final_df = []
    for case in cases:
        series = generate_synthetic_time_series(n=n, start=start, freq=freq, kind=case)

        series["kind"] = case
        final_df.append(series)

    final_df = pd.concat(final_df)
    final_df.to_csv(Path(get_data_folder("data"), "simple_synthetic_dataset_for_learning.csv"), index=False)

    # Generate common picture
    fig_size = (11.0, 6.0)
    fig, ax = plt.subplots(figsize=fig_size)
    for case in cases:
        case_df = final_df[final_df["kind"] == case]
        ax.plot(case_df["datetime"], case_df["value"], c="grey")

    ax.grid(alpha=0.2)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Value")
    ax.set_ylim(-20, 20)
    plt.savefig(Path(get_data_folder("data"), "all_simple_synthetic_dataset_for_learning.png"),
                bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    generate_dataset()
