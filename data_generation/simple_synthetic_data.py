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
                                   freq: str = "D", kind: str = "stationary_noise") -> pd.DataFrame:
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
        - "stationary_noise"          : White noise, no trend, no seasonality, no cycles.
        - "stationary_cycles"         : Stationary with damped cycles (AR-like).
        - "trend_linear_up"           : Non-stationary with upward linear trend.
        - "trend_linear_down"         : Non-stationary with downward linear trend.
        - "trend_exponential_up"      : Non-stationary with exponential upward trend.
        - "non_stationary_seasonal"   : Non-stationary with seasonality (sine wave + noise).
    """
    # Time index
    datetime_index = pd.date_range(start=start, periods=n, freq=freq)
    t = np.arange(n)

    # Base noise
    noise = np.random.normal(0, NOISE_COMPONENT, n)

    if kind == "stationary_noise":
        values = noise

    elif kind == "stationary_cycles":
        # Damped oscillation (like AR process with complex roots)
        values = np.cos(2 * np.pi * t / 20) * np.exp(-t / 100) + noise * 0.5

    elif kind == "non_stationary_seasonal":
        values = np.sin(2 * np.pi * t / 12) + noise  # fixed amplitude sine wave + noise

    elif kind == "trend_linear_up":
        values = 0.1 * t + noise

    elif kind == "trend_linear_down":
        values = -0.1 * t + noise

    elif kind == "trend_exponential_up":
        values = np.exp(0.02 * t) + noise

    elif kind == "trend_exponential_down":
        values = np.exp(-0.02 * t) + noise

    else:
        raise ValueError(f"Unknown kind {kind}. Choose from predefined options!")

    return pd.DataFrame({"datetime": datetime_index, "value": values})


def generate_dataset(n: int = 100,
                     start: str = "2025-01-01",
                     freq: str = "D"):
    cases = ["stationary_noise", "stationary_cycles", "non_stationary_seasonal",
             "trend_linear_up", "trend_linear_down", "trend_exponential_up"]

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
        ax.plot(case_df["datetime"], case_df["value"], c='grey')

    ax.grid(alpha=0.2)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Value")
    plt.savefig(Path(get_data_folder("data"), "all_simple_synthetic_dataset_for_learning.png"),
                bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    generate_dataset()
