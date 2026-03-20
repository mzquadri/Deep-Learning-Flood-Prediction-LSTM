"""
Generate synthetic hydrological data for flood prediction.

Creates realistic daily time series of:
- Precipitation (mm/day)
- Temperature (°C)
- Soil moisture (%)
- River discharge (m³/s) — target variable

The discharge is modeled as a nonlinear function of lagged precipitation,
temperature, and soil moisture, with seasonal patterns and noise,
mimicking a conceptual rainfall-runoff relationship.
"""

import numpy as np
import pandas as pd
import os


def generate_hydrological_data(
    n_years: int = 20,
    seed: int = 42,
    save_path: str = None,
) -> pd.DataFrame:
    """
    Generate synthetic daily hydrological time series.

    Parameters
    ----------
    n_years : int
        Number of years of data to generate.
    seed : int
        Random seed for reproducibility.
    save_path : str, optional
        Path to save the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, precipitation, temperature,
        soil_moisture, discharge.
    """
    rng = np.random.RandomState(seed)
    n_days = n_years * 365

    dates = pd.date_range(start="2000-01-01", periods=n_days, freq="D")
    day_of_year = dates.dayofyear.values.astype(float)

    # --- Precipitation (mm/day) ---
    # Seasonal pattern: wetter in winter/spring, drier in summer
    seasonal_precip = 3.0 + 2.5 * np.sin(2 * np.pi * (day_of_year - 30) / 365)
    # Intermittent rainfall: some days are dry
    rain_occurrence = rng.binomial(1, 0.4, size=n_days)
    rain_intensity = rng.exponential(scale=seasonal_precip)
    precipitation = rain_occurrence * rain_intensity
    # Occasional extreme events
    extreme_mask = rng.binomial(1, 0.005, size=n_days).astype(bool)
    precipitation[extreme_mask] *= rng.uniform(3, 8, size=extreme_mask.sum())
    precipitation = np.maximum(precipitation, 0.0)

    # --- Temperature (°C) ---
    seasonal_temp = 10.0 + 12.0 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
    temperature = seasonal_temp + rng.normal(0, 2.5, size=n_days)

    # --- Soil Moisture (fraction 0-1, stored as %) ---
    soil_moisture = np.zeros(n_days)
    soil_moisture[0] = 0.5
    for t in range(1, n_days):
        # Simple bucket model
        infiltration = 0.6 * precipitation[t] / 100.0
        evapotranspiration = max(0, 0.002 * (temperature[t] + 5))
        drainage = 0.05 * soil_moisture[t - 1]
        soil_moisture[t] = (
            soil_moisture[t - 1] + infiltration - evapotranspiration - drainage
        )
        soil_moisture[t] = np.clip(soil_moisture[t], 0.05, 0.98)
    soil_moisture_pct = soil_moisture * 100.0

    # --- River Discharge (m³/s) ---
    # Nonlinear rainfall-runoff with memory (lagged precipitation)
    discharge = np.zeros(n_days)
    baseflow = 15.0  # m³/s
    for t in range(n_days):
        # Weighted sum of recent precipitation (unit hydrograph concept)
        precip_response = 0.0
        weights = [0.05, 0.10, 0.20, 0.25, 0.18, 0.12, 0.06, 0.03, 0.01]
        for lag, w in enumerate(weights):
            idx = t - lag
            if idx >= 0:
                precip_response += w * precipitation[idx]

        # Nonlinear runoff generation
        runoff_coeff = 0.3 + 0.5 * soil_moisture[t]  # wetter soil -> more runoff
        quickflow = runoff_coeff * precip_response * 2.0

        # Seasonal baseflow variation
        seasonal_base = baseflow + 5.0 * np.sin(2 * np.pi * (day_of_year[t] - 60) / 365)

        # Snowmelt contribution in spring
        snowmelt = 0.0
        if 80 < day_of_year[t] < 160 and temperature[t] > 2:
            snowmelt = 3.0 * max(0, temperature[t] - 2) * 0.3

        discharge[t] = seasonal_base + quickflow + snowmelt
        discharge[t] += rng.normal(0, 1.5)  # measurement noise
        discharge[t] = max(discharge[t], 1.0)

    df = pd.DataFrame(
        {
            "date": dates,
            "precipitation_mm": np.round(precipitation, 2),
            "temperature_c": np.round(temperature, 2),
            "soil_moisture_pct": np.round(soil_moisture_pct, 2),
            "discharge_m3s": np.round(discharge, 2),
        }
    )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved {len(df)} records to {save_path}")

    return df


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    save_path = os.path.join(data_dir, "synthetic_hydrology.csv")
    df = generate_hydrological_data(n_years=20, save_path=save_path)
    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nSummary statistics:")
    print(df.describe().round(2))
