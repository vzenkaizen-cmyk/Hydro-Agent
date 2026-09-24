import os
import joblib
import numpy as np
import pandas as pd

DATA_PATH = "data/HydroGen_Model_Ready_Dataset.csv"
MODEL_DIR = "models"

FEATURES = [
    "Rainfall_Input_mm",
    "Lag1_Generation_kWh",
    "Lag7_Generation_kWh",
    "Lag14_Generation_kWh",
    "Lag30_Generation_kWh",
    "Rolling7_Generation_kWh",
    "Rolling30_Generation_kWh",
    "Lag1_Rainfall_mm",
    "Lag7_Rainfall_mm",
    "Lag14_Rainfall_mm",
    "Lag30_Rainfall_mm",
    "Rolling7_Rainfall_mm",
    "Rolling30_Rainfall_mm",
    "Month",
    "DayOfYear",
    "WeekOfYear",
    "Sin_DayOfYear",
    "Cos_DayOfYear",
    "Rainfall_Coverage_30D",
]

def load_history(data_path=DATA_PATH):
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Generation_kWh"] = pd.to_numeric(df["Generation_kWh"], errors="coerce")
    df["Rainfall_Input_mm"] = pd.to_numeric(
        df["Rainfall_Input_mm"], errors="coerce"
    )
    return df.dropna(subset=["Site", "Date", "Generation_kWh"]).sort_values(
        ["Site", "Date"]
    )

def load_model(site):
    path = os.path.join(MODEL_DIR, f"{site}_xgb_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found: {path}. Run train_model.py first."
        )
    return joblib.load(path)

def _value_on_date(history, date, column, fallback=0.0):
    s = history.loc[history["Date"] == pd.Timestamp(date), column]
    if len(s):
        value = s.iloc[-1]
        return float(value) if pd.notna(value) else float(fallback)
    return float(fallback)

def _lag_generation(history, date, days):
    return _value_on_date(
        history, pd.Timestamp(date) - pd.Timedelta(days=days),
        "Generation_kWh", fallback=0.0
    )

def _lag_rainfall(history, date, days, fallback):
    return _value_on_date(
        history, pd.Timestamp(date) - pd.Timedelta(days=days),
        "Rainfall_Input_mm", fallback=fallback
    )

def build_feature_row(history, site, date, rainfall):
    date = pd.Timestamp(date)
    site_history = history[history["Site"] == site].sort_values("Date").copy()

    # IMPORTANT: all history-based features use only dates BEFORE the
    # forecast date. This matches leakage-safe training.
    previous = site_history[site_history["Date"] < date].copy()

    if previous.empty:
        raise ValueError(f"No historical data available for site {site}.")

    last_30 = previous.tail(30)

    rolling7_gen = previous["Generation_kWh"].tail(7).mean()
    rolling30_gen = previous["Generation_kWh"].tail(30).mean()

    rolling7_rain = previous["Rainfall_Input_mm"].tail(7).sum()
    rolling30_rain = previous["Rainfall_Input_mm"].tail(30).sum()

    coverage = previous["Rainfall_Input_mm"].tail(30).notna().mean()

    doy = date.dayofyear

    row = {
        "Rainfall_Input_mm": float(rainfall),
        "Lag1_Generation_kWh": _lag_generation(previous, date, 1),
        "Lag7_Generation_kWh": _lag_generation(previous, date, 7),
        "Lag14_Generation_kWh": _lag_generation(previous, date, 14),
        "Lag30_Generation_kWh": _lag_generation(previous, date, 30),
        "Rolling7_Generation_kWh": float(rolling7_gen),
        "Rolling30_Generation_kWh": float(rolling30_gen),
        "Lag1_Rainfall_mm": _lag_rainfall(previous, date, 1, rainfall),
        "Lag7_Rainfall_mm": _lag_rainfall(previous, date, 7, rainfall),
        "Lag14_Rainfall_mm": _lag_rainfall(previous, date, 14, rainfall),
        "Lag30_Rainfall_mm": _lag_rainfall(previous, date, 30, rainfall),
        "Rolling7_Rainfall_mm": float(rolling7_rain),
        "Rolling30_Rainfall_mm": float(rolling30_rain),
        "Month": int(date.month),
        "DayOfYear": int(doy),
        "WeekOfYear": int(date.isocalendar().week),
        "Sin_DayOfYear": float(np.sin(2 * np.pi * doy / 365.25)),
        "Cos_DayOfYear": float(np.cos(2 * np.pi * doy / 365.25)),
        "Rainfall_Coverage_30D": float(coverage),
    }

    return pd.DataFrame([row], columns=FEATURES)

def forecast_generation(site, weather_daily, data_path=DATA_PATH):
    history = load_history(data_path)
    model = load_model(site)

    site_history = history[history["Site"] == site].copy().sort_values("Date")
    if site_history.empty:
        raise ValueError(f"No historical data for {site}.")

    # Ensure the recursive forecast has the exact last observed rainfall.
    site_history["Rainfall_Input_mm"] = site_history["Rainfall_Input_mm"].fillna(
        0.0
    )

    all_history = history.copy()
    forecasts = []

    for _, w in weather_daily.sort_values("Date").iterrows():
        date = pd.Timestamp(w["Date"])
        rainfall = max(0.0, float(w["Rainfall_Input_mm"]))

        x = build_feature_row(all_history, site, date, rainfall)
        pred = float(model.predict(x)[0])
        pred = max(0.0, pred)

        forecasts.append({
            "Site": site,
            "Date": date,
            "Forecast_Rainfall_mm": rainfall,
            "Rain_Probability": float(w.get("Rain_Probability", np.nan)),
            "Temperature_C": w.get("Temperature_C"),
            "Humidity_pct": w.get("Humidity_pct"),
            "Forecast_Generation_kWh": pred,
            "Forecast_Source": "OpenWeather + XGBoost",
        })

        # Append prediction so the next future day can use it as a lag.
        all_history = pd.concat([
            all_history,
            pd.DataFrame([{
                "Site": site,
                "Date": date,
                "Generation_kWh": pred,
                "Rainfall_Input_mm": rainfall,
            }])
        ], ignore_index=True)

    return pd.DataFrame(forecasts)

def run_hydro_forecast(site, weather_daily, data_path=DATA_PATH):
    result = forecast_generation(site, weather_daily, data_path)
    return result
