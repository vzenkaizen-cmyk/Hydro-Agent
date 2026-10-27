from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "HydroGen_Model_Ready_Dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
HORIZON = 365

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


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    required = ["Site", "Date", "Generation_kWh", "Rainfall_Input_mm"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing {missing}. Available columns: {list(df.columns)}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Generation_kWh"] = pd.to_numeric(df["Generation_kWh"], errors="coerce")
    df["Rainfall_Input_mm"] = pd.to_numeric(df["Rainfall_Input_mm"], errors="coerce")
    df = df.dropna(subset=["Site", "Date", "Generation_kWh"]).copy()
    df["Site"] = df["Site"].astype(str).str.upper().str.strip()
    return df.sort_values(["Site", "Date"]).drop_duplicates(
        ["Site", "Date"], keep="last"
    ).reset_index(drop=True)


def load_model(site):
    site = site.upper()
    model_path = MODEL_DIR / f"{site}_xgb_model.pkl"
    feature_path = MODEL_DIR / f"{site}_features.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
    model = joblib.load(model_path)
    saved_features = list(joblib.load(feature_path))
    if saved_features != FEATURES:
        raise RuntimeError(
            f"{site}: saved model features do not match this script.\n"
            f"Expected: {FEATURES}\nSaved: {saved_features}\n"
            "Retrain this site with the matching training script."
        )
    return model


def monthly_rainfall(site_df):
    month_values = site_df.assign(Month=site_df["Date"].dt.month).groupby(
        "Month"
    )["Rainfall_Input_mm"].median().to_dict()
    overall = float(site_df["Rainfall_Input_mm"].median())
    if not np.isfinite(overall):
        overall = 0.0
    return {
        m: max(
            0.0,
            float(month_values.get(m, overall))
            if pd.notna(month_values.get(m, overall))
            else overall,
        )
        for m in range(1, 13)
    }


def _safe_lag(series, lag, fallback=0.0):
    value = series.iloc[-lag] if len(series) >= lag else fallback
    return float(value) if pd.notna(value) else float(fallback)


def build_features(generation_history, rainfall_history, date, rainfall):
    g = pd.Series(generation_history, dtype="float64")
    r = pd.Series(rainfall_history, dtype="float64")
    g_shift, r_shift = g.shift(1), r.shift(1)
    fallback_g = float(g.dropna().iloc[0]) if g.notna().any() else 0.0
    nonnull_r = r.dropna()
    fallback_r = float(nonnull_r.iloc[0]) if len(nonnull_r) else 0.0
    row = {
        "Rainfall_Input_mm": float(rainfall) if pd.notna(rainfall) else fallback_r,
        "Lag1_Generation_kWh": _safe_lag(g, 1, fallback_g),
        "Lag7_Generation_kWh": _safe_lag(g, 7, fallback_g),
        "Lag14_Generation_kWh": _safe_lag(g, 14, fallback_g),
        "Lag30_Generation_kWh": _safe_lag(g, 30, fallback_g),
        "Rolling7_Generation_kWh": float(g_shift.tail(7).mean()),
        "Rolling30_Generation_kWh": float(g_shift.tail(30).mean()),
        "Lag1_Rainfall_mm": _safe_lag(r, 1, fallback_r),
        "Lag7_Rainfall_mm": _safe_lag(r, 7, fallback_r),
        "Lag14_Rainfall_mm": _safe_lag(r, 14, fallback_r),
        "Lag30_Rainfall_mm": _safe_lag(r, 30, fallback_r),
        "Rolling7_Rainfall_mm": float(r_shift.tail(7).sum()),
        "Rolling30_Rainfall_mm": float(r_shift.tail(30).sum()),
        "Month": date.month,
        "DayOfYear": date.dayofyear,
        "WeekOfYear": int(date.isocalendar().week),
        "Sin_DayOfYear": np.sin(2 * np.pi * date.dayofyear / 365.25),
        "Cos_DayOfYear": np.cos(2 * np.pi * date.dayofyear / 365.25),
        "Rainfall_Coverage_30D": float(r_shift.tail(30).count() / 30.0),
    }
    X = pd.DataFrame([[row[c] for c in FEATURES]], columns=FEATURES)
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)[FEATURES]


def forecast_site(site, full_df):
    site = site.upper().strip()
    site_df = full_df[full_df["Site"] == site].sort_values("Date").copy()
    if site_df.empty:
        raise ValueError(f"No historical data found for {site}")
    model = load_model(site)
    rainfall_by_month = monthly_rainfall(site_df)
    last_date = site_df["Date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    generation_history = site_df["Generation_kWh"].astype(float).tolist()
    rainfall_history = site_df["Rainfall_Input_mm"].tolist()
    results = []
    for future_date in future_dates:
        rain = rainfall_by_month[future_date.month]
        X = build_features(generation_history, rainfall_history, future_date, rain)
        pred = max(0.0, float(np.asarray(model.predict(X)).reshape(-1)[0]))
        results.append({
            "Site": site,
            "Date": future_date,
            "Forecast_Generation_kWh": pred,
            "Forecast_Rainfall_mm": rain,
            "Forecast_Source": "Historical monthly median rainfall + XGBoost",
        })
        generation_history.append(pred)
        rainfall_history.append(rain)
    return pd.DataFrame(results)


def historical_backtest_site(site, full_df):
    """
    One-day-ahead historical backtest. Each prediction uses only data dated
    before the target date. Actual rainfall on the target date is used, so
    this is a conditional diagnostic backtest, not a simulation of weather
    forecast error. MAPE is omitted for zero actual generation.
    """
    site = site.upper().strip()
    site_df = full_df[full_df["Site"] == site].sort_values("Date").reset_index(drop=True)
    if site_df.empty:
        raise ValueError(f"No historical data found for {site}")
    model = load_model(site)
    results = []
    for i, row in site_df.iterrows():
        # Use only prior actuals; start once there is enough history for lags.
        if i < 30:
            continue
        prior = site_df.iloc[:i]
        rain = row["Rainfall_Input_mm"]
        if pd.isna(rain):
            # Avoid using target-date rainfall if unavailable; use prior month median.
            month_vals = prior.loc[
                prior["Date"].dt.month == row["Date"].month,
                "Rainfall_Input_mm",
            ].dropna()
            prior_median = prior["Rainfall_Input_mm"].median()
            rain = (
                float(month_vals.median())
                if not month_vals.empty
                else float(prior_median) if pd.notna(prior_median) else 0.0
            )
        X = build_features(
            prior["Generation_kWh"].astype(float).tolist(),
            prior["Rainfall_Input_mm"].tolist(),
            row["Date"],
            rain,
        )
        pred = max(0.0, float(np.asarray(model.predict(X)).reshape(-1)[0]))
        actual = float(row["Generation_kWh"])
        results.append({
            "Site": site,
            "Date": row["Date"],
            "Actual_Generation_kWh": actual,
            "Backtest_Forecast_Generation_kWh": pred,
            "Generation_Difference_kWh": pred - actual,
            "Absolute_Error_kWh": abs(pred - actual),
            "Absolute_Percentage_Error": abs(pred - actual) / abs(actual) * 100 if actual != 0 else np.nan,
            "Actual_Rainfall_mm": row["Rainfall_Input_mm"],
            "Backtest_Rainfall_Input_mm": rain,
            "Backtest_Type": "One-day-ahead using prior actual generation",
        })
    return pd.DataFrame(results)


def save_outputs(result, backtest=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = result.copy()
    result["Date"] = pd.to_datetime(result["Date"])
    result.to_csv(OUTPUT_DIR / "next_year_forecast.csv", index=False)

    monthly = result.assign(
        Year=result["Date"].dt.year,
        Month_Number=result["Date"].dt.month,
        Month=result["Date"].dt.strftime("%b"),
    ).groupby(["Site", "Year", "Month_Number", "Month"], as_index=False).agg(
        Forecast_Generation_kWh=("Forecast_Generation_kWh", "sum"),
        Forecast_Rainfall_mm=("Forecast_Rainfall_mm", "sum"),
    ).sort_values(["Site", "Year", "Month_Number"])
    monthly.to_csv(OUTPUT_DIR / "monthly_forecast.csv", index=False)

    yearly = result.groupby("Site", as_index=False).agg(
        Forecast_Days=("Date", "nunique"),
        Total_Forecast_Generation_kWh=("Forecast_Generation_kWh", "sum"),
        Average_Daily_Generation_kWh=("Forecast_Generation_kWh", "mean"),
        Maximum_Daily_Generation_kWh=("Forecast_Generation_kWh", "max"),
        Total_Forecast_Rainfall_mm=("Forecast_Rainfall_mm", "sum"),
    )
    yearly.to_csv(OUTPUT_DIR / "yearly_forecast.csv", index=False)

    if backtest is not None and not backtest.empty:
        backtest.to_csv(OUTPUT_DIR / "historical_backtest.csv", index=False)
        monthly_bt = backtest.assign(
            Year=pd.to_datetime(backtest["Date"]).dt.year,
            Month_Number=pd.to_datetime(backtest["Date"]).dt.month,
            Month=pd.to_datetime(backtest["Date"]).dt.strftime("%b"),
        ).groupby(["Site", "Year", "Month_Number", "Month"], as_index=False).agg(
            Actual_Generation_kWh=("Actual_Generation_kWh", "sum"),
            Backtest_Forecast_Generation_kWh=("Backtest_Forecast_Generation_kWh", "sum"),
            Actual_Rainfall_mm=("Actual_Rainfall_mm", "sum"),
            Backtest_Rainfall_Input_mm=("Backtest_Rainfall_Input_mm", "sum"),
            MAE_kWh=("Absolute_Error_kWh", "mean"),
            MAPE_Percent=("Absolute_Percentage_Error", "mean"),
        ).sort_values(["Site", "Year", "Month_Number"])
        monthly_bt.to_csv(OUTPUT_DIR / "monthly_historical_comparison.csv", index=False)

    print("FORECAST COMPLETE")
    print(f"Saved: {OUTPUT_DIR / 'next_year_forecast.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'monthly_forecast.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'yearly_forecast.csv'}")
    if backtest is not None and not backtest.empty:
        print(f"Saved: {OUTPUT_DIR / 'historical_backtest.csv'}")
        print(f"Saved: {OUTPUT_DIR / 'monthly_historical_comparison.csv'}")


def main():
    full_df = load_data()
    requested_site = None
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        requested_site = sys.argv[1].upper().strip()
    sites = [requested_site] if requested_site else sorted(full_df["Site"].unique())
    results, backtests, errors = [], [], []
    for site in sites:
        try:
            results.append(forecast_site(site, full_df))
            bt = historical_backtest_site(site, full_df)
            if not bt.empty:
                backtests.append(bt)
            print(f"[OK] {site}: forecast and historical backtest generated")
        except Exception as exc:
            errors.append(f"{site}: {exc}")
            print(f"[ERROR] {site}: {exc}")
    if not results:
        raise RuntimeError("No forecasts were generated.\n" + "\n".join(errors))
    save_outputs(
        pd.concat(results, ignore_index=True),
        pd.concat(backtests, ignore_index=True) if backtests else pd.DataFrame(),
    )
    if errors:
        print("WARNING - failed sites:")
        for error in errors:
            print(" -", error)


if __name__ == "__main__":
    main()
