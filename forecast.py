
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

# EXACTLY the same feature list and definitions used by train_model.py
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
        raise ValueError(
            f"Dataset is missing {missing}. Available columns: {list(df.columns)}"
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Generation_kWh"] = pd.to_numeric(df["Generation_kWh"], errors="coerce")
    df["Rainfall_Input_mm"] = pd.to_numeric(df["Rainfall_Input_mm"], errors="coerce")
    df = df.dropna(subset=["Site", "Date", "Generation_kWh"]).copy()
    df["Site"] = df["Site"].astype(str).str.upper().str.strip()
    return (
        df.sort_values(["Site", "Date"])
        .drop_duplicates(["Site", "Date"], keep="last")
        .reset_index(drop=True)
    )


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
            f"{site}: saved model features do not match train_model.py.\n"
            f"Expected: {FEATURES}\n"
            f"Saved: {saved_features}\n"
            "Retrain this site with the supplied train_model.py."
        )
    return model


def monthly_rainfall(site_df):
    # Long-term rainfall scenario: historical median for each calendar month.
    month_values = (
        site_df.assign(Month=site_df["Date"].dt.month)
        .groupby("Month")["Rainfall_Input_mm"]
        .median()
        .to_dict()
    )
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
    if len(series) >= lag:
        value = series.iloc[-lag]
    else:
        value = fallback
    return float(value) if pd.notna(value) else float(fallback)


def build_features(generation_history, rainfall_history, date, rainfall):
    """
    Exact equivalent of train_model.py prepare() for one future row.

    Training uses:
      generation.shift(1).rolling(7).mean()
      rainfall.shift(1).rolling(7).sum()
      rainfall.shift(1).rolling(30).count() / 30

    The histories passed here contain only values before `date`, so the
    current future generation is never leaked into its own features.
    """
    g = pd.Series(generation_history, dtype="float64")
    r = pd.Series(rainfall_history, dtype="float64")

    g_shift = g.shift(1)
    r_shift = r.shift(1)

    # For a future row, rainfall input is the current scenario value.
    # All lag/rolling/coverage features use previous observations only.
    fallback_g = float(g.iloc[0]) if len(g) else 0.0
    nonnull_r = r.dropna()
    fallback_r = float(nonnull_r.iloc[0]) if len(nonnull_r) else 0.0

    row = {
        "Rainfall_Input_mm": float(rainfall),

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

        # EXACTLY matches:
        # s.shift(1).rolling(30, min_periods=1).count() / 30
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

    future_dates = pd.date_range(
        last_date + pd.Timedelta(days=1),
        periods=HORIZON,
        freq="D",
    )

    generation_history = site_df["Generation_kWh"].astype(float).tolist()
    rainfall_history = site_df["Rainfall_Input_mm"].tolist()
    results = []

    print("=" * 72)
    print(f"FORECASTING {site}")
    print(f"Last actual date: {last_date.date()}")
    print(f"Forecast: {future_dates[0].date()} -> {future_dates[-1].date()}")
    print(f"Features: {len(FEATURES)}")
    print("=" * 72)

    for i, future_date in enumerate(future_dates, 1):
        rain = rainfall_by_month[future_date.month]

        X = build_features(
            generation_history,
            rainfall_history,
            future_date,
            rain,
        )

        pred = float(np.asarray(model.predict(X)).reshape(-1)[0])
        pred = max(0.0, pred)

        results.append({
            "Site": site,
            "Date": future_date,
            "Forecast_Generation_kWh": pred,
            "Forecast_Rainfall_mm": rain,
            "Forecast_Source": "Historical monthly median rainfall + XGBoost",
        })

        # Recursive prediction: this new value becomes history for later days.
        generation_history.append(pred)
        rainfall_history.append(rain)

        if i in (1, 7, 30, 90, 180, 365):
            print(
                f"[OK] {future_date.date()} | "
                f"Rain={rain:.2f} mm | Generation={pred:,.0f} kWh"
            )

    return pd.DataFrame(results)


def save_outputs(result):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = result.copy()
    result["Date"] = pd.to_datetime(result["Date"])

    result.to_csv(OUTPUT_DIR / "next_year_forecast.csv", index=False)

    monthly = (
        result.assign(
            Year=result["Date"].dt.year,
            Month_Number=result["Date"].dt.month,
            Month=result["Date"].dt.strftime("%b"),
        )
        .groupby(["Site", "Year", "Month_Number", "Month"], as_index=False)
        .agg(
            Forecast_Generation_kWh=("Forecast_Generation_kWh", "sum"),
            Forecast_Rainfall_mm=("Forecast_Rainfall_mm", "sum"),
        )
        .sort_values(["Site", "Year", "Month_Number"])
    )
    monthly.to_csv(OUTPUT_DIR / "monthly_forecast.csv", index=False)

    yearly = (
        result.groupby("Site", as_index=False)
        .agg(
            Forecast_Days=("Date", "nunique"),
            Total_Forecast_Generation_kWh=("Forecast_Generation_kWh", "sum"),
            Average_Daily_Generation_kWh=("Forecast_Generation_kWh", "mean"),
            Maximum_Daily_Generation_kWh=("Forecast_Generation_kWh", "max"),
            Total_Forecast_Rainfall_mm=("Forecast_Rainfall_mm", "sum"),
        )
    )
    yearly.to_csv(OUTPUT_DIR / "yearly_forecast.csv", index=False)

    print("\nFORECAST COMPLETE")
    print(f"Saved: {OUTPUT_DIR / 'next_year_forecast.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'monthly_forecast.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'yearly_forecast.csv'}")


def main():
    full_df = load_data()

    if len(sys.argv) > 1:
        sites = [sys.argv[1].upper().strip()]
    else:
        sites = sorted(full_df["Site"].unique())

    results = []
    errors = []

    for site in sites:
        try:
            results.append(forecast_site(site, full_df))
        except Exception as exc:
            errors.append(f"{site}: {exc}")
            print(f"[ERROR] {site}: {exc}")

    if not results:
        raise RuntimeError(
            "No forecasts were generated.\n" + "\n".join(errors)
        )

    save_outputs(pd.concat(results, ignore_index=True))

    if errors:
        print("\nWARNING - failed sites:")
        for error in errors:
            print(" -", error)


if __name__ == "__main__":
    main()
