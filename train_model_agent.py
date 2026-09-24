import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

DATA_PATH = "data/HydroGen_Model_Ready_Dataset.csv"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

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

PARAMETERS = {
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "max_depth": 5,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}

def prepare(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Generation_kWh"] = pd.to_numeric(df["Generation_kWh"], errors="coerce")
    df["Rainfall_Input_mm"] = pd.to_numeric(df["Rainfall_Input_mm"], errors="coerce")
    df = df.dropna(subset=["Site", "Date", "Generation_kWh"])
    df = df.sort_values(["Site", "Date"])

    # Rebuild features so training and future inference use exactly the same logic.
    g = df.groupby("Site", group_keys=False)
    df["Lag1_Generation_kWh"] = g["Generation_kWh"].shift(1)
    df["Lag7_Generation_kWh"] = g["Generation_kWh"].shift(7)
    df["Lag14_Generation_kWh"] = g["Generation_kWh"].shift(14)
    df["Lag30_Generation_kWh"] = g["Generation_kWh"].shift(30)

    # Leakage-safe: shift BEFORE rolling.
    shifted_g = g["Generation_kWh"].shift(1)
    df["Rolling7_Generation_kWh"] = shifted_g.groupby(df["Site"]).transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    df["Rolling30_Generation_kWh"] = shifted_g.groupby(df["Site"]).transform(
        lambda s: s.rolling(30, min_periods=1).mean()
    )

    df["Lag1_Rainfall_mm"] = g["Rainfall_Input_mm"].shift(1)
    df["Lag7_Rainfall_mm"] = g["Rainfall_Input_mm"].shift(7)
    df["Lag14_Rainfall_mm"] = g["Rainfall_Input_mm"].shift(14)
    df["Lag30_Rainfall_mm"] = g["Rainfall_Input_mm"].shift(30)

    shifted_r = g["Rainfall_Input_mm"].shift(1)
    df["Rolling7_Rainfall_mm"] = shifted_r.groupby(df["Site"]).transform(
        lambda s: s.rolling(7, min_periods=1).sum()
    )
    df["Rolling30_Rainfall_mm"] = shifted_r.groupby(df["Site"]).transform(
        lambda s: s.rolling(30, min_periods=1).sum()
    )

    df["Month"] = df["Date"].dt.month
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

    doy = df["DayOfYear"]
    df["Sin_DayOfYear"] = np.sin(2 * np.pi * doy / 365.25)
    df["Cos_DayOfYear"] = np.cos(2 * np.pi * doy / 365.25)

    df["Rainfall_Coverage_30D"] = (
        df.groupby("Site")["Rainfall_Input_mm"]
        .transform(lambda s: s.shift(1).rolling(30, min_periods=1).count() / 30)
    )

    return df

def train_site(site, df):
    site_df = df[df["Site"] == site].dropna(subset=FEATURES).copy()
    site_df = site_df.sort_values("Date")

    if len(site_df) < 100:
        print(f"[SKIP] {site}: only {len(site_df)} usable rows")
        return None

    split = int(len(site_df) * 0.80)
    train_df = site_df.iloc[:split]
    test_df = site_df.iloc[split:]

    model = XGBRegressor(**PARAMETERS)
    model.fit(train_df[FEATURES], train_df["Generation_kWh"])

    pred = np.maximum(0, model.predict(test_df[FEATURES]))
    actual = test_df["Generation_kWh"].to_numpy()

    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred)
    mape = np.mean(
        np.abs((actual - pred) / np.where(actual == 0, np.nan, actual))
    ) * 100

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Refit final model on all usable historical data.
    final_model = XGBRegressor(**PARAMETERS)
    final_model.fit(site_df[FEATURES], site_df["Generation_kWh"])

    joblib.dump(final_model, os.path.join(MODEL_DIR, f"{site}_xgb_model.pkl"))
    joblib.dump(FEATURES, os.path.join(MODEL_DIR, f"{site}_features.pkl"))

    pd.DataFrame({
        "Site": site,
        "Date": test_df["Date"],
        "Actual_Generation_kWh": actual,
        "Predicted_Generation_kWh": pred,
    }).to_csv(
        os.path.join(OUTPUT_DIR, f"{site}_test_predictions.csv"),
        index=False
    )

    return {
        "Site": site,
        "Rows": len(site_df),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE_pct": mape,
    }

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    raw = pd.read_csv(DATA_PATH)
    df = prepare(raw)

    results = []
    for site in sorted(df["Site"].dropna().unique()):
        result = train_site(site, df)
        if result:
            results.append(result)

    pd.DataFrame(results).to_csv(
        os.path.join(OUTPUT_DIR, "model_performance_agent.csv"),
        index=False
    )

    print("\nTraining completed.")
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    main()
