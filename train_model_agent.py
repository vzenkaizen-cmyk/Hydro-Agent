import os
import joblib
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
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

# Fast, highly accurate XGBoost settings
TUNED_PARAMETERS = {
    "n_estimators": 1000,
    "learning_rate": 0.03,        # Faster convergence
    "max_depth": 5,
    "min_child_weight": 2,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.01,
    "reg_lambda": 0.5,
    "objective": "reg:squarederror",
    "random_state": 42,
    "tree_method": "hist",        # Fast histogram binning
    "n_jobs": 1,                  # 1 job per site process for optimal multiprocessing
    "early_stopping_rounds": 30,  # Fast stop when validation score plateaus
}

def prepare(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Generation_kWh"] = pd.to_numeric(df["Generation_kWh"], errors="coerce")
    df["Rainfall_Input_mm"] = pd.to_numeric(df["Rainfall_Input_mm"], errors="coerce")
    df = df.dropna(subset=["Site", "Date", "Generation_kWh"])
    df = df.sort_values(["Site", "Date"])

    g = df.groupby("Site", group_keys=False)
    
    # Generation Lags
    df["Lag1_Generation_kWh"] = g["Generation_kWh"].shift(1)
    df["Lag7_Generation_kWh"] = g["Generation_kWh"].shift(7)
    df["Lag14_Generation_kWh"] = g["Generation_kWh"].shift(14)
    df["Lag30_Generation_kWh"] = g["Generation_kWh"].shift(30)

    # Shift BEFORE rolling to prevent target leakage (Fast Vectorized Pandas)
    shifted_gen = g["Generation_kWh"].shift(1)
    df["Rolling7_Generation_kWh"] = shifted_gen.groupby(df["Site"]).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
    df["Rolling30_Generation_kWh"] = shifted_gen.groupby(df["Site"]).rolling(30, min_periods=1).mean().reset_index(level=0, drop=True)

    # Rainfall Lags
    df["Lag1_Rainfall_mm"] = g["Rainfall_Input_mm"].shift(1)
    df["Lag7_Rainfall_mm"] = g["Rainfall_Input_mm"].shift(7)
    df["Lag14_Rainfall_mm"] = g["Rainfall_Input_mm"].shift(14)
    df["Lag30_Rainfall_mm"] = g["Rainfall_Input_mm"].shift(30)

    shifted_rain = g["Rainfall_Input_mm"].shift(1)
    df["Rolling7_Rainfall_mm"] = shifted_rain.groupby(df["Site"]).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
    df["Rolling30_Rainfall_mm"] = shifted_rain.groupby(df["Site"]).rolling(30, min_periods=1).mean().reset_index(level=0, drop=True)

    # Cyclical Features
    df["Month"] = df["Date"].dt.month
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

    doy = df["DayOfYear"]
    df["Sin_DayOfYear"] = np.sin(2 * np.pi * doy / 365.25)
    df["Cos_DayOfYear"] = np.cos(2 * np.pi * doy / 365.25)

    df["Rainfall_Coverage_30D"] = (
        shifted_rain.groupby(df["Site"]).rolling(30, min_periods=1).count().reset_index(level=0, drop=True) / 30.0
    )

    return df

def train_site(site, site_df):
    site_df = site_df.dropna(subset=FEATURES).sort_values("Date")

    if len(site_df) < 100:
        print(f"[SKIP] {site}: only {len(site_df)} usable rows")
        return None

    # Temporal Split: 70% Train / 10% Val / 20% Test
    n = len(site_df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.80)

    train_df = site_df.iloc[:train_end]
    val_df = site_df.iloc[train_end:val_end]
    test_df = site_df.iloc[val_end:]

    # Early stopping evaluation model
    model = XGBRegressor(**TUNED_PARAMETERS)
    model.fit(
        train_df[FEATURES],
        train_df["Generation_kWh"],
        eval_set=[(val_df[FEATURES], val_df["Generation_kWh"])],
        verbose=False,
    )

    pred = np.maximum(0, model.predict(test_df[FEATURES]))
    actual = test_df["Generation_kWh"].to_numpy()

    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred)
    mape = np.mean(
        np.abs((actual - pred) / np.where(actual == 0, np.nan, actual))
    ) * 100

    # Retrain final model using optimal tree count directly
    best_trees = getattr(model, "best_iteration", 500)
    final_params = {k: v for k, v in TUNED_PARAMETERS.items() if k != "early_stopping_rounds"}
    final_params["n_estimators"] = max(best_trees, 50)

    final_model = XGBRegressor(**final_params)
    final_model.fit(site_df[FEATURES], site_df["Generation_kWh"])

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        "Best_Trees": best_trees,
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

    sites = sorted(df["Site"].dropna().unique())
    results = []

    # Parallelize training across all CPU cores
    max_workers = min(len(sites), os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(train_site, site, df[df["Site"] == site].copy()): site
            for site in sites
        }
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)

    pd.DataFrame(results).to_csv(
        os.path.join(OUTPUT_DIR, "model_performance_agent.csv"),
        index=False
    )

    print("\nTraining completed fast & successfully.")
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    main()
