from __future__ import annotations

import os
import warnings
from pathlib import Path

# Limit native numerical-library threads to avoid CPU oversubscription.
CPU_THREADS = max(1, min(4, os.cpu_count() or 1))
os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "HydroGen_Model_Ready_Dataset.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
MINIMUM_USABLE_ROWS = 120

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

MAX_ESTIMATORS = 500
EARLY_STOPPING_ROUNDS = 25

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

MODEL_PARAMETERS = {
    "n_estimators": MAX_ESTIMATORS,
    "learning_rate": 0.04,
    "max_depth": 4,
    "min_child_weight": 3,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.05,
    "reg_lambda": 2.0,
    "gamma": 0.05,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": CPU_THREADS,
    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
}


# ============================================================
# DATA PREPARATION AND LEAKAGE-SAFE FEATURES
# ============================================================

def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean records and create lagged features using past data only."""

    required = [
        "Site",
        "Date",
        "Generation_kWh",
        "Rainfall_Input_mm",
    ]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")

    df = raw.copy()
    df["Site"] = df["Site"].astype("string").str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Generation_kWh"] = pd.to_numeric(
        df["Generation_kWh"], errors="coerce"
    )
    df["Rainfall_Input_mm"] = pd.to_numeric(
        df["Rainfall_Input_mm"], errors="coerce"
    )

    df = df.dropna(
        subset=["Site", "Date", "Generation_kWh"]
    )
    df = df[df["Site"].ne("")].copy()

    # Negative generation/rainfall values are invalid physical measurements.
    df = df[df["Generation_kWh"] >= 0].copy()
    df = df[
        df["Rainfall_Input_mm"].isna()
        | (df["Rainfall_Input_mm"] >= 0)
    ].copy()

    df["Date"] = df["Date"].dt.normalize()
    df = df.sort_values(["Site", "Date"]).reset_index(drop=True)

    duplicates = df.duplicated(["Site", "Date"], keep=False)
    if duplicates.any():
        examples = (
            df.loc[duplicates, ["Site", "Date"]]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            "Duplicate Site-Date records were found. Resolve or aggregate "
            f"them before training. Examples: {examples}"
        )

    grouped = df.groupby("Site", sort=False)

    # Generation lags: only previous observations are used.
    for lag in (1, 7, 14, 30):
        df[f"Lag{lag}_Generation_kWh"] = (
            grouped["Generation_kWh"].shift(lag)
        )

    shifted_generation = grouped["Generation_kWh"].shift(1)

    for window in (7, 30):
        df[f"Rolling{window}_Generation_kWh"] = (
            shifted_generation.groupby(df["Site"], sort=False)
            .transform(lambda values: values.rolling(window, min_periods=window).mean())
        )

    # Rainfall lags and rolling features are also shifted by one record.
    for lag in (1, 7, 14, 30):
        df[f"Lag{lag}_Rainfall_mm"] = (
            grouped["Rainfall_Input_mm"].shift(lag)
        )

    shifted_rainfall = grouped["Rainfall_Input_mm"].shift(1)

    for window in (7, 30):
        df[f"Rolling{window}_Rainfall_mm"] = (
            shifted_rainfall.groupby(df["Site"], sort=False)
            .transform(lambda values: values.rolling(window, min_periods=window).mean())
        )

    # Coverage represents the fraction of the previous 30 records with rainfall data.
    df["Rainfall_Coverage_30D"] = (
        shifted_rainfall.groupby(df["Site"], sort=False)
        .transform(lambda values: values.rolling(30, min_periods=1).count())
        / 30.0
    )

    # Calendar features.
    df["Month"] = df["Date"].dt.month.astype("int16")
    df["DayOfYear"] = df["Date"].dt.dayofyear.astype("int16")
    df["WeekOfYear"] = (
        df["Date"].dt.isocalendar().week.astype("int16")
    )

    day_of_year = df["DayOfYear"].astype(float)
    df["Sin_DayOfYear"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["Cos_DayOfYear"] = np.cos(2 * np.pi * day_of_year / 365.25)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))

    if len(actual) >= 2 and np.var(actual) > 0:
        r2 = float(r2_score(actual, predicted))
    else:
        r2 = np.nan

    # MAPE is undefined when actual generation is zero.
    positive_actual = actual > 0
    if positive_actual.any():
        mape = float(
            np.mean(
                np.abs(
                    (actual[positive_actual] - predicted[positive_actual])
                    / actual[positive_actual]
                )
            ) * 100
        )
    else:
        mape = np.nan

    return {
        "MAE": float(mae),
        "RMSE": rmse,
        "R2": r2,
        "MAPE_pct": mape,
    }


# ============================================================
# XGBOOST COMPATIBILITY
# ============================================================

def fit_with_early_stopping(model, X_train, y_train, X_val, y_val):
    """
    Support current XGBoost versions (early_stopping_rounds in constructor)
    and older versions (early_stopping_rounds in fit()).
    """
    try:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    except TypeError as error:
        if "early_stopping_rounds" not in str(error):
            raise

        params = MODEL_PARAMETERS.copy()
        params.pop("early_stopping_rounds", None)
        model = XGBRegressor(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=False,
        )

    return model


def predict_best_iteration(model, X):
    """Use the selected best iteration where the installed XGBoost supports it."""
    best_iteration = getattr(model, "best_iteration", None)

    if best_iteration is not None:
        try:
            return model.predict(X, iteration_range=(0, int(best_iteration) + 1))
        except TypeError:
            try:
                return model.predict(X, ntree_limit=int(best_iteration) + 1)
            except TypeError:
                pass

    return model.predict(X)


# ============================================================
# TRAIN ONE PLANT
# ============================================================

def train_site(site: str, site_df: pd.DataFrame):
    print(f"\n[START] Training {site}")

    site_df = (
        site_df.sort_values("Date")
        .dropna(subset=FEATURES + ["Generation_kWh"])
        .reset_index(drop=True)
    )

    n_rows = len(site_df)
    if n_rows < MINIMUM_USABLE_ROWS:
        print(f"[SKIP] {site}: only {n_rows} usable rows.")
        return None

    train_end = int(n_rows * TRAIN_RATIO)
    validation_end = int(n_rows * (TRAIN_RATIO + VALIDATION_RATIO))

    train_df = site_df.iloc[:train_end]
    validation_df = site_df.iloc[train_end:validation_end]
    test_df = site_df.iloc[validation_end:]

    if min(len(train_df), len(validation_df), len(test_df)) < 10:
        print(f"[SKIP] {site}: insufficient rows in a temporal split.")
        return None

    X_train = train_df[FEATURES]
    y_train = train_df["Generation_kWh"]
    X_val = validation_df[FEATURES]
    y_val = validation_df["Generation_kWh"]
    X_test = test_df[FEATURES]
    y_test = test_df["Generation_kWh"]

    # Fit only on the earlier training period and select trees on validation.
    early_model = XGBRegressor(**MODEL_PARAMETERS)
    early_model = fit_with_early_stopping(
        early_model, X_train, y_train, X_val, y_val
    )

    best_iteration = getattr(early_model, "best_iteration", None)
    if best_iteration is None:
        best_trees = MAX_ESTIMATORS
    else:
        best_trees = max(1, int(best_iteration) + 1)

    # Evaluate once on the untouched final chronological test period.
    test_predictions = np.maximum(
        0.0, predict_best_iteration(early_model, X_test)
    )
    metrics = calculate_metrics(y_test.to_numpy(), test_predictions)

    prediction_output = pd.DataFrame({
        "Site": site,
        "Date": test_df["Date"].to_numpy(),
        "Actual_Generation_kWh": y_test.to_numpy(),
        "Predicted_Generation_kWh": test_predictions,
        "Absolute_Error_kWh": np.abs(
            y_test.to_numpy() - test_predictions
        ),
    })

    actual_values = y_test.to_numpy(dtype=float)
    prediction_output["Error_Percentage"] = np.where(
        actual_values > 0,
        prediction_output["Absolute_Error_kWh"]
        / np.where(actual_values > 0, actual_values, np.nan) * 100,
        np.nan,
    )

    prediction_output.to_csv(
        OUTPUT_DIR / f"{site}_test_predictions.csv",
        index=False,
    )

    # Train the deployable model on all available historical rows.
    # Use the tree count selected without looking at the test set.
    final_params = MODEL_PARAMETERS.copy()
    final_params.pop("early_stopping_rounds", None)
    final_params["n_estimators"] = best_trees

    final_model = XGBRegressor(**final_params)
    final_model.fit(
        site_df[FEATURES],
        site_df["Generation_kWh"],
        verbose=False,
    )

    joblib.dump(
        final_model,
        MODEL_DIR / f"{site}_xgb_model.pkl",
        compress=3,
    )
    joblib.dump(
        FEATURES,
        MODEL_DIR / f"{site}_features.pkl",
    )

    metadata = {
        "Site": site,
        "Features": FEATURES,
        "Best_Trees": best_trees,
        "Training_Rows": len(train_df),
        "Validation_Rows": len(validation_df),
        "Test_Rows": len(test_df),
        "Total_Usable_Rows": n_rows,
        "Last_Training_Date": str(site_df["Date"].max().date()),
        "Validation_Method": "Chronological 70/15/15",
        "Metrics": metrics,
    }
    joblib.dump(metadata, MODEL_DIR / f"{site}_metadata.pkl")

    result = {
        "Site": site,
        "Rows": n_rows,
        "Best_Trees": best_trees,
        **metrics,
    }

    mape_text = (
        f"{metrics['MAPE_pct']:.2f}%"
        if np.isfinite(metrics["MAPE_pct"])
        else "N/A"
    )
    print(
        f"[DONE] {site} | trees={best_trees} | "
        f"MAE={metrics['MAE']:.2f} | "
        f"RMSE={metrics['RMSE']:.2f} | "
        f"R2={metrics['R2']:.4f} | MAPE={mape_text}"
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("HYDRO INTELLIGENCE AGENT - MODEL TRAINING")
    print(f"CPU threads per model: {CPU_THREADS}")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    raw = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(raw):,} raw records.")

    df = prepare_data(raw)
    sites = sorted(df["Site"].dropna().unique().tolist())

    if not sites:
        raise RuntimeError("No valid plant/site records were found.")

    print(f"Plants found: {', '.join(sites)}")

    results = []
    failures = []

    # Sequential training is intentional for small cloud instances.
    # XGBoost itself uses a controlled number of CPU threads.
    for site in sites:
        try:
            result = train_site(
                site,
                df.loc[df["Site"] == site].copy(),
            )
            if result is not None:
                results.append(result)
        except Exception as error:
            failures.append({"Site": site, "Error": str(error)})
            print(f"[ERROR] {site}: {error}")

    if results:
        performance_df = pd.DataFrame(results)
        performance_df.to_csv(
            OUTPUT_DIR / "model_performance_agent.csv",
            index=False,
        )
        print("\nMODEL PERFORMANCE")
        print(performance_df.to_string(index=False))
    else:
        raise RuntimeError(
            "No plant models were trained. Check data quality, row counts, "
            "and the errors printed above."
        )

    if failures:
        pd.DataFrame(failures).to_csv(
            OUTPUT_DIR / "model_training_errors.csv",
            index=False,
        )
        print(
            f"\nCompleted with {len(failures)} plant training failure(s). "
            "See outputs/model_training_errors.csv."
        )
    else:
        print("\nAll eligible plant models trained successfully.")

    print(f"Models saved in: {MODEL_DIR}")
    print(f"Reports saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
