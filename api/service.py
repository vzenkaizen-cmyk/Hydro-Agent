from __future__ import annotations
from pathlib import Path
from datetime import date
from functools import lru_cache
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import json

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "HydroGen_Model_Ready_Dataset.csv"
FEATURES = [
    "Rainfall_Input_mm", "Lag1_Generation_kWh", "Lag7_Generation_kWh",
    "Lag14_Generation_kWh", "Lag30_Generation_kWh",
    "Rolling7_Generation_kWh", "Rolling30_Generation_kWh",
    "Lag1_Rainfall_mm", "Lag7_Rainfall_mm", "Lag14_Rainfall_mm",
    "Lag30_Rainfall_mm", "Rolling7_Rainfall_mm", "Rolling30_Rainfall_mm",
    "Month", "DayOfYear", "WeekOfYear", "Sin_DayOfYear",
    "Cos_DayOfYear", "Rainfall_Coverage_30D",
]

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f'Dataset not found: {DATA_PATH}')
    data = pd.read_csv(DATA_PATH)
    required = {'Site', 'Date', 'Generation_kWh', 'Rainfall_Input_mm'}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data['Generation_kWh'] = pd.to_numeric(data['Generation_kWh'], errors='coerce')
    data['Rainfall_Input_mm'] = pd.to_numeric(data['Rainfall_Input_mm'], errors='coerce')
    data = data.dropna(subset=['Site', 'Date', 'Generation_kWh']).copy()
    data['Site'] = data['Site'].astype(str).str.strip()
    data['Rainfall_Input_mm'] = data['Rainfall_Input_mm'].clip(lower=0)
    data = data.sort_values(['Site', 'Date']).drop_duplicates(['Site', 'Date'], keep='last').reset_index(drop=True)
    return data

def train_site_model(site: str, data_signature: str) -> dict:
    """
    Chronological model selection:
      70% train, 15% validation, 15% final test.
    The final test is not used for hyperparameter selection.
    A separate production model is then fitted on all eligible historical rows.
    """
    data = load_data()
    site_data = data.loc[data['Site'].eq(site)].sort_values('Date').copy()
    missing_features = [c for c in FEATURES if c not in site_data.columns]
    if missing_features:
        raise ValueError('Dataset is missing model features: ' + ', '.join(missing_features))
    if 'Model_Ready' in site_data.columns:
        ready = site_data['Model_Ready'].astype(str).str.strip().str.lower()
        site_data = site_data.loc[ready.isin(['true', '1', 'yes', 'y'])].copy()
    numeric_cols = FEATURES + ['Generation_kWh']
    for col in numeric_cols:
        site_data[col] = pd.to_numeric(site_data[col], errors='coerce')
    site_data = site_data.replace([np.inf, -np.inf], np.nan)
    site_data = site_data.dropna(subset=numeric_cols).sort_values('Date').reset_index(drop=True)
    if len(site_data) < 100:
        raise ValueError(f'{site} has only {len(site_data)} model-ready rows. At least 100 rows are recommended for chronological train/validation/test evaluation.')
    X = site_data[FEATURES].astype(float)
    y = site_data['Generation_kWh'].astype(float)
    n = len(site_data)
    train_end = int(n * 0.7)
    validation_end = int(n * 0.85)
    if train_end < 30 or validation_end <= train_end or validation_end >= n:
        raise ValueError('Not enough rows to create chronological train/validation/test sets.')
    X_train, y_train = (X.iloc[:train_end], y.iloc[:train_end])
    X_val, y_val = (X.iloc[train_end:validation_end], y.iloc[train_end:validation_end])
    X_test, y_test = (X.iloc[validation_end:], y.iloc[validation_end:])
    candidates = [{'n_estimators': 350, 'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.85, 'colsample_bytree': 0.85, 'reg_lambda': 5.0}, {'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.025, 'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_lambda': 8.0}, {'n_estimators': 350, 'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.85, 'colsample_bytree': 0.85, 'reg_lambda': 8.0}]
    best_params = None
    best_val_mae = float('inf')
    for params in candidates:
        candidate = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1, **params)
        candidate.fit(X_train, y_train, verbose=False)
        val_pred = np.maximum(candidate.predict(X_val), 0)
        val_mae = float(np.mean(np.abs(y_val.to_numpy() - val_pred)))
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_params = params
    test_model = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1, **best_params)
    test_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]), verbose=False)
    test_prediction = np.maximum(test_model.predict(X_test), 0)
    test_results = site_data.iloc[validation_end:].copy().reset_index(drop=True)
    test_results['Actual_Generation_kWh'] = y_test.to_numpy()
    test_results['Forecast_Generation_kWh'] = test_prediction
    test_results['Generation_Difference_kWh'] = test_results['Forecast_Generation_kWh'] - test_results['Actual_Generation_kWh']
    test_results['Absolute_Error_kWh'] = test_results['Generation_Difference_kWh'].abs()
    actual_denominator = test_results['Actual_Generation_kWh'].abs().replace(0, np.nan)
    test_results['Error_Percentage'] = 100 * test_results['Absolute_Error_kWh'] / actual_denominator
    test_results['Actual_Rainfall_mm'] = test_results['Rainfall_Input_mm']
    actual = test_results['Actual_Generation_kWh'].to_numpy()
    predicted = test_results['Forecast_Generation_kWh'].to_numpy()
    errors = predicted - actual
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    nonzero = np.abs(actual) > 0
    mape = float(np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100) if nonzero.any() else np.nan
    wape_denominator = float(np.sum(np.abs(actual)))
    wape = float(np.sum(np.abs(errors)) / wape_denominator * 100) if wape_denominator else np.nan
    if len(actual) > 1 and np.var(actual) > 0:
        ss_res = float(np.sum(errors ** 2))
        ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
        r2 = 1 - ss_res / ss_tot
    else:
        r2 = np.nan
    production_model = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1, **best_params)
    production_model.fit(X, y, verbose=False)
    return {'model': production_model, 'test_results': test_results, 'metrics': {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'WAPE': wape, 'R2': r2, 'Validation_MAE': best_val_mae}, 'best_params': best_params, 'counts': {'total': n, 'train': len(X_train), 'validation': len(X_val), 'test': len(X_test)}}

def create_feature_row(future_date: pd.Timestamp, history: pd.DataFrame, rainfall: float) -> pd.DataFrame:
    """Create one forecast row using the same feature definitions as training."""
    history = history.sort_values('Date').reset_index(drop=True)

    def value_at_lag(column: str, days: int, fallback: float) -> float:
        prior = history.loc[history['Date'].lt(future_date), column].dropna()
        if len(prior) >= days:
            return float(prior.iloc[-days])
        if not prior.empty:
            return float(prior.iloc[0])
        return float(fallback)
    gen = history['Generation_kWh'].dropna()
    rain = history['Rainfall_Input_mm'].dropna()
    gen_fallback = float(gen.tail(30).mean()) if not gen.empty else 0.0
    rain_fallback = float(rain.tail(30).median()) if not rain.empty else 0.0
    lag1 = value_at_lag('Generation_kWh', 1, gen_fallback)
    lag7 = value_at_lag('Generation_kWh', 7, gen_fallback)
    lag14 = value_at_lag('Generation_kWh', 14, gen_fallback)
    lag30 = value_at_lag('Generation_kWh', 30, gen_fallback)
    previous_gen = gen.tail(30)
    rolling7_gen = float(previous_gen.tail(7).mean()) if not previous_gen.empty else gen_fallback
    rolling30_gen = float(previous_gen.mean()) if not previous_gen.empty else gen_fallback
    lag_rain1 = value_at_lag('Rainfall_Input_mm', 1, rainfall if np.isfinite(rainfall) else rain_fallback)
    lag_rain7 = value_at_lag('Rainfall_Input_mm', 7, rainfall if np.isfinite(rainfall) else rain_fallback)
    lag_rain14 = value_at_lag('Rainfall_Input_mm', 14, rainfall if np.isfinite(rainfall) else rain_fallback)
    lag_rain30 = value_at_lag('Rainfall_Input_mm', 30, rainfall if np.isfinite(rainfall) else rain_fallback)
    previous_rain = rain.tail(30)
    rolling7_rain = float(previous_rain.tail(7).sum()) if not previous_rain.empty else 0.0
    rolling30_rain = float(previous_rain.sum()) if not previous_rain.empty else 0.0
    coverage = float(history['Rainfall_Input_mm'].tail(30).notna().sum() / 30)
    day_of_year = int(future_date.dayofyear)
    week_of_year = int(future_date.isocalendar().week)
    row = {'Rainfall_Input_mm': max(float(rainfall), 0.0), 'Lag1_Generation_kWh': lag1, 'Lag7_Generation_kWh': lag7, 'Lag14_Generation_kWh': lag14, 'Lag30_Generation_kWh': lag30, 'Rolling7_Generation_kWh': rolling7_gen, 'Rolling30_Generation_kWh': rolling30_gen, 'Lag1_Rainfall_mm': lag_rain1, 'Lag7_Rainfall_mm': lag_rain7, 'Lag14_Rainfall_mm': lag_rain14, 'Lag30_Rainfall_mm': lag_rain30, 'Rolling7_Rainfall_mm': rolling7_rain, 'Rolling30_Rainfall_mm': rolling30_rain, 'Month': int(future_date.month), 'DayOfYear': day_of_year, 'WeekOfYear': week_of_year, 'Sin_DayOfYear': float(np.sin(2 * np.pi * day_of_year / 365.25)), 'Cos_DayOfYear': float(np.cos(2 * np.pi * day_of_year / 365.25)), 'Rainfall_Coverage_30D': coverage}
    return pd.DataFrame([[row[c] for c in FEATURES]], columns=FEATURES)

def forecast_plant(data: pd.DataFrame, site: str, forecast_start: pd.Timestamp, horizon_days: int, model) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Bridge any historical-data gap recursively, then return requested dates."""
    site_history = data.loc[data['Site'].eq(site), ['Date', 'Generation_kWh', 'Rainfall_Input_mm']].copy().sort_values('Date').reset_index(drop=True)
    if site_history.empty:
        raise ValueError(f'No historical records found for {site}.')
    last_actual_date = site_history['Date'].max()
    forecast_end = forecast_start + pd.Timedelta(days=horizon_days - 1)
    rain_profile = site_history.assign(Month=site_history['Date'].dt.month).groupby('Month')['Rainfall_Input_mm'].median().to_dict()
    fallback_rain = float(site_history['Rainfall_Input_mm'].median())
    if not np.isfinite(fallback_rain):
        fallback_rain = 0.0
    simulation_start = last_actual_date + pd.Timedelta(days=1)
    simulation_end = forecast_end
    all_dates = pd.date_range(simulation_start, simulation_end, freq='D')
    history = site_history.copy()
    results = []
    for forecast_date in all_dates:
        rainfall = float(rain_profile.get(int(forecast_date.month), fallback_rain))
        if not np.isfinite(rainfall):
            rainfall = fallback_rain
        x_future = create_feature_row(forecast_date, history, rainfall)
        prediction = float(model.predict(x_future)[0])
        prediction = max(prediction, 0.0)
        history = pd.concat([history, pd.DataFrame([{'Date': forecast_date, 'Generation_kWh': prediction, 'Rainfall_Input_mm': rainfall}])], ignore_index=True)
        if forecast_date >= forecast_start:
            results.append({'Site': site, 'Date': forecast_date, 'Forecast_Generation_kWh': prediction, 'Forecast_Rainfall_mm': rainfall, 'Forecast_Type': 'Historical monthly rainfall baseline'})
    return (pd.DataFrame(results), last_actual_date)

def _records(frame):
    if frame is None or frame.empty:
        return []
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    out = out.replace([np.inf, -np.inf], np.nan)
    return json.loads(out.to_json(orient="records", date_format="iso"))

@lru_cache(maxsize=16)
def _cached_training(site: str, data_signature: str):
    return train_site_model(site, data_signature)

def build_dashboard(site: str, forecast_days: int = 30, validation_days: int = 30):
    import json
    site = site.strip().upper()
    if forecast_days not in [7, 14, 30, 90, 180, 365]:
        raise ValueError("Unsupported forecast horizon.")
    if validation_days not in [7, 14, 30, 60, 90]:
        raise ValueError("Unsupported validation window.")
    historical = load_data()
    sites = sorted(historical["Site"].dropna().unique().tolist())
    if site not in sites:
        raise ValueError(f"Unknown plant {site}. Available plants: {', '.join(sites)}")
    site_history = historical.loc[historical["Site"].eq(site)].copy().sort_values("Date")
    trained = _cached_training(site, f"{DATA_PATH.stat().st_mtime_ns}-{DATA_PATH.stat().st_size}")
    tomorrow = pd.Timestamp(date.today()) + pd.Timedelta(days=1)
    forecast_df, last_actual_date = forecast_plant(
        historical, site, tomorrow, forecast_days, trained["model"]
    )
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])
    latest_actual = site_history.iloc[-1]
    test_results = trained["test_results"].copy()
    test_results["Date"] = pd.to_datetime(test_results["Date"])
    display_test = test_results.tail(validation_days).copy()
    display_test["Generation_Difference_kWh"] = (
        display_test["Forecast_Generation_kWh"] - display_test["Actual_Generation_kWh"]
    )
    display_test["Absolute_Error_kWh"] = display_test["Generation_Difference_kWh"].abs()
    denom = display_test["Actual_Generation_kWh"].abs().replace(0, np.nan)
    display_test["Error_Percentage"] = 100 * display_test["Absolute_Error_kWh"] / denom
    display_test["Actual_Rainfall_mm"] = display_test["Rainfall_Input_mm"]

    actual = site_history[["Date", "Generation_kWh", "Rainfall_Input_mm"]].copy()
    actual["Date"] = pd.to_datetime(actual["Date"]).dt.normalize()
    future = forecast_df.copy()
    future["Date"] = pd.to_datetime(future["Date"]).dt.normalize()
    comparison = future.merge(actual, on="Date", how="left", validate="one_to_one")
    comparison = comparison.rename(columns={
        "Generation_kWh": "Actual_Generation_kWh",
        "Rainfall_Input_mm": "Actual_Rainfall_mm",
    })
    comparison["Generation_Difference_kWh"] = (
        comparison["Forecast_Generation_kWh"] - comparison["Actual_Generation_kWh"]
    )
    comparison["Error_Percentage"] = (
        100 * comparison["Generation_Difference_kWh"].abs()
        / comparison["Actual_Generation_kWh"].abs().replace(0, np.nan)
    )
    future_monthly = forecast_df.assign(
        Month=forecast_df["Date"].dt.to_period("M").astype(str)
    ).groupby("Month", as_index=False).agg(
        Forecast_Generation_kWh=("Forecast_Generation_kWh", "sum"),
        Forecast_Rainfall_mm=("Forecast_Rainfall_mm", "sum"),
        Days=("Date", "count"),
    )
    monthly_test = display_test.assign(
        Month=display_test["Date"].dt.to_period("M").astype(str)
    ).groupby("Month", as_index=False).agg(
        Days_Compared=("Date", "count"),
        Actual_Generation_kWh=("Actual_Generation_kWh", "sum"),
        Forecast_Generation_kWh=("Forecast_Generation_kWh", "sum"),
        Measured_Rainfall_mm=("Actual_Rainfall_mm", lambda x: x.sum(min_count=1)),
    )
    monthly_test["Generation_Difference_kWh"] = (
        monthly_test["Forecast_Generation_kWh"] - monthly_test["Actual_Generation_kWh"]
    )
    monthly_test["Error_Percentage"] = (
        100 * monthly_test["Generation_Difference_kWh"].abs()
        / monthly_test["Actual_Generation_kWh"].abs().replace(0, np.nan)
    )
    metrics = {}
    for key, value in trained["metrics"].items():
        metrics[key] = float(value) if np.isfinite(value) else None
    return {
        "site": site,
        "sites": sites,
        "forecast_days": forecast_days,
        "validation_days": validation_days,
        "last_actual_date": pd.Timestamp(last_actual_date).strftime("%Y-%m-%d"),
        "latest_actual_generation": float(latest_actual["Generation_kWh"]),
        "next_forecast": _records(forecast_df.head(1))[0],
        "forecast": _records(forecast_df),
        "future_comparison": _records(comparison),
        "future_monthly": _records(future_monthly),
        "test_results": _records(display_test),
        "monthly_test": _records(monthly_test),
        "metrics": metrics,
        "counts": trained["counts"],
        "best_params": trained["best_params"],
        "rainfall_note": "Future rainfall uses historical same-month median assumptions, not a live weather forecast.",
    }
