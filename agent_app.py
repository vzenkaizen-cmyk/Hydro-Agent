from __future__ import annotations

from pathlib import Path
from datetime import date
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from xgboost import XGBRegressor


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Hydro Intelligence Agent",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Respect the active Streamlit theme in Plotly charts.
IS_DARK_THEME = st.get_option("theme.base") == "dark"
PLOTLY_TEMPLATE = "plotly_dark" if IS_DARK_THEME else "plotly"

# Respect the active Streamlit theme in Plotly charts.
IS_DARK_THEME = st.get_option("theme.base") == "dark"
PLOTLY_TEMPLATE = "plotly_dark" if IS_DARK_THEME else "plotly"

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "HydroGen_Model_Ready_Dataset.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "Rainfall_Input_mm",
    "Lag1_Generation_kWh", "Lag7_Generation_kWh",
    "Lag14_Generation_kWh", "Lag30_Generation_kWh",
    "Rolling7_Generation_kWh", "Rolling30_Generation_kWh",
    "Lag1_Rainfall_mm", "Lag7_Rainfall_mm",
    "Lag14_Rainfall_mm", "Lag30_Rainfall_mm",
    "Rolling7_Rainfall_mm", "Rolling30_Rainfall_mm",
    "Month", "DayOfYear", "WeekOfYear",
    "Sin_DayOfYear", "Cos_DayOfYear",
    "Rainfall_Coverage_30D",
]


# ============================================================
# CLEAN, CONSISTENT PAGE STYLING
# ============================================================
st.markdown(
    """
    <style>
    /* Prevent the title and first line from being hidden under Streamlit chrome. */
    .block-container {
        padding-top: 2.25rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1500px;
    }
    [data-testid="stHeader"] {
        height: 2.8rem;
        background: var(--background-color, #ffffff);
    }
    [data-testid="stToolbar"] { right: 1rem; }
    h1, h2, h3, h4, p, label, [data-testid="stMetricLabel"] {
        overflow: visible !important;
        line-height: 1.35 !important;
    }
    h1 {
        padding-top: .15rem !important;
        margin-top: .15rem !important;
        margin-bottom: .35rem !important;
        font-weight: 750 !important;
        letter-spacing: -.025em;
    }
    h2 { margin-top: 1.3rem !important; }
    h3 { margin-top: .7rem !important; }
    [data-testid="stMetric"] {
        background: var(--secondary-background-color, #ffffff);
        color: var(--text-color, #262730);
        border: 1px solid var(--border-color, #e5e7eb);
        border-radius: 12px;
        padding: 15px 17px;
        min-height: 112px;
    }
    [data-testid="stMetric"] [data-testid="stMetricLabel"],
    [data-testid="stMetric"] [data-testid="stMetricValue"],
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: var(--text-color, #262730) !important;
    }
    [data-theme="dark"] [data-testid="stMetric"],
    body[data-theme="dark"] [data-testid="stMetric"] {
        background: #262b36;
        border-color: #454b58;
        color: #f1f5f9;
    }
    [data-theme="dark"] [data-testid="stMetric"] [data-testid="stMetricLabel"],
    [data-theme="dark"] [data-testid="stMetric"] [data-testid="stMetricValue"],
    [data-theme="dark"] [data-testid="stMetric"] [data-testid="stMetricDelta"],
    body[data-theme="dark"] [data-testid="stMetric"] [data-testid="stMetricLabel"],
    body[data-theme="dark"] [data-testid="stMetric"] [data-testid="stMetricValue"],
    body[data-theme="dark"] [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #f1f5f9 !important;
    }
    [data-testid="stMetricLabel"] p {
        white-space: normal !important;
        font-size: .9rem !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.65rem !important;
        line-height: 1.3 !important;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
    }
    .hero {
        background: linear-gradient(110deg, #092b4c 0%, #124c78 100%);
        color: white;
        border-radius: 14px;
        padding: 22px 26px;
        margin: 1rem 0 1.5rem 0;
        box-shadow: 0 5px 18px rgba(12, 42, 70, .12);
    }
    .hero-kicker {
        font-size: .72rem;
        letter-spacing: .12em;
        text-transform: uppercase;
        font-weight: 700;
        color: #9bd8ff;
        margin-bottom: 7px;
    }
    .hero-title {
        font-size: 1.45rem;
        line-height: 1.35;
        font-weight: 750;
        margin: 0;
    }
    .subtle { color: var(--text-color, #64748b); opacity: .72; font-size: .9rem; }
    .section-note { color: var(--text-color, #64748b); opacity: .72; font-size: .88rem; margin-top: -.35rem; }
    [data-theme="dark"] .subtle,
    [data-theme="dark"] .section-note,
    body[data-theme="dark"] .subtle,
    body[data-theme="dark"] .section-note {
        color: #cbd5e1;
        opacity: .9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA AND MODEL LOADING
# ============================================================
# ============================================================
# DATA AND MODEL LOADING
# ============================================================
OCTOBER_UPDATE_PATH = BASE_DIR / "data" / "hydro_october_2026_updates.csv"


def _rebuild_model_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuild the model features from the merged daily actual records.
    This ensures the newly imported October rows have the same lag/rolling
    feature structure as the existing model dataset.
    """
    data = data.sort_values(["Site", "Date"]).reset_index(drop=True).copy()

    for col in FEATURES:
        if col not in data.columns:
            data[col] = np.nan

    rebuilt_groups = []
    for _, group in data.groupby("Site", sort=False):
        group = group.sort_values("Date").copy()
        gen = pd.to_numeric(group["Generation_kWh"], errors="coerce")
        rain = pd.to_numeric(group["Rainfall_Input_mm"], errors="coerce")

        # Generation lags and rolling means use previous observations only.
        for lag in (1, 7, 14, 30):
            shifted = gen.shift(lag)
            if len(gen):
                first_value = gen.iloc[0]
                shifted = shifted.fillna(first_value)
            group[f"Lag{lag}_Generation_kWh"] = shifted

        previous_gen = gen.shift(1)
        group["Rolling7_Generation_kWh"] = previous_gen.rolling(
            7, min_periods=1
        ).mean()
        group["Rolling30_Generation_kWh"] = previous_gen.rolling(
            30, min_periods=1
        ).mean()

        # Rainfall lags follow the agent's observed-rainfall history logic:
        # use the last available rainfall observations, not zero-filled gaps.
        rain_values = rain.tolist()
        lag_values = {1: [], 7: [], 14: [], 30: []}
        rolling7_values, rolling30_values, coverage_values = [], [], []

        for i in range(len(group)):
            prior_rain = [v for v in rain_values[:i] if pd.notna(v)]
            for lag in (1, 7, 14, 30):
                if len(prior_rain) >= lag:
                    lag_value = prior_rain[-lag]
                elif prior_rain:
                    lag_value = prior_rain[0]
                else:
                    lag_value = np.nan
                lag_values[lag].append(lag_value)

            rolling7_values.append(float(sum(prior_rain[-7:])) if prior_rain else 0.0)
            rolling30_values.append(float(sum(prior_rain[-30:])) if prior_rain else 0.0)
            last_30_rows = rain_values[max(0, i - 30):i]
            coverage_values.append(
                sum(pd.notna(v) for v in last_30_rows) / 30.0
            )

        for lag in (1, 7, 14, 30):
            group[f"Lag{lag}_Rainfall_mm"] = lag_values[lag]
        group["Rolling7_Rainfall_mm"] = rolling7_values
        group["Rolling30_Rainfall_mm"] = rolling30_values
        group["Rainfall_Coverage_30D"] = coverage_values

        group["Month"] = group["Date"].dt.month.astype(float)
        group["DayOfYear"] = group["Date"].dt.dayofyear.astype(float)
        group["WeekOfYear"] = group["Date"].dt.isocalendar().week.astype(float)
        group["Sin_DayOfYear"] = np.sin(
            2 * np.pi * group["Date"].dt.dayofyear / 365.25
        )
        group["Cos_DayOfYear"] = np.cos(
            2 * np.pi * group["Date"].dt.dayofyear / 365.25
        )

        rebuilt_groups.append(group)

    if rebuilt_groups:
        data = pd.concat(rebuilt_groups, ignore_index=True)

    feature_values = data[FEATURES].replace([np.inf, -np.inf], np.nan)
    data["Model_Ready"] = feature_values.notna().all(axis=1)
    return data.sort_values(["Site", "Date"]).reset_index(drop=True)


@st.cache_data(show_spinner="Loading historical generation and rainfall data...")
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    required = {"Site", "Date", "Generation_kWh", "Rainfall_Input_mm"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing columns: {', '.join(sorted(missing))}"
        )

    data["Site"] = data["Site"].astype(str).str.strip().str.upper()
    # The October report labels this site ORK; historical data uses ORIC.
    data["Site"] = data["Site"].replace({"ORIC": "ORK", "ORIK": "ORK"})
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce").dt.normalize()
    data["Generation_kWh"] = pd.to_numeric(
        data["Generation_kWh"], errors="coerce"
    )
    data["Rainfall_Input_mm"] = pd.to_numeric(
        data["Rainfall_Input_mm"], errors="coerce"
    )
    data = data.dropna(subset=["Site", "Date", "Generation_kWh"]).copy()

    # Apply the verified October 2026 daily readings on top of the existing
    # historical dataset. The original dataset remains the historical base.
    if OCTOBER_UPDATE_PATH.exists():
        updates = pd.read_csv(OCTOBER_UPDATE_PATH)
        updates["Site"] = updates["Site"].astype(str).str.strip().str.upper()
        updates["Site"] = updates["Site"].replace({"ORIC": "ORK", "ORIK": "ORK"})
        updates["Date"] = pd.to_datetime(
            updates["Date"], errors="coerce"
        ).dt.normalize()
        updates["Generation_kWh"] = pd.to_numeric(
            updates["Generation_kWh"], errors="coerce"
        )
        updates["Rainfall_Input_mm"] = pd.to_numeric(
            updates["Rainfall_Input_mm"], errors="coerce"
        )
        updates["Plant_Factor"] = pd.to_numeric(
            updates.get("Plant_Factor"), errors="coerce"
        )
        updates["Rainfall_Report_Available"] = (
            updates["Rainfall_Report_Available"]
            .astype(str).str.strip().str.lower()
            .isin(["true", "1", "yes", "y"])
        )

        updates = updates.dropna(
            subset=["Site", "Date", "Generation_kWh"]
        ).drop_duplicates(["Site", "Date"], keep="last")

        # Ensure optional columns exist before merging.
        if "Plant_Factor" not in data.columns:
            data["Plant_Factor"] = np.nan
        if "Rainfall_Stations_Reported" not in data.columns:
            data["Rainfall_Stations_Reported"] = np.nan

        base = data.set_index(["Site", "Date"]).sort_index()
        upd = updates.set_index(["Site", "Date"]).sort_index()
        all_index = base.index.union(upd.index)
        base = base.reindex(all_index)

        # Generation and plant factor are replaced by the report values.
        base.loc[upd.index, "Generation_kWh"] = upd["Generation_kWh"]
        base.loc[upd.index, "Plant_Factor"] = upd["Plant_Factor"]

        # For the seven plants with supplied rainfall reports, apply the report
        # exactly, including blank/missing gauge dates. For plants without a
        # rainfall PDF, retain any rainfall already present in the base data.
        report_sites = set(
            updates.loc[
                updates["Rainfall_Report_Available"], "Site"
            ].unique()
        )
        report_index = upd.index[
            upd.index.get_level_values("Site").isin(report_sites)
        ]
        base.loc[report_index, "Rainfall_Input_mm"] = upd.loc[
            report_index, "Rainfall_Input_mm"
        ]
        base.loc[upd.index, "Rainfall_Stations_Reported"] = upd[
            "Rainfall_Stations_Reported"
        ]

        data = base.reset_index()

    data["Rainfall_Input_mm"] = pd.to_numeric(
        data["Rainfall_Input_mm"], errors="coerce"
    ).clip(lower=0)
    data = (
        data.sort_values(["Site", "Date"])
        .drop_duplicates(["Site", "Date"], keep="last")
        .reset_index(drop=True)
    )

    # Recalculate lag and rolling features so October data is model-compatible.
    data = _rebuild_model_features(data)
    return data


@st.cache_resource(show_spinner=False)
def train_site_model(site: str, data_signature: str) -> dict:
    """
    Chronological model selection:
      70% train, 15% validation, 15% final test.
    The final test is not used for hyperparameter selection.
    A separate production model is then fitted on all eligible historical rows.
    """
    data = load_data()
    site_data = data.loc[data["Site"].eq(site)].sort_values("Date").copy()

    missing_features = [c for c in FEATURES if c not in site_data.columns]
    if missing_features:
        raise ValueError(
            "Dataset is missing model features: " + ", ".join(missing_features)
        )

    if "Model_Ready" in site_data.columns:
        ready = site_data["Model_Ready"].astype(str).str.strip().str.lower()
        site_data = site_data.loc[ready.isin(["true", "1", "yes", "y"])].copy()

    numeric_cols = FEATURES + ["Generation_kWh"]
    for col in numeric_cols:
        site_data[col] = pd.to_numeric(site_data[col], errors="coerce")

    site_data = site_data.replace([np.inf, -np.inf], np.nan)
    site_data = site_data.dropna(subset=numeric_cols).sort_values("Date").reset_index(drop=True)

    if len(site_data) < 100:
        raise ValueError(
            f"{site} has only {len(site_data)} model-ready rows. "
            "At least 100 rows are recommended for chronological train/validation/test evaluation."
        )

    X = site_data[FEATURES].astype(float)
    y = site_data["Generation_kWh"].astype(float)

    n = len(site_data)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    if train_end < 30 or validation_end <= train_end or validation_end >= n:
        raise ValueError("Not enough rows to create chronological train/validation/test sets.")

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:validation_end], y.iloc[train_end:validation_end]
    X_test, y_test = X.iloc[validation_end:], y.iloc[validation_end:]

    candidates = [
        {"n_estimators": 350, "max_depth": 3, "learning_rate": 0.03,
         "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 5.0},
        {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.025,
         "subsample": 0.90, "colsample_bytree": 0.90, "reg_lambda": 8.0},
        {"n_estimators": 350, "max_depth": 4, "learning_rate": 0.03,
         "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 8.0},
    ]

    best_params = None
    best_val_mae = float("inf")
    for params in candidates:
        candidate = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            **params,
        )
        candidate.fit(X_train, y_train, verbose=False)
        val_pred = np.maximum(candidate.predict(X_val), 0)
        val_mae = float(np.mean(np.abs(y_val.to_numpy() - val_pred)))
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_params = params

    # Final unbiased holdout model: train + validation only.
    test_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        **best_params,
    )
    test_model.fit(
        pd.concat([X_train, X_val]),
        pd.concat([y_train, y_val]),
        verbose=False,
    )
    test_prediction = np.maximum(test_model.predict(X_test), 0)

    test_results = site_data.iloc[validation_end:].copy().reset_index(drop=True)
    test_results["Actual_Generation_kWh"] = y_test.to_numpy()
    test_results["Forecast_Generation_kWh"] = test_prediction
    test_results["Generation_Difference_kWh"] = (
        test_results["Forecast_Generation_kWh"] - test_results["Actual_Generation_kWh"]
    )
    test_results["Absolute_Error_kWh"] = test_results["Generation_Difference_kWh"].abs()
    actual_denominator = test_results["Actual_Generation_kWh"].abs().replace(0, np.nan)
    test_results["Error_Percentage"] = (
        100 * test_results["Absolute_Error_kWh"] / actual_denominator
    )
    # In this one-day-ahead holdout, measured rainfall is an input feature,
    # not an independently predicted rainfall forecast.
    test_results["Actual_Rainfall_mm"] = test_results["Rainfall_Input_mm"]

    actual = test_results["Actual_Generation_kWh"].to_numpy()
    predicted = test_results["Forecast_Generation_kWh"].to_numpy()
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

    # Production model uses all available model-ready records only after evaluation.
    production_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        **best_params,
    )
    production_model.fit(X, y, verbose=False)

    return {
        "model": production_model,
        "test_results": test_results,
        "metrics": {
            "MAE": mae, "RMSE": rmse, "MAPE": mape, "WAPE": wape, "R2": r2,
            "Validation_MAE": best_val_mae,
        },
        "best_params": best_params,
        "counts": {
            "total": n, "train": len(X_train), "validation": len(X_val), "test": len(X_test)
        },
    }


def create_feature_row(
    future_date: pd.Timestamp,
    history: pd.DataFrame,
    rainfall: float,
) -> pd.DataFrame:
    """Create one forecast row using the same feature definitions as training."""
    history = history.sort_values("Date").reset_index(drop=True)

    def value_at_lag(column: str, days: int, fallback: float) -> float:
        # Lag features are row shifts in the daily training series. Use the
        # corresponding previous observation; never replace a lag with an
        # average of unrelated recent values.
        prior = history.loc[history["Date"].lt(future_date), column].dropna()
        if len(prior) >= days:
            return float(prior.iloc[-days])
        if not prior.empty:
            return float(prior.iloc[0])
        return float(fallback)

    gen = history["Generation_kWh"].dropna()
    rain = history["Rainfall_Input_mm"].dropna()
    gen_fallback = float(gen.tail(30).mean()) if not gen.empty else 0.0
    rain_fallback = float(rain.tail(30).median()) if not rain.empty else 0.0

    lag1 = value_at_lag("Generation_kWh", 1, gen_fallback)
    lag7 = value_at_lag("Generation_kWh", 7, gen_fallback)
    lag14 = value_at_lag("Generation_kWh", 14, gen_fallback)
    lag30 = value_at_lag("Generation_kWh", 30, gen_fallback)

    previous_gen = gen.tail(30)
    rolling7_gen = float(previous_gen.tail(7).mean()) if not previous_gen.empty else gen_fallback
    rolling30_gen = float(previous_gen.mean()) if not previous_gen.empty else gen_fallback

    lag_rain1 = value_at_lag("Rainfall_Input_mm", 1, rainfall if np.isfinite(rainfall) else rain_fallback)
    lag_rain7 = value_at_lag("Rainfall_Input_mm", 7, rainfall if np.isfinite(rainfall) else rain_fallback)
    lag_rain14 = value_at_lag("Rainfall_Input_mm", 14, rainfall if np.isfinite(rainfall) else rain_fallback)
    lag_rain30 = value_at_lag("Rainfall_Input_mm", 30, rainfall if np.isfinite(rainfall) else rain_fallback)

    previous_rain = rain.tail(30)
    rolling7_rain = float(previous_rain.tail(7).sum()) if not previous_rain.empty else 0.0
    rolling30_rain = float(previous_rain.sum()) if not previous_rain.empty else 0.0

    coverage = float(history["Rainfall_Input_mm"].tail(30).notna().sum() / 30)
    day_of_year = int(future_date.dayofyear)
    week_of_year = int(future_date.isocalendar().week)

    row = {
        "Rainfall_Input_mm": max(float(rainfall), 0.0),
        "Lag1_Generation_kWh": lag1,
        "Lag7_Generation_kWh": lag7,
        "Lag14_Generation_kWh": lag14,
        "Lag30_Generation_kWh": lag30,
        "Rolling7_Generation_kWh": rolling7_gen,
        "Rolling30_Generation_kWh": rolling30_gen,
        "Lag1_Rainfall_mm": lag_rain1,
        "Lag7_Rainfall_mm": lag_rain7,
        "Lag14_Rainfall_mm": lag_rain14,
        "Lag30_Rainfall_mm": lag_rain30,
        "Rolling7_Rainfall_mm": rolling7_rain,
        "Rolling30_Rainfall_mm": rolling30_rain,
        "Month": int(future_date.month),
        "DayOfYear": day_of_year,
        "WeekOfYear": week_of_year,
        "Sin_DayOfYear": float(np.sin(2 * np.pi * day_of_year / 365.25)),
        "Cos_DayOfYear": float(np.cos(2 * np.pi * day_of_year / 365.25)),
        "Rainfall_Coverage_30D": coverage,
    }
    return pd.DataFrame([[row[c] for c in FEATURES]], columns=FEATURES)


def forecast_plant(
    data: pd.DataFrame,
    site: str,
    forecast_start: pd.Timestamp,
    horizon_days: int,
    model,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Bridge any historical-data gap recursively, then return requested dates."""
    site_history = data.loc[data["Site"].eq(site), [
        "Date", "Generation_kWh", "Rainfall_Input_mm"
    ]].copy().sort_values("Date").reset_index(drop=True)

    if site_history.empty:
        raise ValueError(f"No historical records found for {site}.")

    last_actual_date = site_history["Date"].max()
    forecast_end = forecast_start + pd.Timedelta(days=horizon_days - 1)

    # Use the plant's historical median rainfall for the same calendar month.
    # This is a baseline scenario, not a live weather forecast.
    rain_profile = (
        site_history.assign(Month=site_history["Date"].dt.month)
        .groupby("Month")["Rainfall_Input_mm"]
        .median()
        .to_dict()
    )
    fallback_rain = float(site_history["Rainfall_Input_mm"].median())
    if not np.isfinite(fallback_rain):
        fallback_rain = 0.0

    # Fill the model's recursive history up to the requested forecast start.
    # This is important when the latest uploaded actual data is older than today.
    simulation_start = last_actual_date + pd.Timedelta(days=1)
    simulation_end = forecast_end
    all_dates = pd.date_range(simulation_start, simulation_end, freq="D")
    history = site_history.copy()
    results = []

    for forecast_date in all_dates:
        rainfall = float(rain_profile.get(int(forecast_date.month), fallback_rain))
        if not np.isfinite(rainfall):
            rainfall = fallback_rain

        x_future = create_feature_row(forecast_date, history, rainfall)
        prediction = float(model.predict(x_future)[0])
        prediction = max(prediction, 0.0)

        history = pd.concat(
            [
                history,
                pd.DataFrame([{
                    "Date": forecast_date,
                    "Generation_kWh": prediction,
                    "Rainfall_Input_mm": rainfall,
                }]),
            ],
            ignore_index=True,
        )

        if forecast_date >= forecast_start:
            results.append({
                "Site": site,
                "Date": forecast_date,
                "Forecast_Generation_kWh": prediction,
                "Forecast_Rainfall_mm": rainfall,
                "Forecast_Type": "Historical monthly rainfall baseline",
            })

    return pd.DataFrame(results), last_actual_date


# ============================================================
# HEADER
# ============================================================
st.title("Hydro Intelligence Agent")
st.caption(
    "Hydropower generation forecasting using historical plant performance, "
    "rainfall patterns and plant-specific XGBoost models"
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Hydro generation planning</div>
        <p class="hero-title">Forecast with confidence. Plan with intelligence.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================
try:
    historical = load_data()
except Exception as exc:
    st.error(f"Unable to load the forecasting dataset: {exc}")
    st.stop()

sites = sorted(historical["Site"].dropna().unique().tolist())
if not sites:
    st.error("No hydro plant records were found in the dataset.")
    st.stop()


# ============================================================
# FORECAST COMPARISON AND BACKTEST HELPERS
# ============================================================
def build_comparison(actual: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    """Compare only matched dates; missing actuals remain missing, not zero."""
    actual = actual[["Date", "Generation_kWh", "Rainfall_Input_mm"]].copy()
    actual["Date"] = pd.to_datetime(actual["Date"]).dt.normalize()
    predicted = predicted.copy()
    predicted["Date"] = pd.to_datetime(predicted["Date"]).dt.normalize()
    joined = predicted.merge(actual, on="Date", how="left", validate="one_to_one")
    joined = joined.rename(columns={
        "Generation_kWh": "Actual_Generation_kWh",
        "Rainfall_Input_mm": "Actual_Rainfall_mm",
    })
    joined["Generation_Difference_kWh"] = (
        joined["Forecast_Generation_kWh"] - joined["Actual_Generation_kWh"]
    )
    joined["Absolute_Error_kWh"] = joined["Generation_Difference_kWh"].abs()
    denominator = joined["Actual_Generation_kWh"].where(joined["Actual_Generation_kWh"].ne(0))
    joined["Error_Percentage"] = 100 * joined["Absolute_Error_kWh"] / denominator
    joined["Rainfall_Difference_mm"] = joined["Forecast_Rainfall_mm"] - joined["Actual_Rainfall_mm"]
    return joined


def comparison_chart(df: pd.DataFrame, actual_col: str, forecast_col: str,
                     title: str, yaxis: str, bars: bool = False):
    fig = go.Figure()
    x = df["Date"] if "Date" in df.columns else df["Month"]
    for col, label, color in [(actual_col, "Actual", "#16866c"),
                              (forecast_col, "Forecast", "#1677b8")]:
        if bars:
            fig.add_trace(go.Bar(x=x, y=df[col], name=label, marker_color=color))
        else:
            fig.add_trace(go.Scatter(x=x, y=df[col], name=label, mode="lines+markers",
                                     line=dict(color=color, width=2), marker=dict(size=4)))
    fig.update_layout(title=title, height=350, margin=dict(l=10, r=10, t=55, b=10),
                      template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
                      barmode="group", legend=dict(orientation="h", y=1.1),
                      yaxis_title=yaxis, xaxis_title="", yaxis=dict(rangemode="tozero"))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# SIDEBAR CONTROLS
# ============================================================
with st.sidebar:
    st.subheader("Forecast Controls")
    selected_site = st.selectbox("Select Hydro Plant", sites, index=sites.index("BBO") if "BBO" in sites else 0)
    forecast_days = st.select_slider("Forecast horizon", options=[7, 14, 30, 90, 180, 365],
                                     value=30, format_func=lambda x: f"{x} days")
    validation_days = st.select_slider("Historical validation window", options=[7, 14, 30, 60, 90],
                                       value=30, format_func=lambda x: f"{x} days")
    st.caption("Controls how many recent chronological test days are shown in the comparison.")
    st.divider()
    st.caption("Forecast inputs: plant-specific XGBoost model, historical generation and monthly rainfall baseline")

site_history = historical.loc[historical["Site"].eq(selected_site)].copy().sort_values("Date")
today = pd.Timestamp(date.today())
tomorrow = today + pd.Timedelta(days=1)

# Train/evaluate chronologically, then use the production model for future forecasts.
try:
    stat = DATA_PATH.stat()
    update_stat = OCTOBER_UPDATE_PATH.stat() if OCTOBER_UPDATE_PATH.exists() else None
    update_signature = (
        f"{update_stat.st_mtime_ns}-{update_stat.st_size}"
        if update_stat else "no-october-update"
    )
    data_signature = f"{stat.st_mtime_ns}-{stat.st_size}-{update_signature}"
    with st.spinner("Training and validating the plant-specific XGBoost model..."):
        trained = train_site_model(selected_site, data_signature)
    with st.spinner("Calculating upcoming generation forecasts..."):
        forecast_df, last_actual_date = forecast_plant(
            historical, selected_site, tomorrow, forecast_days, trained["model"]
        )
except Exception as exc:
    st.error(f"Unable to prepare the {selected_site} forecasting model: {exc}")
    st.stop()

if forecast_df.empty:
    st.warning("No future forecast records were generated.")
    st.stop()

forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])
future_comparison = build_comparison(site_history, forecast_df)
last_actual_date = pd.Timestamp(last_actual_date)
latest_actual = site_history.iloc[-1]
next_row = forecast_df.iloc[0]
data_gap_days = max((today - last_actual_date.normalize()).days, 0)

st.subheader(f"{selected_site} Generation Forecast")
st.markdown(f"<p class='section-note'>Latest recorded actual: <b>{last_actual_date:%d %B %Y}</b> · "
            f"Next forecast: <b>{tomorrow:%d %B %Y}</b></p>", unsafe_allow_html=True)
if data_gap_days > 3:
    st.warning(f"Actual records are {data_gap_days} days old. Forecasts bridge this gap using recursive "
               "predictions and historical rainfall assumptions; update measured data for better reliability.")

st.subheader("Tomorrow's Generation Forecast")
a, b, c, d = st.columns(4)
a.metric("Forecast Generation", f"{next_row['Forecast_Generation_kWh']:,.0f} kWh")
b.metric("Assumed Rainfall", f"{next_row['Forecast_Rainfall_mm']:,.1f} mm")
c.metric("Latest Actual Generation", f"{latest_actual['Generation_kWh']:,.0f} kWh")
d.metric("Latest Actual Date", last_actual_date.strftime("%d %b %Y"))
st.caption("The latest actual and tomorrow's forecast refer to different dates; their difference is not a forecast error.")
st.info("Rainfall forecasts shown here are historical same-month median assumptions, not live weather forecasts. "
        "Future actuals and errors are unavailable until actual readings are recorded.")

st.subheader(f"Upcoming {forecast_days}-Day Forecast")
future_chart = go.Figure()
future_chart.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecast_Generation_kWh"],
                                  name="Forecast generation", mode="lines+markers",
                                  line=dict(color="#1677b8", width=2)))
future_chart.update_layout(height=350, margin=dict(l=10,r=10,t=25,b=10),
                           template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)",
                           yaxis_title="Generation (kWh)", xaxis_title="", hovermode="x unified")
st.plotly_chart(future_chart, use_container_width=True)

st.subheader("Upcoming Daily Generation and Rainfall")
future_display = future_comparison[["Date", "Forecast_Generation_kWh", "Forecast_Rainfall_mm",
                                    "Actual_Generation_kWh", "Actual_Rainfall_mm", "Error_Percentage"]].copy()
future_display["Date"] = future_display["Date"].dt.strftime("%d %b %Y")
st.dataframe(future_display.rename(columns={
    "Forecast_Generation_kWh":"Forecast Generation (kWh)", "Forecast_Rainfall_mm":"Assumed Rainfall (mm)",
    "Actual_Generation_kWh":"Actual Generation (kWh)", "Actual_Rainfall_mm":"Actual Rainfall (mm)",
    "Error_Percentage":"Error (%)"}), use_container_width=True, hide_index=True)

future_monthly = forecast_df.assign(Month=forecast_df["Date"].dt.to_period("M").astype(str)).groupby(
    "Month", as_index=False).agg(Forecast_Generation_kWh=("Forecast_Generation_kWh","sum"),
                                Forecast_Rainfall_mm=("Forecast_Rainfall_mm","sum"),
                                Days=("Date","count"))
st.subheader("Upcoming Monthly Generation and Rainfall")
st.dataframe(future_monthly, use_container_width=True, hide_index=True)
st.caption("Partial months include only the forecast dates within the selected horizon.")

# ============================================================
# CHRONOLOGICAL HOLDOUT VALIDATION
# ============================================================
st.subheader("Historical Actual vs Forecast Comparison")

test_results = trained["test_results"].copy()
test_results["Date"] = pd.to_datetime(test_results["Date"])
metrics = trained["metrics"]
counts = trained["counts"]

st.caption(
    f"Chronological split: {counts['train']:,} training rows (70%), "
    f"{counts['validation']:,} validation rows (15%), and "
    f"{counts['test']:,} untouched test rows (15%). "
    "The final test model is fitted on training + validation data only."
)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Test MAE", f"{metrics['MAE']:,.0f} kWh")
m2.metric("Test RMSE", f"{metrics['RMSE']:,.0f} kWh")
m3.metric("Test MAPE", f"{metrics['MAPE']:.2f}%" if np.isfinite(metrics["MAPE"]) else "N/A")
m4.metric("Test WAPE", f"{metrics['WAPE']:.2f}%" if np.isfinite(metrics["WAPE"]) else "N/A")
m5.metric("Test R²", f"{metrics['R2']:.3f}" if np.isfinite(metrics["R2"]) else "N/A")

st.caption(
    f"Best validation MAE: {metrics['Validation_MAE']:,.0f} kWh · "
    f"Selected XGBoost parameters: {trained['best_params']}"
)
st.info(
    "This is a chronological one-day-ahead holdout evaluation. Actual historical rainfall "
    "is supplied as an input for each test day; rainfall is not being forecast in this test. "
    "Future forecasts use historical same-month median rainfall assumptions."
)

# ============================================================
# FULL HISTORICAL DAILY / MONTHLY COMPARISON
# ============================================================
# Show every month available in the source dataset. Forecast values are
# populated only for the untouched chronological test period, so earlier
# historical rows are not misleadingly presented as out-of-sample forecasts.
st.subheader("Historical Actual vs Forecast Comparison")

test_results = trained["test_results"].copy()
test_results["Date"] = pd.to_datetime(test_results["Date"]).dt.normalize()
metrics = trained["metrics"]
counts = trained["counts"]

st.caption(
    f"Chronological split: {counts['train']:,} training rows (70%), "
    f"{counts['validation']:,} validation rows (15%), and "
    f"{counts['test']:,} untouched test rows (15%). "
    "The final test model is fitted on training + validation data only."
)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Test MAE", f"{metrics['MAE']:,.0f} kWh")
m2.metric("Test RMSE", f"{metrics['RMSE']:,.0f} kWh")
m3.metric("Test MAPE", f"{metrics['MAPE']:.2f}%" if np.isfinite(metrics["MAPE"]) else "N/A")
m4.metric("Test WAPE", f"{metrics['WAPE']:.2f}%" if np.isfinite(metrics["WAPE"]) else "N/A")
m5.metric("Test R²", f"{metrics['R2']:.3f}" if np.isfinite(metrics["R2"]) else "N/A")

st.caption(
    f"Best validation MAE: {metrics['Validation_MAE']:,.0f} kWh · "
    f"Selected XGBoost parameters: {trained['best_params']}"
)
st.info(
    "Historical actual generation and measured rainfall are shown for every date "
    "available in the selected plant's source dataset. Forecast-versus-actual errors "
    "are shown only for the untouched chronological test dates. Historical rainfall "
    "is measured input data; it is not a separately predicted rainfall forecast."
)

# Start with the complete historical record, then attach only genuine test forecasts.
historical_comparison = site_history[
    ["Date", "Generation_kWh", "Rainfall_Input_mm"]
].copy()
historical_comparison["Date"] = pd.to_datetime(
    historical_comparison["Date"]
).dt.normalize()
historical_comparison = historical_comparison.rename(columns={
    "Generation_kWh": "Actual_Generation_kWh",
    "Rainfall_Input_mm": "Actual_Rainfall_mm",
})

forecast_columns = [
    "Date", "Forecast_Generation_kWh"
]
historical_comparison = historical_comparison.merge(
    test_results[forecast_columns].drop_duplicates("Date"),
    on="Date",
    how="left",
    validate="one_to_one",
)
historical_comparison["Generation_Difference_kWh"] = (
    historical_comparison["Forecast_Generation_kWh"]
    - historical_comparison["Actual_Generation_kWh"]
)
historical_comparison["Absolute_Error_kWh"] = (
    historical_comparison["Generation_Difference_kWh"].abs()
)
actual_denominator = historical_comparison["Actual_Generation_kWh"].abs().replace(0, np.nan)
historical_comparison["Error_Percentage"] = (
    100 * historical_comparison["Absolute_Error_kWh"] / actual_denominator
)
historical_comparison["Month"] = (
    historical_comparison["Date"].dt.to_period("M").astype(str)
)

if historical_comparison.empty:
    st.warning("There are no historical records available for the selected plant.")
else:
    daily_tab, monthly_tab = st.tabs(["Daily Comparison", "Monthly Comparison"])

    with daily_tab:
        month_options = sorted(historical_comparison["Month"].dropna().unique().tolist())
        # Prefer the most recent month that has genuine test forecasts.
        forecast_months = historical_comparison.loc[
            historical_comparison["Forecast_Generation_kWh"].notna(), "Month"
        ].dropna().unique().tolist()
        default_month = forecast_months[-1] if forecast_months else month_options[-1]
        default_index = month_options.index(default_month)

        selected_month = st.selectbox(
            "Select Month for Daily Comparison",
            month_options,
            index=default_index,
            key=f"{selected_site}_daily_comparison_month",
        )
        selected_month_data = historical_comparison.loc[
            historical_comparison["Month"].eq(selected_month)
        ].copy().sort_values("Date")

        forecast_days_in_month = int(
            selected_month_data["Forecast_Generation_kWh"].notna().sum()
        )
        if forecast_days_in_month == 0:
            st.warning(
                "No out-of-sample forecast predictions are available for this month. "
                "The table and rainfall chart show the actual records available in the dataset."
            )
        else:
            st.caption(
                f"{forecast_days_in_month} day(s) in this month have out-of-sample "
                "forecast predictions available for a fair actual-versus-forecast comparison."
            )

        comparison_chart(
            selected_month_data,
            "Actual_Generation_kWh",
            "Forecast_Generation_kWh",
            f"Daily Actual vs Forecast Generation - {selected_month}",
            "Generation (kWh)",
        )

        # Historical rainfall is measured rainfall, not a rainfall forecast.
        rain_fig = go.Figure()
        rain_fig.add_trace(go.Bar(
            x=selected_month_data["Date"],
            y=selected_month_data["Actual_Rainfall_mm"],
            name="Measured rainfall",
            marker_color="#16866c",
        ))
        rain_fig.update_layout(
            title=f"Daily Measured Rainfall - {selected_month}",
            height=330,
            margin=dict(l=10, r=10, t=55, b=10),
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Rainfall (mm)",
            xaxis_title="",
            hovermode="x unified",
        )
        st.plotly_chart(rain_fig, use_container_width=True)

        daily_view = selected_month_data[[
            "Date",
            "Actual_Generation_kWh",
            "Forecast_Generation_kWh",
            "Generation_Difference_kWh",
            "Absolute_Error_kWh",
            "Error_Percentage",
            "Actual_Rainfall_mm",
        ]].copy()
        daily_view["Date"] = daily_view["Date"].dt.strftime("%d %b %Y")
        daily_view = daily_view.rename(columns={
            "Actual_Generation_kWh": "Actual Generation (kWh)",
            "Forecast_Generation_kWh": "Forecast Generation (kWh)",
            "Generation_Difference_kWh": "Difference (kWh)",
            "Absolute_Error_kWh": "Absolute Error (kWh)",
            "Error_Percentage": "Error (%)",
            "Actual_Rainfall_mm": "Measured Rainfall (mm)",
        })
        st.dataframe(daily_view, use_container_width=True, hide_index=True)

        st.download_button(
            "Download Daily Comparison CSV",
            selected_month_data.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_site}_{selected_month}_daily_comparison.csv",
            mime="text/csv",
        )

        excel_buffer = BytesIO()
        # Monthly totals for the workbook: forecast totals and matched actual
        # totals are calculated over exactly the same test dates.
        matched_month = selected_month_data.dropna(
            subset=["Forecast_Generation_kWh", "Actual_Generation_kWh"]
        )
        month_summary = pd.DataFrame([{
            "Plant": selected_site,
            "Month": selected_month,
            "Historical_Actual_Days": len(selected_month_data),
            "Out_of_Sample_Forecast_Days": len(matched_month),
            "Actual_Generation_All_Available_Days_kWh":
                selected_month_data["Actual_Generation_kWh"].sum(min_count=1),
            "Actual_Generation_On_Forecast_Days_kWh":
                matched_month["Actual_Generation_kWh"].sum(min_count=1),
            "Forecast_Generation_On_Matched_Days_kWh":
                matched_month["Forecast_Generation_kWh"].sum(min_count=1),
            "Measured_Rainfall_mm":
                selected_month_data["Actual_Rainfall_mm"].sum(min_count=1),
        }])
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            selected_month_data.to_excel(
                writer, sheet_name="Daily Comparison", index=False
            )
            month_summary.to_excel(
                writer, sheet_name="Monthly Summary", index=False
            )

        st.download_button(
            "Download Selected Month Excel",
            data=excel_buffer.getvalue(),
            file_name=f"{selected_site}_{selected_month}_comparison.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with monthly_tab:
        monthly = historical_comparison.groupby("Month", as_index=False).agg(
            Historical_Actual_Days=("Date", "count"),
            Actual_Generation_All_Days_kWh=("Actual_Generation_kWh", "sum"),
            Measured_Rainfall_mm=("Actual_Rainfall_mm", lambda x: x.sum(min_count=1)),
            Forecast_Days=("Forecast_Generation_kWh", "count"),
        )

        matched_monthly = historical_comparison.dropna(
            subset=["Forecast_Generation_kWh", "Actual_Generation_kWh"]
        ).groupby("Month", as_index=False).agg(
            Actual_Generation_On_Forecast_Days_kWh=("Actual_Generation_kWh", "sum"),
            Forecast_Generation_On_Matched_Days_kWh=("Forecast_Generation_kWh", "sum"),
        )
        monthly = monthly.merge(matched_monthly, on="Month", how="left")
        monthly["Generation_Difference_kWh"] = (
            monthly["Forecast_Generation_On_Matched_Days_kWh"]
            - monthly["Actual_Generation_On_Forecast_Days_kWh"]
        )
        monthly_denominator = (
            monthly["Actual_Generation_On_Forecast_Days_kWh"].abs().replace(0, np.nan)
        )
        monthly["Error_Percentage"] = (
            100 * monthly["Generation_Difference_kWh"].abs() / monthly_denominator
        )

        monthly_chart = go.Figure()
        monthly_chart.add_trace(go.Bar(
            x=monthly["Month"],
            y=monthly["Actual_Generation_On_Forecast_Days_kWh"],
            name="Actual generation (matched test days)",
            marker_color="#16866c",
        ))
        monthly_chart.add_trace(go.Bar(
            x=monthly["Month"],
            y=monthly["Forecast_Generation_On_Matched_Days_kWh"],
            name="Forecast generation (matched test days)",
            marker_color="#1677b8",
        ))
        monthly_chart.update_layout(
            title="Monthly Actual vs Forecast Generation (matched test days)",
            height=350,
            margin=dict(l=10, r=10, t=55, b=10),
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
            barmode="group",
            legend=dict(orientation="h", y=1.1),
            yaxis_title="Generation (kWh)",
            xaxis_title="Month",
            yaxis=dict(rangemode="tozero"),
        )
        st.plotly_chart(monthly_chart, use_container_width=True)

        rain_monthly_fig = go.Figure()
        rain_monthly_fig.add_trace(go.Bar(
            x=monthly["Month"],
            y=monthly["Measured_Rainfall_mm"],
            name="Measured rainfall",
            marker_color="#16866c",
        ))
        rain_monthly_fig.update_layout(
            title="Monthly Measured Rainfall (all available historical days)",
            height=330,
            margin=dict(l=10, r=10, t=55, b=10),
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Rainfall (mm)",
            xaxis_title="Month",
            hovermode="x unified",
        )
        st.plotly_chart(rain_monthly_fig, use_container_width=True)
        st.dataframe(monthly, use_container_width=True, hide_index=True)
        st.caption(
            "Historical actual and rainfall totals include all available records. "
            "Forecast totals and forecast errors use only matched out-of-sample test days; "
            "months without test predictions therefore have blank forecast values."
        )
        st.download_button(
            "Download Monthly Comparison CSV",
            monthly.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_site}_monthly_historical_comparison.csv",
            mime="text/csv",
        )

st.download_button("Download Upcoming Forecast CSV", future_comparison.to_csv(index=False).encode(),
                   file_name=f"{selected_site}_future_forecast.csv", mime="text/csv")
st.caption("Forecast accuracy is measured from historical replay, not guaranteed for future dates. "
           "Improve accuracy by updating recent actual generation and rainfall, using verified near-term weather "
           "forecasts, and retraining/validating the plant models on newer data.")
