from __future__ import annotations

from pathlib import Path
from datetime import date

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
        background: rgba(255,255,255,0.96);
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
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 15px 17px;
        min-height: 112px;
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
    .subtle { color: #64748b; font-size: .9rem; }
    .section-note { color: #64748b; font-size: .88rem; margin-top: -.35rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA AND MODEL LOADING
# ============================================================
@st.cache_data(show_spinner="Loading historical generation and rainfall data...")
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    required = {"Site", "Date", "Generation_kWh", "Rainfall_Input_mm"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Generation_kWh"] = pd.to_numeric(data["Generation_kWh"], errors="coerce")
    data["Rainfall_Input_mm"] = pd.to_numeric(data["Rainfall_Input_mm"], errors="coerce")
    data = data.dropna(subset=["Site", "Date", "Generation_kWh"]).copy()
    data["Site"] = data["Site"].astype(str).str.strip()
    data["Rainfall_Input_mm"] = data["Rainfall_Input_mm"].clip(lower=0)
    data = (
        data.sort_values(["Site", "Date"])
        .drop_duplicates(["Site", "Date"], keep="last")
        .reset_index(drop=True)
    )
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
    x = df["Month"] if "Month" in df.columns else df["Date"]
    for col, label, color in [(actual_col, "Actual", "#16866c"),
                              (forecast_col, "Forecast", "#1677b8")]:
        if bars:
            fig.add_trace(go.Bar(x=x, y=df[col], name=label, marker_color=color))
        else:
            fig.add_trace(go.Scatter(x=x, y=df[col], name=label, mode="lines+markers",
                                     line=dict(color=color, width=2), marker=dict(size=4)))
    fig.update_layout(title=title, height=350, margin=dict(l=10, r=10, t=55, b=10),
                      paper_bgcolor="white", plot_bgcolor="white", hovermode="x unified",
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
    data_signature = f"{stat.st_mtime_ns}-{stat.st_size}"
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

# Use the sidebar setting to control how many of the most recent test days are displayed.
display_test = test_results.tail(validation_days).copy()
if display_test.empty:
    st.warning("There are no test rows available for the selected plant.")
else:
    display_test["Month"] = display_test["Date"].dt.to_period("M").astype(str)
    display_test["Generation_Difference_kWh"] = (
        display_test["Forecast_Generation_kWh"] - display_test["Actual_Generation_kWh"]
    )
    display_test["Absolute_Error_kWh"] = display_test["Generation_Difference_kWh"].abs()
    denom = display_test["Actual_Generation_kWh"].abs().replace(0, np.nan)
    display_test["Error_Percentage"] = 100 * display_test["Absolute_Error_kWh"] / denom

    daily_tab, monthly_tab = st.tabs(["Daily Comparison", "Monthly Comparison"])
    with daily_tab:
        comparison_chart(
            display_test, "Actual_Generation_kWh", "Forecast_Generation_kWh",
            "Daily Actual vs Forecast Generation", "Generation (kWh)"
        )

        # Rainfall is shown as one measured input series, not as a fake
        # actual-vs-forecast comparison.
        rain_fig = go.Figure()
        rain_fig.add_trace(go.Bar(
            x=display_test["Date"], y=display_test["Actual_Rainfall_mm"],
            name="Measured rainfall", marker_color="#16866c"
        ))
        rain_fig.update_layout(
            title="Daily Measured Rainfall Used by the Model",
            height=330, margin=dict(l=10, r=10, t=55, b=10),
            paper_bgcolor="white", plot_bgcolor="white",
            yaxis_title="Rainfall (mm)", xaxis_title="", hovermode="x unified"
        )
        st.plotly_chart(rain_fig, use_container_width=True)

        daily_view = display_test[[
            "Date", "Actual_Generation_kWh", "Forecast_Generation_kWh",
            "Generation_Difference_kWh", "Absolute_Error_kWh", "Error_Percentage",
            "Actual_Rainfall_mm"
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
            display_test.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_site}_daily_test_comparison.csv",
            mime="text/csv",
        )

    with monthly_tab:
        monthly = display_test.assign(
            Month=display_test["Date"].dt.to_period("M").astype(str)
        ).groupby("Month", as_index=False).agg(
            Days_Compared=("Date", "count"),
            Actual_Generation_kWh=("Actual_Generation_kWh", "sum"),
            Forecast_Generation_kWh=("Forecast_Generation_kWh", "sum"),
            Measured_Rainfall_mm=("Actual_Rainfall_mm", lambda x: x.sum(min_count=1)),
        )
        monthly["Generation_Difference_kWh"] = (
            monthly["Forecast_Generation_kWh"] - monthly["Actual_Generation_kWh"]
        )
        monthly_denom = monthly["Actual_Generation_kWh"].abs().replace(0, np.nan)
        monthly["Error_Percentage"] = (
            100 * monthly["Generation_Difference_kWh"].abs() / monthly_denom
        )

        comparison_chart(
            monthly, "Actual_Generation_kWh", "Forecast_Generation_kWh",
            "Monthly Actual vs Forecast Generation (test days)", "Generation (kWh)", bars=True
        )
        rain_monthly_fig = go.Figure()
        rain_monthly_fig.add_trace(go.Bar(
            x=monthly["Month"], y=monthly["Measured_Rainfall_mm"],
            name="Measured rainfall", marker_color="#16866c"
        ))
        rain_monthly_fig.update_layout(
            title="Monthly Measured Rainfall (test days)",
            height=330, margin=dict(l=10, r=10, t=55, b=10),
            paper_bgcolor="white", plot_bgcolor="white",
            yaxis_title="Rainfall (mm)", xaxis_title="", hovermode="x unified"
        )
        st.plotly_chart(rain_monthly_fig, use_container_width=True)
        st.dataframe(monthly, use_container_width=True, hide_index=True)
        st.caption(
            "Monthly totals include only test-period days displayed above. "
            "Monthly error is calculated from monthly generation totals."
        )
        st.download_button(
            "Download Monthly Comparison CSV",
            monthly.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_site}_monthly_test_comparison.csv",
            mime="text/csv",
        )

st.download_button("Download Upcoming Forecast CSV", future_comparison.to_csv(index=False).encode(),
                   file_name=f"{selected_site}_future_forecast.csv", mime="text/csv")
st.caption("Forecast accuracy is measured from historical replay, not guaranteed for future dates. "
           "Improve accuracy by updating recent actual generation and rainfall, using verified near-term weather "
           "forecasts, and retraining/validating the plant models on newer data.")
