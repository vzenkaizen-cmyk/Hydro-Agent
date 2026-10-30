from __future__ import annotations

from pathlib import Path
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


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


@st.cache_resource
def load_model(site: str):
    model_path = MODEL_DIR / f"{site}_xgb_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found for {site}: {model_path}")
    model = joblib.load(model_path)
    if hasattr(model, "n_features_in_") and model.n_features_in_ != len(FEATURES):
        raise ValueError(
            f"{site} model expects {model.n_features_in_} features, "
            f"but this app supplies {len(FEATURES)}. Retrain the model with "
            "the supplied training script before deploying."
        )
    return model


def create_feature_row(
    future_date: pd.Timestamp,
    history: pd.DataFrame,
    rainfall: float,
) -> pd.DataFrame:
    """Create one forecast row using the same feature definitions as training."""
    history = history.sort_values("Date").reset_index(drop=True)

    def value_at_lag(column: str, days: int, fallback: float) -> float:
        target_date = future_date - pd.Timedelta(days=days)
        match = history.loc[history["Date"].eq(target_date), column].dropna()
        if not match.empty:
            return float(match.iloc[-1])
        prior = history.loc[history["Date"].lt(future_date), column].dropna()
        if not prior.empty:
            return float(prior.tail(days).mean())
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
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Bridge any historical-data gap recursively, then return requested dates."""
    site_history = data.loc[data["Site"].eq(site), [
        "Date", "Generation_kWh", "Rainfall_Input_mm"
    ]].copy().sort_values("Date").reset_index(drop=True)

    if site_history.empty:
        raise ValueError(f"No historical records found for {site}.")

    model = load_model(site)
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


def monthly_comparison(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["Month"] = frame["Date"].dt.to_period("M").astype(str)
    # Aggregate only days with actual generation for fair monthly comparisons.
    paired = frame.loc[frame["Actual_Generation_kWh"].notna()].copy()
    if paired.empty:
        return pd.DataFrame()
    paired["Rainfall_Pair"] = paired["Actual_Rainfall_mm"].notna()
    paired["Forecast_Rainfall_Paired_mm"] = paired["Forecast_Rainfall_mm"].where(paired["Rainfall_Pair"])
    result = paired.groupby("Month", as_index=False).agg(
        Days_Compared=("Date", "count"),
        Actual_Generation_kWh=("Actual_Generation_kWh", "sum"),
        Forecast_Generation_kWh=("Forecast_Generation_kWh", "sum"),
        Actual_Rainfall_mm=("Actual_Rainfall_mm", lambda x: x.sum(min_count=1)),
        Forecast_Rainfall_mm=("Forecast_Rainfall_Paired_mm", lambda x: x.sum(min_count=1)),
    )
    result["Generation_Difference_kWh"] = result["Forecast_Generation_kWh"] - result["Actual_Generation_kWh"]
    denom = result["Actual_Generation_kWh"].where(result["Actual_Generation_kWh"].ne(0))
    result["Error_Percentage"] = 100 * result["Generation_Difference_kWh"].abs() / denom
    result["Rainfall_Difference_mm"] = result["Forecast_Rainfall_mm"] - result["Actual_Rainfall_mm"]
    return result


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
    st.caption("Validation simulates forecasts from a past cutoff date, then compares them with actual measurements.")
    st.divider()
    st.caption("Forecast inputs: plant-specific XGBoost model, historical generation and monthly rainfall baseline")

site_history = historical.loc[historical["Site"].eq(selected_site)].copy().sort_values("Date")
today = pd.Timestamp(date.today())
tomorrow = today + pd.Timedelta(days=1)

# Forecast future dates using existing plant model and recursive feature generation.
try:
    with st.spinner("Calculating upcoming generation forecasts..."):
        forecast_df, last_actual_date = forecast_plant(historical, selected_site, tomorrow, forecast_days)
except Exception as exc:
    st.error(f"Unable to forecast {selected_site}: {exc}")
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

# Genuine out-of-sample-style historical replay: remove validation dates before forecasting.
st.subheader("Historical Actual vs Forecast Comparison")
actual_through_today = site_history.loc[site_history["Date"].le(today)].copy()
if len(actual_through_today) < validation_days + 35:
    st.warning("Not enough historical observations for the chosen validation window. "
               "Choose a shorter window or provide more historical data.")
else:
    validation_actual = actual_through_today.tail(validation_days).copy()
    cutoff = validation_actual["Date"].min() - pd.Timedelta(days=1)
    train_history = actual_through_today.loc[actual_through_today["Date"].le(cutoff)].copy()
    if train_history.empty:
        st.warning("No training history before the validation period.")
    else:
        validation_start = validation_actual["Date"].min()
        validation_end = validation_actual["Date"].max()
        validation_horizon = (validation_end - validation_start).days + 1
        try:
            with st.spinner("Replaying historical forecasts and calculating errors..."):
                backtest_input = historical.loc[historical["Site"].ne(selected_site)].copy()
                backtest_input = pd.concat([backtest_input, train_history], ignore_index=True)
                predicted_history, _ = forecast_plant(backtest_input, selected_site,
                                                       validation_start, validation_horizon)
            daily = build_comparison(validation_actual, predicted_history)
            paired = daily.loc[daily["Actual_Generation_kWh"].notna()].copy()
            if paired.empty:
                st.warning("No matching actual generation measurements were found for validation.")
            else:
                mae = paired["Absolute_Error_kWh"].mean()
                rmse = np.sqrt(np.mean(paired["Generation_Difference_kWh"] ** 2))
                nonzero = paired["Error_Percentage"].replace([np.inf,-np.inf], np.nan).dropna()
                mape = nonzero.mean() if not nonzero.empty else np.nan
                wape_den = paired["Actual_Generation_kWh"].abs().sum()
                wape = 100 * paired["Absolute_Error_kWh"].sum() / wape_den if wape_den else np.nan
                st.caption(f"Validation: {validation_start:%d %b %Y} to {validation_end:%d %b %Y} · "
                           f"{len(paired)} matched days · recursive replay without actual generation from validation dates")
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Mean Absolute Error", f"{mae:,.0f} kWh")
                m2.metric("RMSE", f"{rmse:,.0f} kWh")
                m3.metric("Mean Daily Error", f"{mape:.2f}%" if np.isfinite(mape) else "N/A")
                m4.metric("Weighted Error", f"{wape:.2f}%" if np.isfinite(wape) else "N/A")
                st.caption("Error (%) = |Forecast − Actual| / |Actual| × 100. "
                           "Zero-actual days have undefined daily percentage errors and are excluded from MAPE. "
                           "Weighted error = sum of absolute errors / sum of absolute actual generation × 100.")
                daily_tab, monthly_tab = st.tabs(["Daily Comparison", "Monthly Comparison"])
                with daily_tab:
                    comparison_chart(paired, "Actual_Generation_kWh", "Forecast_Generation_kWh",
                                     "Daily Actual vs Forecast Generation", "Generation (kWh)")
                    comparison_chart(paired, "Actual_Rainfall_mm", "Forecast_Rainfall_mm",
                                     "Daily Actual vs Assumed Rainfall", "Rainfall (mm)")
                    daily_view = paired[["Date","Actual_Generation_kWh","Forecast_Generation_kWh",
                                         "Generation_Difference_kWh","Absolute_Error_kWh","Error_Percentage",
                                         "Actual_Rainfall_mm","Forecast_Rainfall_mm","Rainfall_Difference_mm"]].copy()
                    daily_view["Date"] = daily_view["Date"].dt.strftime("%d %b %Y")
                    st.dataframe(daily_view, use_container_width=True, hide_index=True)
                    st.download_button("Download Daily Comparison CSV", daily.to_csv(index=False).encode(),
                                       file_name=f"{selected_site}_daily_validation.csv", mime="text/csv")
                with monthly_tab:
                    monthly = monthly_comparison(paired)
                    comparison_chart(monthly, "Actual_Generation_kWh", "Forecast_Generation_kWh",
                                     "Monthly Actual vs Forecast Generation (matched days)",
                                     "Generation (kWh)", bars=True)
                    comparison_chart(monthly, "Actual_Rainfall_mm", "Forecast_Rainfall_mm",
                                     "Monthly Actual vs Assumed Rainfall (matched days)",
                                     "Rainfall (mm)", bars=True)
                    st.dataframe(monthly, use_container_width=True, hide_index=True)
                    st.caption("Monthly comparisons cover only the matched validation days, not necessarily full calendar months. "
                               "Monthly error uses the difference between monthly generation totals.")
                    st.download_button("Download Monthly Comparison CSV", monthly.to_csv(index=False).encode(),
                                       file_name=f"{selected_site}_monthly_validation.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"Historical validation could not be completed: {exc}")

st.download_button("Download Upcoming Forecast CSV", future_comparison.to_csv(index=False).encode(),
                   file_name=f"{selected_site}_future_forecast.csv", mime="text/csv")
st.caption("Forecast accuracy is measured from historical replay, not guaranteed for future dates. "
           "Improve accuracy by updating recent actual generation and rainfall, using verified near-term weather "
           "forecasts, and retraining/validating the plant models on newer data.")
