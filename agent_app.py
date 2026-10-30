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


def create_feature_row(future_date: pd.Timestamp, history: pd.DataFrame, rainfall: float) -> pd.DataFrame:
    """Build features with exactly the same shifted rolling definitions as training."""
    history = history.sort_values("Date").reset_index(drop=True)
    prior = history.loc[history["Date"] < future_date].copy()
    if prior.empty:
        raise ValueError("At least one prior observation is required to forecast.")
    gen = pd.to_numeric(prior["Generation_kWh"], errors="coerce")
    rain = pd.to_numeric(prior["Rainfall_Input_mm"], errors="coerce")
    def lag_value(series, lag, fallback):
        value = series.iloc[-lag] if len(series) >= lag else np.nan
        return float(value) if pd.notna(value) else float(fallback)
    gv, rv = gen.dropna(), rain.dropna()
    gf = float(gv.tail(30).mean()) if not gv.empty else 0.0
    rf = float(rv.tail(30).median()) if not rv.empty else 0.0
    gt, rt = gen.tail(30), rain.tail(30)
    doy = int(future_date.dayofyear)
    row = {
        "Rainfall_Input_mm": max(float(rainfall), 0.0),
        "Lag1_Generation_kWh": lag_value(gen, 1, gf),
        "Lag7_Generation_kWh": lag_value(gen, 7, gf),
        "Lag14_Generation_kWh": lag_value(gen, 14, gf),
        "Lag30_Generation_kWh": lag_value(gen, 30, gf),
        "Rolling7_Generation_kWh": float(gt.tail(7).mean()) if gt.notna().any() else gf,
        "Rolling30_Generation_kWh": float(gt.mean()) if gt.notna().any() else gf,
        "Lag1_Rainfall_mm": lag_value(rain, 1, rf),
        "Lag7_Rainfall_mm": lag_value(rain, 7, rf),
        "Lag14_Rainfall_mm": lag_value(rain, 14, rf),
        "Lag30_Rainfall_mm": lag_value(rain, 30, rf),
        # Training uses rolling MEANS, not sums.
        "Rolling7_Rainfall_mm": float(rt.tail(7).mean()) if rt.notna().any() else rf,
        "Rolling30_Rainfall_mm": float(rt.mean()) if rt.notna().any() else rf,
        "Month": int(future_date.month), "DayOfYear": doy,
        "WeekOfYear": int(future_date.isocalendar().week),
        "Sin_DayOfYear": float(np.sin(2*np.pi*doy/365.25)),
        "Cos_DayOfYear": float(np.cos(2*np.pi*doy/365.25)),
        "Rainfall_Coverage_30D": float(rt.count()/30.0),
    }
    return pd.DataFrame([[row[c] for c in FEATURES]], columns=FEATURES).replace([np.inf,-np.inf],np.nan).fillna(0.0)


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

    # Daily seasonal rainfall baseline: median from +/- 7 calendar days
    # across historical years, with monthly median as fallback.
    site_history["DayOfYear"] = site_history["Date"].dt.dayofyear
    rain_values = pd.to_numeric(site_history["Rainfall_Input_mm"], errors="coerce")
    fallback_rain = float(rain_values.median()) if rain_values.notna().any() else 0.0
    month_rain = site_history.assign(Month=site_history["Date"].dt.month, _rain=rain_values).groupby("Month")["_rain"].median().to_dict()
    def historical_daily_rain(target_date):
        doy = int(target_date.dayofyear)
        distance = np.minimum((site_history["DayOfYear"]-doy).abs(), 366-(site_history["DayOfYear"]-doy).abs())
        sample = rain_values.loc[distance <= 7].dropna()
        if not sample.empty:
            return max(0.0, float(sample.median()))
        value = month_rain.get(int(target_date.month), fallback_rain)
        return max(0.0, float(value)) if pd.notna(value) else fallback_rain

    # Fill the model's recursive history up to the requested forecast start.
    # This is important when the latest uploaded actual data is older than today.
    simulation_start = last_actual_date + pd.Timedelta(days=1)
    simulation_end = forecast_end
    all_dates = pd.date_range(simulation_start, simulation_end, freq="D")
    history = site_history.copy()
    results = []

    for forecast_date in all_dates:
        rainfall = historical_daily_rain(forecast_date)
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
                "Forecast_Type": "Historical day-of-year rainfall baseline",
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
# SIDEBAR CONTROLS
# ============================================================
with st.sidebar:
    st.subheader("Forecast Controls")
    selected_site = st.selectbox("Select Hydro Plant", sites, index=sites.index("BBO") if "BBO" in sites else 0)
    forecast_days = st.select_slider(
        "Forecast horizon",
        options=[7, 14, 30, 90, 180, 365],
        value=365,
        format_func=lambda x: f"{x} days",
    )
    run_forecast = st.button("Run Generation Forecast", type="primary", use_container_width=True)
    st.caption("The forecast is recalculated when you change the plant or horizon.")
    st.divider()
    st.caption("Forecast inputs")
    st.write("Historical generation")
    st.write("Historical day-of-year rainfall profile")
    st.write("Plant-specific XGBoost model")


# ============================================================
# RUN FORECAST
# ============================================================
today = pd.Timestamp(date.today())
tomorrow = today + pd.Timedelta(days=1)

try:
    forecast_df, last_actual_date = forecast_plant(
        historical,
        selected_site,
        tomorrow,
        forecast_days,
    )
except Exception as exc:
    st.error(f"Unable to generate the forecast for {selected_site}: {exc}")
    st.stop()

if forecast_df.empty:
    st.warning("No forecast rows were generated. Check the model and historical data.")
    st.stop()

forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])
tomorrow_row = forecast_df.loc[forecast_df["Date"].eq(tomorrow)]
tomorrow_generation = (
    float(tomorrow_row["Forecast_Generation_kWh"].iloc[0])
    if not tomorrow_row.empty else np.nan
)
tomorrow_rainfall = (
    float(tomorrow_row["Forecast_Rainfall_mm"].iloc[0])
    if not tomorrow_row.empty else np.nan
)

# Latest actual data freshness is shown explicitly to prevent a stale-data
# forecast from being mistaken for a real-time prediction.
data_gap_days = max((today - last_actual_date.normalize()).days, 0)

st.subheader(f"{selected_site} Generation Forecast")
st.markdown(
    f"<p class='section-note'>Latest actual generation record: "
    f"<b>{last_actual_date:%d %B %Y}</b> · Forecast date: "
    f"<b>{tomorrow:%d %B %Y}</b></p>",
    unsafe_allow_html=True,
)

if data_gap_days > 3:
    st.warning(
        f"The latest historical record is {data_gap_days} days old. "
        "The model recursively estimates the gap before forecasting tomorrow. "
        "Update the dataset with the latest actual generation and rainfall "
        "measurements for a more dependable operational forecast."
    )


# ============================================================
# TOMORROW'S GENERATION FORECAST
# ============================================================
st.subheader("Tomorrow's Generation Forecast")
st.markdown(
    f"<p class='section-note'>Forecast for {tomorrow:%A, %d %B %Y} "
    f"using the selected plant's XGBoost model.</p>",
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Tomorrow's Generation",
    f"{tomorrow_generation:,.0f} kWh" if np.isfinite(tomorrow_generation) else "N/A",
)
k2.metric(
    "Expected Rainfall",
    f"{tomorrow_rainfall:,.1f} mm" if np.isfinite(tomorrow_rainfall) else "N/A",
)
k3.metric("Forecast Horizon", f"{forecast_days} days")
k4.metric("Latest Actual Date", last_actual_date.strftime("%d %b %Y"))

st.info(
    "Rainfall assumption: historical median rainfall for nearby calendar days "
    "across previous years at this plant. This project archive does not contain a working live "
    "weather connector, so this is a baseline rainfall scenario rather than "
    "an OpenWeather forecast."
)


# ============================================================
# 365-DAY GENERATION FORECAST
# ============================================================
st.subheader(f"{forecast_days}-Day Generation Forecast")
st.markdown(
    "<p class='section-note'>Daily predicted generation for the selected hydro plant.</p>",
    unsafe_allow_html=True,
)

chart = go.Figure()
chart.add_trace(
    go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Forecast_Generation_kWh"],
        mode="lines",
        name="Predicted generation",
        line=dict(width=2.7, color="#1677b8"),
        hovertemplate="%{x|%d %b %Y}<br>Generation: %{y:,.0f} kWh<extra></extra>",
    )
)
chart.add_vline(
    x=tomorrow,
    line_width=1.5,
    line_dash="dash",
    line_color="#e58a27",
)
chart.update_layout(
    height=390,
    margin=dict(l=10, r=10, t=25, b=10),
    paper_bgcolor="white",
    plot_bgcolor="white",
    hovermode="x unified",
    legend=dict(orientation="h", y=1.12, x=0),
    xaxis=dict(title="", showgrid=True, gridcolor="#e8edf2"),
    yaxis=dict(title="Generation (kWh)", rangemode="tozero", showgrid=True, gridcolor="#e8edf2"),
)
st.plotly_chart(chart, use_container_width=True)


# ============================================================
# MONTHLY SUMMARY
# ============================================================
st.subheader("Monthly Generation Forecast")
monthly = (
    forecast_df.assign(Month=forecast_df["Date"].dt.to_period("M").astype(str))
    .groupby("Month", as_index=False)
    .agg(
        Forecast_Generation_kWh=("Forecast_Generation_kWh", "sum"),
        Average_Daily_Generation_kWh=("Forecast_Generation_kWh", "mean"),
        Forecast_Rainfall_mm=("Forecast_Rainfall_mm", "sum"),
    )
)
monthly_chart = go.Figure()
monthly_chart.add_trace(
    go.Bar(
        x=monthly["Month"],
        y=monthly["Forecast_Generation_kWh"],
        name="Monthly generation",
        marker_color="#3a91c8",
        hovertemplate="%{x}<br>Generation: %{y:,.0f} kWh<extra></extra>",
    )
)
monthly_chart.update_layout(
    height=320,
    margin=dict(l=10, r=10, t=15, b=10),
    paper_bgcolor="white",
    plot_bgcolor="white",
    xaxis_title="Month",
    yaxis_title="Forecast generation (kWh)",
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor="#e8edf2", rangemode="tozero"),
)
st.plotly_chart(monthly_chart, use_container_width=True)

# ============================================================
# FORECAST TABLE AND DOWNLOAD
# ============================================================
st.subheader("Daily Forecast Details")
display_df = forecast_df.copy()
display_df["Date"] = display_df["Date"].dt.strftime("%d %b %Y")
display_df = display_df.rename(columns={
    "Site": "Plant",
    "Date": "Forecast Date",
    "Forecast_Generation_kWh": "Forecast Generation (kWh)",
    "Forecast_Rainfall_mm": "Assumed Rainfall (mm)",
    "Forecast_Type": "Rainfall Scenario",
})
st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Forecast Generation (kWh)": st.column_config.NumberColumn(format="%,.0f"),
        "Assumed Rainfall (mm)": st.column_config.NumberColumn(format="%.1f"),
    },
)

csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Forecast as CSV",
    data=csv_bytes,
    file_name=f"{selected_site}_generation_forecast_{tomorrow:%Y%m%d}.csv",
    mime="text/csv",
)

st.caption(
    "Model outputs are estimates, not guaranteed generation values. "
    "Forecast quality depends on recent, complete historical records, "
    "rainfall inputs and validation performance for each plant."
)
