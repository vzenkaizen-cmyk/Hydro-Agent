import os
import sys
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent.plant_locations import PLANTS
from agent.weather_tool import get_weather_forecast
from agent.generation_tool import run_hydro_forecast

import db


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hydro Intelligence Agent",
    page_icon="💧",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


# ============================================================
# ENVIRONMENT
# ============================================================

def get_secret(name):
    value = os.getenv(name)

    if value:
        return str(value).strip()

    try:
        value = st.secrets.get(name)
        if value:
            return str(value).strip()
    except Exception:
        pass

    return None


OPENWEATHER_API_KEY = get_secret("OPENWEATHER_API_KEY")

if OPENWEATHER_API_KEY:
    os.environ["OPENWEATHER_API_KEY"] = OPENWEATHER_API_KEY


# ============================================================
# DATABASE
# ============================================================

if db.db_available():
    try:
        db.init_db(PLANTS)
        DB_STATUS = db.database_status()
    except Exception:
        DB_STATUS = "Connection failed"
else:
    DB_STATUS = "Not configured"



# ============================================================
# LANDING PAGE / HERO
# ============================================================

if "hydro_agent_started" not in st.session_state:
    st.session_state["hydro_agent_started"] = False

if "short_forecast" not in st.session_state:
    st.session_state["short_forecast"] = pd.DataFrame()

if "long_forecast" not in st.session_state:
    st.session_state["long_forecast"] = pd.DataFrame()

if "forecast_site" not in st.session_state:
    st.session_state["forecast_site"] = None


def show_hydro_hero():
    """Professional dark opening screen. Uses HTML intentionally."""
    st.markdown(
        """
        <style>
        .hydro-hero {
            margin: 0.5rem 0 1.5rem 0;
            padding: 58px 64px 48px 64px;
            border-radius: 24px;
            background:
                radial-gradient(circle at 82% 18%, rgba(44, 158, 255, .24), transparent 28%),
                radial-gradient(circle at 12% 88%, rgba(0, 92, 190, .18), transparent 30%),
                linear-gradient(135deg, #061426 0%, #08213b 52%, #0a3157 100%);
            color: #ffffff;
            box-shadow: 0 18px 50px rgba(3, 20, 40, .20);
        }
        .hydro-icon {
            width: 68px;
            height: 68px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 18px;
            background: linear-gradient(135deg, #1688ff, #2357e8);
            box-shadow: 0 10px 30px rgba(22, 136, 255, .35);
            font-size: 34px;
            margin-bottom: 22px;
        }
        .hydro-eyebrow {
            display: inline-block;
            padding: 6px 12px;
            margin-bottom: 15px;
            border: 1px solid rgba(126, 203, 255, .25);
            border-radius: 999px;
            background: rgba(80, 170, 255, .08);
            color: #a9dcff;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1.1px;
        }
        .hydro-title {
            margin: 0;
            font-size: clamp(34px, 4.5vw, 56px);
            line-height: 1.06;
            font-weight: 800;
            letter-spacing: -1.5px;
        }
        .hydro-title span { color: #58b6ff; }
        .hydro-subtitle {
            max-width: 820px;
            margin: 18px 0 0 0;
            color: #c1d3e7;
            font-size: 16px;
            line-height: 1.65;
        }
        .hydro-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 9px;
            margin-top: 25px;
        }
        .hydro-badge {
            padding: 8px 12px;
            border-radius: 10px;
            border: 1px solid rgba(150, 210, 255, .14);
            background: rgba(255,255,255,.055);
            color: #d8eaff;
            font-size: 12px;
        }
        .hydro-bottom {
            margin-top: 27px;
            color: #8aaaca;
            font-size: 12px;
        }
        </style>

        <div class="hydro-hero">
            <div class="hydro-icon">💧</div>
            <div class="hydro-eyebrow">
                AI POWERED • HYDRO INTELLIGENCE • DATA-DRIVEN FORECASTING
            </div>
            <h1 class="hydro-title">
                Hydro <span>Intelligence</span> Agent
            </h1>
            <p class="hydro-subtitle">
                An intelligent forecasting workspace combining real-time
                weather intelligence, XGBoost generation forecasting and
                PostgreSQL analytics to turn hydropower data into
                actionable plant-level insights.
            </p>
            <div class="hydro-badges">
                <div class="hydro-badge">🌧️ OpenWeather Intelligence</div>
                <div class="hydro-badge">⚡ XGBoost Forecasting</div>
                <div class="hydro-badge">📊 365-Day Outlook</div>
                <div class="hydro-badge">🗄️ PostgreSQL Analytics</div>
                <div class="hydro-badge">🏭 Multi-Plant Support</div>
            </div>
            <div class="hydro-bottom">
                Forecast • Analyse • Discover
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_started_header():
    st.markdown(
        """
        <div style="
            padding: 12px 18px;
            margin-bottom: 18px;
            border-radius: 14px;
            background: linear-gradient(90deg, #061426, #0b3157);
            color: white;
            box-shadow: 0 5px 20px rgba(5,30,55,.10);
        ">
            <div style="
                font-size: 11px;
                letter-spacing: 1px;
                color: #8fd0ff;
                font-weight: 700;
                text-transform: uppercase;
            ">
                Hydro Intelligence Agent
            </div>
            <div style="
                font-size: 22px;
                font-weight: 750;
                margin-top: 3px;
            ">
                Forecast with confidence. Plan with intelligence.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Always show the opening screen first.
if not st.session_state["hydro_agent_started"]:
    show_hydro_hero()
    _, hero_button_col, _ = st.columns([1.5, 1, 1.5])
    with hero_button_col:
        if st.button(
            "🚀  Get Started",
            type="primary",
            use_container_width=True,
            key="hero_get_started",
        ):
            st.session_state["hydro_agent_started"] = True
            st.rerun()

    st.markdown(
        """
        <div style="
            text-align:center;
            color:#64748b;
            font-size:12px;
            margin-top:10px;
            margin-bottom:30px;
        ">
            Weather intelligence → Generation prediction → 365-day planning
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def kpi_card(title, value):
    """
    Custom KPI card.
    Uses nowrap + visible overflow so large numbers are not
    displayed as '...' inside narrow Streamlit metric cards.
    """
    st.markdown(
        f"""
        <div style="
            border:1px solid #E5E7EB;
            border-radius:12px;
            padding:16px 18px;
            min-height:105px;
            background:#FFFFFF;
            box-shadow:0 1px 3px rgba(0,0,0,0.06);
        ">
            <div style="
                font-size:14px;
                color:#6B7280;
                margin-bottom:8px;
                white-space:nowrap;
            ">
                {title}
            </div>
            <div style="
                font-size:25px;
                font-weight:650;
                color:#111827;
                white-space:nowrap;
                overflow:visible;
                line-height:1.2;
            ">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clean_forecast_dataframe(df, site=None):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "Date" not in df.columns:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    if "Forecast_Generation_kWh" not in df.columns:
        return pd.DataFrame()

    df["Forecast_Generation_kWh"] = pd.to_numeric(
        df["Forecast_Generation_kWh"],
        errors="coerce",
    )
    df = df.dropna(subset=["Forecast_Generation_kWh"]).copy()

    if "Forecast_Rainfall_mm" in df.columns:
        df["Forecast_Rainfall_mm"] = pd.to_numeric(
            df["Forecast_Rainfall_mm"],
            errors="coerce",
        )

    if site:
        df["Site"] = site

    return df.sort_values("Date").reset_index(drop=True)


def load_csv_long_term(site):
    path = BASE_DIR / "outputs" / "next_year_forecast.csv"

    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    df.columns = df.columns.str.strip()

    site_column = next(
        (
            c
            for c in ["Site", "site", "SITE", "Plant", "plant"]
            if c in df.columns
        ),
        None,
    )

    if site_column:
        if site_column != "Site":
            df = df.rename(columns={site_column: "Site"})

        df = df[
            df["Site"].astype(str).str.upper() == site.upper()
        ].copy()
    elif {
        "Date",
        "Forecast_Generation_kWh",
    }.issubset(df.columns):
        # Single-plant CSV.
        df["Site"] = site
    else:
        return pd.DataFrame()

    return clean_forecast_dataframe(df, site)



def run_long_term_forecast(site):
    """
    Regenerate the rolling 365-day forecast for the selected plant.

    forecast.py must accept the plant/site code as its first command-line
    argument. Running it with the selected site prevents an unrelated plant
    from blocking the current dashboard.
    """
    forecast_script = BASE_DIR / "forecast.py"

    if not forecast_script.exists():
        raise FileNotFoundError(
            f"forecast.py was not found in the project folder: {forecast_script}"
        )

    result = subprocess.run(
        [sys.executable, str(forecast_script), str(site)],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        timeout=900,
    )

    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            "forecast.py failed for "
            f"{site}.\n\n{details[-5000:]}"
        )

    return result.stdout


def save_long_term_to_database(site):
    """
    Save the selected plant's regenerated 365-day CSV forecast to PostgreSQL.
    """
    df = load_csv_long_term(site)

    if df.empty:
        return False

    return db.save_generation_forecast(
        site=site,
        forecast_df=df,
        horizon="365_day",
        source="XGBoost + historical monthly rainfall scenario",
    )

def load_long_term(site):
    """
    Load the latest forecast and keep it available across Streamlit reruns.

    PostgreSQL is preferred. The local CSV remains a fallback.
    """
    current_site = st.session_state.get("forecast_site")

    if (
        current_site == site
        and isinstance(st.session_state.get("long_forecast"), pd.DataFrame)
        and not st.session_state["long_forecast"].empty
    ):
        return st.session_state["long_forecast"].copy()

    if db.db_available():
        try:
            db_df = db.load_latest_generation_forecast(site, "365_day")
            db_df = clean_forecast_dataframe(db_df, site)

            if not db_df.empty:
                st.session_state["long_forecast"] = db_df.copy()
                st.session_state["forecast_site"] = site
                return db_df
        except Exception:
            pass

    csv_df = load_csv_long_term(site)

    if not csv_df.empty:
        st.session_state["long_forecast"] = csv_df.copy()
        st.session_state["forecast_site"] = site

    return csv_df


def load_short_term_from_state(site):
    if (
        st.session_state.get("forecast_site") == site
        and isinstance(st.session_state.get("short_forecast"), pd.DataFrame)
    ):
        return st.session_state["short_forecast"].copy()
    return pd.DataFrame()




def forecast_is_stale(df):
    """
    A rolling 365-day forecast is considered stale when its final
    date is less than 330 days from today.
    """
    if df.empty:
        return True

    last_date = pd.to_datetime(df["Date"], errors="coerce").max()

    if pd.isna(last_date):
        return True

    today = pd.Timestamp.today().normalize()
    days_remaining = (last_date.normalize() - today).days

    return days_remaining < 330



# ============================================================
# FORECAST FILTERS
# ============================================================

def apply_forecast_filters(df, key_prefix="global"):
    """
    One persistent filter panel for the whole dashboard.

    The filtered dataframe is stored in session state, so changing a
    filter causes a Streamlit rerun without losing the generated forecast.
    """
    if df is None or df.empty or "Date" not in df.columns:
        return pd.DataFrame()

    filtered = df.copy()
    filtered["Date"] = pd.to_datetime(filtered["Date"], errors="coerce")
    filtered = filtered.dropna(subset=["Date"]).copy()

    if filtered.empty:
        return filtered

    filtered["Year"] = filtered["Date"].dt.year
    filtered["Month_Number"] = filtered["Date"].dt.month
    filtered["Month_Name"] = filtered["Date"].dt.strftime("%B")

    years = sorted(filtered["Year"].unique().tolist())
    month_numbers = list(range(1, 13))
    month_names = {
        i: pd.Timestamp(year=2000, month=i, day=1).strftime("%B")
        for i in month_numbers
    }

    f1, f2, f3 = st.columns([1.0, 1.25, 1.7])

    with f1:
        selected_years = st.multiselect(
            "Year",
            options=years,
            default=years,
            key=f"{key_prefix}_years",
        )

    with f2:
        selected_months = st.multiselect(
            "Month",
            options=month_numbers,
            default=month_numbers,
            format_func=lambda x: month_names[x],
            key=f"{key_prefix}_months",
        )

    min_date = filtered["Date"].min().date()
    max_date = filtered["Date"].max().date()

    with f3:
        selected_dates = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key=f"{key_prefix}_dates",
        )

    if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
        start_date = pd.Timestamp(selected_dates[0])
        end_date = pd.Timestamp(selected_dates[1])
    else:
        start_date = pd.Timestamp(selected_dates)
        end_date = start_date

    if not selected_years or not selected_months:
        return pd.DataFrame(columns=filtered.columns)

    result = filtered[
        filtered["Year"].isin(selected_years)
        & filtered["Month_Number"].isin(selected_months)
        & (filtered["Date"] >= start_date)
        & (filtered["Date"] <= end_date)
    ].copy()

    return (
        result.sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )


# ============================================================
# HEADER
# ============================================================

st.title("💧 Hydro Intelligence Agent")
st.caption(
    "OpenWeather short-term forecasting + XGBoost generation forecasting "
    "+ PostgreSQL forecast storage"
)

# ============================================================
# API KEY CHECK
# ============================================================

if not OPENWEATHER_API_KEY:
    st.error(
        "⚠️ OpenWeather API key is not configured. "
        "Add OPENWEATHER_API_KEY to your .env file or Streamlit Secrets."
    )
    st.code(
        "OPENWEATHER_API_KEY=YOUR_API_KEY_HERE\n"
        "OPENWEATHER_UNITS=metric"
    )
    st.stop()

# ============================================================
# ACTIVE DASHBOARD HEADER
# ============================================================

show_started_header()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Forecast Controls")

if st.sidebar.button("🏠 Back to Home", use_container_width=True):
    st.session_state["hydro_agent_started"] = False
    st.rerun()

st.sidebar.markdown("---")

site = st.sidebar.selectbox(
    "🏭 Select Hydro Plant",
    list(PLANTS.keys()),
    format_func=lambda x: f"{x} — {PLANTS[x]['name']}",
)

# When the user changes plant, clear only the old plant's in-memory
# forecast so data from another plant cannot appear.
if st.session_state.get("forecast_site") != site:
    st.session_state["short_forecast"] = pd.DataFrame()
    st.session_state["long_forecast"] = pd.DataFrame()
    st.session_state["forecast_site"] = site

run_forecast = st.sidebar.button(
    "🚀 Run Hydro Agent",
    type="primary",
    use_container_width=True,
)

refresh_365 = st.sidebar.button(
    "🔄 Refresh 365-Day Forecast",
    use_container_width=True,
)

st.sidebar.markdown("---")

if DB_STATUS == "Connected":
    st.sidebar.success("🗄️ PostgreSQL: Connected")
else:
    st.sidebar.warning(
        "🗄️ PostgreSQL: Not connected\n\n"
        "Set DATABASE_URL in .env or Streamlit Secrets."
    )

# ============================================================
# PLANT INFORMATION
# ============================================================

plant = PLANTS[site]

st.subheader(f"🏭 {site} — {plant['name']}")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Plant", site)

with c2:
    st.metric(
        "Latitude",
        f"{plant['lat']:.5f}" if plant.get("lat") is not None else "Not configured",
    )

with c3:
    st.metric(
        "Longitude",
        f"{plant['lon']:.5f}" if plant.get("lon") is not None else "Not configured",
    )

# ============================================================
# REFRESH 365-DAY FORECAST
# ============================================================

if refresh_365:
    try:
        with st.spinner("📅 Regenerating rolling 365-day generation forecast..."):
            run_long_term_forecast(site)

            if db.db_available():
                save_long_term_to_database(site)

            # Reload fresh data into session state.
            fresh_long = load_csv_long_term(site)
            if fresh_long.empty:
                fresh_long = load_long_term(site)

            st.session_state["long_forecast"] = fresh_long.copy()
            st.session_state["forecast_site"] = site
            st.cache_data.clear()

        st.success(f"✅ 365-day forecast refreshed for {site}.")
    except Exception as e:
        st.error("❌ 365-day forecast refresh failed.")
        st.exception(e)

# ============================================================
# RUN AGENT
# ============================================================

if run_forecast:
    try:
        with st.spinner("🌧️ Getting upcoming weather forecast..."):
            weather_daily, raw_weather, plant_data = get_weather_forecast(site)

        if db.db_available():
            try:
                db.save_weather_forecast(
                    site=site,
                    weather_daily=weather_daily,
                    raw_weather=raw_weather,
                )
            except Exception as db_error:
                st.warning(
                    f"Weather was retrieved, but database storage failed: {db_error}"
                )

        with st.spinner("⚡ Predicting upcoming generation..."):
            short_forecast = run_hydro_forecast(
                site=site,
                weather_daily=weather_daily,
                data_path="data/HydroGen_Model_Ready_Dataset.csv",
            )

        short_forecast = clean_forecast_dataframe(short_forecast, site)

        st.session_state["short_forecast"] = short_forecast.copy()
        st.session_state["forecast_site"] = site

        if db.db_available() and not short_forecast.empty:
            try:
                db.save_generation_forecast(
                    site=site,
                    forecast_df=short_forecast,
                    horizon="short_term",
                    source="OpenWeather + XGBoost",
                )
            except Exception as db_error:
                st.warning(
                    f"Forecast was generated, but database storage failed: {db_error}"
                )

        long_forecast = load_long_term(site)

        if long_forecast.empty or forecast_is_stale(long_forecast):
            with st.spinner("📅 Updating the rolling 365-day forecast..."):
                try:
                    run_long_term_forecast(site)

                    if db.db_available():
                        save_long_term_to_database(site)

                    long_forecast = load_csv_long_term(site)
                    st.session_state["long_forecast"] = long_forecast.copy()
                    st.session_state["forecast_site"] = site

                except Exception as long_error:
                    if long_forecast.empty:
                        raise long_error
                    st.warning(
                        "The existing long-term forecast is being shown "
                        "because the latest 365-day refresh failed."
                    )

        st.success("✅ Hydro Intelligence Agent completed successfully.")

    except Exception as e:
        st.error("❌ Hydro Intelligence Agent failed.")
        st.exception(e)

# ============================================================
# ALWAYS LOAD PERSISTED FORECASTS
# IMPORTANT:
# Streamlit reruns the entire script whenever a filter changes.
# Therefore charts must NOT be inside `if run_forecast:`.
# ============================================================

short_forecast = load_short_term_from_state(site)
long_forecast = load_long_term(site)

# ============================================================
# GLOBAL FORECAST FILTERS
# One filter panel controls all long-term charts.
# ============================================================

st.markdown("---")
st.header("🔎 Forecast Filters")
st.caption(
    "Use Year, Month and Date Range to update the KPIs, charts and "
    "forecast table below. The selected forecast remains available "
    "when filters change."
)

if long_forecast.empty:
    st.info(
        "No 365-day forecast is loaded yet. "
        "Click **Run Hydro Agent** or **Refresh 365-Day Forecast**."
    )
else:
    filtered_long = apply_forecast_filters(
        long_forecast,
        key_prefix=f"{site}_global_filter",
    )

    if filtered_long.empty:
        st.warning("No forecast data matches the selected filters.")
    else:
        # ========================================================
        # FILTERED KPI CARDS
        # ========================================================

        filtered_generation = filtered_long["Forecast_Generation_kWh"]

        filtered_total = filtered_generation.sum()
        filtered_average = filtered_generation.mean()
        filtered_max = filtered_generation.max()
        filtered_days = filtered_long["Date"].nunique()

        a1, a2, a3, a4 = st.columns([1, 1.55, 1.25, 1.25])

        with a1:
            kpi_card("Forecast Days", f"{filtered_days:,}")

        with a2:
            kpi_card("Forecast Generation", f"{filtered_total:,.0f} kWh")

        with a3:
            kpi_card("Average Daily", f"{filtered_average:,.0f} kWh")

        with a4:
            kpi_card("Maximum Daily", f"{filtered_max:,.0f} kWh")

        # ========================================================
        # TABS
        # All tabs use the SAME filtered dataframe.
        # ========================================================

        tab1, tab2, tab3 = st.tabs(
            [
                "🌧️ Upcoming Days",
                "📅 Next 12 Months",
                "📊 Monthly Outlook",
            ]
        )

        # ========================================================
        # TAB 1 — UPCOMING DAYS
        # ========================================================

        with tab1:
            st.header("🌧️ Upcoming Days Forecast")
            st.caption("OpenWeather rainfall forecast → XGBoost generation")

            if short_forecast.empty:
                st.info(
                    "No short-term forecast is loaded. "
                    "Click **Run Hydro Agent**."
                )
            else:
                total_gen = short_forecast["Forecast_Generation_kWh"].sum()
                avg_gen = short_forecast["Forecast_Generation_kWh"].mean()

                total_rain = (
                    short_forecast["Forecast_Rainfall_mm"].sum()
                    if "Forecast_Rainfall_mm" in short_forecast.columns
                    else 0
                )

                k1, k2, k3 = st.columns(3)

                with k1:
                    kpi_card("Forecast Generation", f"{total_gen:,.0f} kWh")

                with k2:
                    kpi_card("Average Daily Generation", f"{avg_gen:,.0f} kWh")

                with k3:
                    kpi_card("Forecast Rainfall", f"{total_rain:,.1f} mm")

                st.subheader("⚡ Upcoming Generation")

                gen_chart = short_forecast.copy()
                gen_chart["Date"] = pd.to_datetime(gen_chart["Date"])
                gen_chart = (
                    gen_chart.sort_values("Date")
                    .drop_duplicates("Date", keep="last")
                    .set_index("Date")
                )

                st.line_chart(
                    gen_chart[["Forecast_Generation_kWh"]],
                    height=350,
                )

                if "Forecast_Rainfall_mm" in short_forecast.columns:
                    st.subheader("🌧️ Upcoming Rainfall")

                    rain_chart = short_forecast.copy()
                    rain_chart["Date"] = pd.to_datetime(rain_chart["Date"])
                    rain_chart = (
                        rain_chart.sort_values("Date")
                        .drop_duplicates("Date", keep="last")
                        .set_index("Date")
                    )

                    st.line_chart(
                        rain_chart[["Forecast_Rainfall_mm"]],
                        height=300,
                    )

                st.subheader("📋 Upcoming Forecast Details")

                display = short_forecast.copy()
                display["Date"] = pd.to_datetime(display["Date"]).dt.date

                st.dataframe(
                    display,
                    use_container_width=True,
                    hide_index=True,
                )

        # ========================================================
        # TAB 2 — NEXT 12 MONTHS
        # ========================================================

        with tab2:
            st.header("📅 Next 12 Months Generation Forecast")

            st.subheader("⚡ 365-Day Generation Forecast")

            chart = filtered_long[
                ["Date", "Forecast_Generation_kWh"]
            ].copy()

            chart["Date"] = pd.to_datetime(chart["Date"])

            chart = (
                chart.sort_values("Date")
                .drop_duplicates("Date", keep="last")
                .set_index("Date")
            )

            st.line_chart(
                chart[["Forecast_Generation_kWh"]],
                height=420,
            )

            if "Forecast_Rainfall_mm" in filtered_long.columns:
                st.subheader("🌧️ Long-Term Rainfall Input")

                rain = filtered_long[
                    ["Date", "Forecast_Rainfall_mm"]
                ].copy()

                rain["Date"] = pd.to_datetime(rain["Date"])

                rain = (
                    rain.sort_values("Date")
                    .drop_duplicates("Date", keep="last")
                    .set_index("Date")
                )

                st.line_chart(
                    rain[["Forecast_Rainfall_mm"]],
                    height=300,
                )

            st.subheader("📋 Forecast Details")

            display_filtered = filtered_long.copy()
            display_filtered["Date"] = pd.to_datetime(
                display_filtered["Date"]
            ).dt.date

            display_filtered = display_filtered.drop(
                columns=[
                    "Year",
                    "Month_Number",
                    "Month_Name",
                ],
                errors="ignore",
            )

            st.dataframe(
                display_filtered,
                use_container_width=True,
                hide_index=True,
            )

            csv = display_filtered.to_csv(index=False)

            st.download_button(
                "📥 Download Filtered Forecast",
                data=csv,
                file_name=f"{site}_filtered_forecast.csv",
                mime="text/csv",
                key=f"{site}_download_filtered",
            )

        # ========================================================
        # TAB 3 — MONTHLY OUTLOOK
        # ========================================================

        with tab3:
            st.header("📊 Monthly Generation Outlook")

            monthly = (
                filtered_long.assign(
                    Year=filtered_long["Date"].dt.year,
                    Month_Number=filtered_long["Date"].dt.month,
                    Month=filtered_long["Date"].dt.strftime("%b"),
                )
                .groupby(
                    ["Year", "Month_Number", "Month"],
                    as_index=False,
                )["Forecast_Generation_kWh"]
                .sum()
                .sort_values(["Year", "Month_Number"])
            )

            monthly["Year_Month"] = (
                monthly["Year"].astype(str)
                + "-"
                + monthly["Month"]
            )

            st.subheader("📊 Monthly Generation")

            st.bar_chart(
                monthly.set_index("Year_Month")[
                    "Forecast_Generation_kWh"
                ],
                height=420,
            )

            st.dataframe(
                monthly,
                use_container_width=True,
                hide_index=True,
            )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "Hydro Intelligence Agent | OpenWeather + XGBoost + PostgreSQL"
)

