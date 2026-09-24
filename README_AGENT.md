# Hydro Forecasting Agent

## What this version does

The agent connects:

**OpenWeather → future rainfall → XGBoost → future generation**

For a selected plant it:

1. Finds the plant coordinates from `agent/plant_locations.py`.
2. Uses OpenWeather Geocoding for locations that do not yet have coordinates.
3. Requests the OpenWeather 5-day / 3-hour forecast.
4. Aggregates precipitation into daily rainfall.
5. Builds the same leakage-safe features used by the XGBoost generation model.
6. Predicts daily generation recursively.
7. Displays rainfall and generation forecasts in Streamlit.
8. Saves a CSV forecast.

## Important

The current OpenWeather 5-day / 3-hour product gives short-range weather data.
Do not label a 365-day rainfall scenario as an exact weather prediction.

For a longer horizon, add a separate climatology/scenario tool rather than
pretending that the short-range forecast is a one-year weather forecast.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.env`:

```text
OPENWEATHER_API_KEY=YOUR_KEY
OPENWEATHER_UNITS=metric
```

Do NOT commit `.env` to GitHub.

## Copy your existing project data/models

```text
data/HydroGen_Model_Ready_Dataset.csv
models/BBO_xgb_model.pkl
models/BTO_xgb_model.pkl
...
```

## Run command-line agent

```powershell
python -m agent.hydro_agent --site BBO
```

Or:

```powershell
python run_agent.py --site BBO
```

Output:

```text
outputs/agent_forecast.csv
```

## Run Streamlit

```powershell
streamlit run agent_app.py
```

## Plant coordinates

A separate location file is NOT required.

`agent/plant_locations.py` contains a plant registry. Some plants use
coordinates obtained from public Vidullanka reports; other plants are
geocoded at runtime from their location names.

Verify all plant-level coordinates before using the forecasts operationally.

## Recommended production architecture

```text
                HYDRO FORECASTING AGENT
                         │
             ┌───────────┴───────────┐
             │                       │
       Plant Registry          Historical Data
             │                       │
             ▼                       ▼
       Coordinates              XGBoost Model
             │                       │
             ▼                       │
       OpenWeather API               │
             │                       │
       Rainfall Forecast             │
             └───────────┬───────────┘
                         ▼
                Generation Forecast
                         │
                         ▼
                  Streamlit Dashboard
```
