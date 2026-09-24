import os
from datetime import datetime, timezone
import requests
import pandas as pd
from dotenv import load_dotenv

from .plant_locations import get_plant

load_dotenv()

BASE_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
BASE_GEOCODE_URL = "https://api.openweathermap.org/geo/1.0/direct"

def _api_key():
    key = os.getenv("OPENWEATHER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENWEATHER_API_KEY is missing. Put it in .env locally or "
            "Streamlit Secrets in deployment. Never put the key in GitHub."
        )
    return key

def geocode_plant(site):
    """Resolve missing coordinates through OpenWeather Geocoding."""
    plant = get_plant(site)

    if plant.get("lat") is not None and plant.get("lon") is not None:
        return plant

    params = {
        "q": plant["location_query"],
        "limit": 1,
        "appid": _api_key(),
    }
    r = requests.get(BASE_GEOCODE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    if not data:
        raise RuntimeError(
            f"OpenWeather could not geocode {site}: {plant['location_query']}"
        )

    plant["lat"] = float(data[0]["lat"])
    plant["lon"] = float(data[0]["lon"])
    plant["geocoded_name"] = data[0].get("name")
    plant["geocoded_country"] = data[0].get("country")
    return plant

def get_weather_forecast(site):
    """
    Get OpenWeather's 5-day / 3-hour forecast and aggregate precipitation
    into daily rainfall totals.

    Returns:
        daily dataframe with Date, Rainfall_Input_mm and weather fields.
    """
    plant = geocode_plant(site)

    params = {
        "lat": plant["lat"],
        "lon": plant["lon"],
        "appid": _api_key(),
        "units": os.getenv("OPENWEATHER_UNITS", "metric"),
    }

    r = requests.get(BASE_FORECAST_URL, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()

    rows = []
    for item in payload.get("list", []):
        dt = datetime.fromtimestamp(item["dt"], tz=timezone.utc)
        rain = float(item.get("rain", {}).get("3h", 0.0) or 0.0)
        pop = float(item.get("pop", 0.0) or 0.0)

        rows.append({
            "DateTime_UTC": dt,
            "Date": dt.date(),
            "Rainfall_3h_mm": rain,
            "Rain_Probability": pop,
            "Temperature_C": item.get("main", {}).get("temp"),
            "Humidity_pct": item.get("main", {}).get("humidity"),
            "Pressure_hPa": item.get("main", {}).get("pressure"),
            "Wind_mps": item.get("wind", {}).get("speed"),
            "Weather": (item.get("weather") or [{}])[0].get("description"),
        })

    if not rows:
        raise RuntimeError(f"No forecast records returned for {site}.")

    raw = pd.DataFrame(rows)

    daily = (
        raw.groupby("Date", as_index=False)
        .agg(
            Rainfall_Input_mm=("Rainfall_3h_mm", "sum"),
            Rain_Probability=("Rain_Probability", "max"),
            Temperature_C=("Temperature_C", "mean"),
            Humidity_pct=("Humidity_pct", "mean"),
            Pressure_hPa=("Pressure_hPa", "mean"),
            Wind_mps=("Wind_mps", "mean"),
        )
        .sort_values("Date")
    )

    daily["Date"] = pd.to_datetime(daily["Date"])
    daily["Site"] = site
    daily["Latitude"] = plant["lat"]
    daily["Longitude"] = plant["lon"]

    return daily, raw, plant
