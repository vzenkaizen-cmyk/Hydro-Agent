"""
Hydro Forecasting Agent.

This is a lightweight tool-calling agent:
1. Identify plant location.
2. Retrieve OpenWeather forecast.
3. Convert 3-hour precipitation into daily rainfall.
4. Feed rainfall + historical generation features into the XGBoost model.
5. Return generation and rainfall forecasts.

It is intentionally deterministic: the weather API and XGBoost model perform
the numerical work. An LLM can be added later as a natural-language layer.
"""

import argparse
import os
import pandas as pd

from .weather_tool import get_weather_forecast
from .generation_tool import run_hydro_forecast

def forecast(site, data_path="data/HydroGen_Model_Ready_Dataset.csv"):
    weather_daily, raw_weather, plant = get_weather_forecast(site)
    generation = run_hydro_forecast(
        site=site,
        weather_daily=weather_daily,
        data_path=data_path,
    )

    # generation already contains the rainfall/weather values used for prediction.
    # Keep the weather dataframe separate for auditing/debugging.
    return generation, plant, raw_weather

def print_report(site, result, plant):
    print("\n" + "=" * 72)
    print("HYDRO FORECASTING AGENT")
    print("=" * 72)
    print(f"Plant       : {plant['name']}")
    print(f"Site code   : {site}")
    print(f"Latitude    : {plant['lat']}")
    print(f"Longitude   : {plant['lon']}")
    print(f"Weather     : OpenWeather")
    print(f"Generation  : XGBoost")
    print("-" * 72)

    display = result[
        [
            "Date",
            "Forecast_Rainfall_mm",
            "Rain_Probability",
            "Forecast_Generation_kWh",
        ]
    ].copy()

    display["Date"] = pd.to_datetime(display["Date"]).dt.strftime("%Y-%m-%d")
    display["Forecast_Rainfall_mm"] = display["Forecast_Rainfall_mm"].round(2)
    display["Rain_Probability"] = (
        display["Rain_Probability"].fillna(0).mul(100).round(1)
    )
    display["Forecast_Generation_kWh"] = (
        display["Forecast_Generation_kWh"].round(2)
    )

    print(display.to_string(index=False))

    print("\n" + "-" * 72)
    print("Summary")
    print(f"Forecast days              : {len(result)}")
    print(
        f"Total forecast generation  : "
        f"{result['Forecast_Generation_kWh'].sum():,.2f} kWh"
    )
    print(
        f"Total forecast rainfall    : "
        f"{result['Forecast_Rainfall_mm'].sum():,.2f} mm"
    )
    print("=" * 72)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True, help="Example: BBO")
    parser.add_argument(
        "--data",
        default="data/HydroGen_Model_Ready_Dataset.csv",
        help="Path to model-ready historical dataset",
    )
    parser.add_argument(
        "--output",
        default="outputs/agent_forecast.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    result, plant, raw_weather = forecast(args.site.upper(), args.data)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result.to_csv(args.output, index=False)

    print_report(args.site.upper(), result, plant)
    print(f"\nSaved: {args.output}")

if __name__ == "__main__":
    main()
