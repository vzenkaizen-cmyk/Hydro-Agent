from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from mcp.server.fastmcp import FastMCP

# Reuse the existing forecasting implementation.
# forecast.py expects:
#   data/HydroGen_Model_Ready_Dataset.csv
#   models/<SITE>_xgb_model.pkl
#   models/<SITE>_features.pkl
import forecast as hydro_forecast

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

mcp = FastMCP("Hydro Intelligence MCP Server")


def _json_records(df: pd.DataFrame) -> str:
    """Convert a dataframe to JSON-safe records."""
    if df.empty:
        return json.dumps([])
    clean = df.copy()
    for col in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[col]):
            clean[col] = clean[col].dt.strftime("%Y-%m-%d")
    clean = clean.where(pd.notna(clean), None)
    return json.dumps(clean.to_dict(orient="records"), default=str)


@mcp.tool()
def list_hydro_plants() -> str:
    """List the hydro plant/site codes available in the historical dataset."""
    data = hydro_forecast.load_data()
    sites = sorted(data["Site"].dropna().astype(str).str.upper().unique().tolist())
    return json.dumps({"sites": sites, "count": len(sites)})


@mcp.tool()
def generate_generation_forecast(site: str, horizon_days: int = 30) -> str:
    """
    Generate a plant-specific XGBoost forecast for 1 to 365 days.
    Rainfall values are historical same-month median assumptions, not live
    weather forecasts. Requires the site's saved model and feature pickle files.
    """
    site = str(site).strip().upper()
    if not site:
        raise ValueError("A plant/site code is required.")
    if not 1 <= int(horizon_days) <= 365:
        raise ValueError("horizon_days must be between 1 and 365.")

    data = hydro_forecast.load_data()
    available_sites = set(data["Site"].astype(str).str.upper().unique())
    if site not in available_sites:
        raise ValueError(
            f"Unknown site {site}. Available sites: {', '.join(sorted(available_sites))}"
        )

    # Existing forecast.py creates a 365-day forecast using the saved model.
    result = hydro_forecast.forecast_site(site, data).head(int(horizon_days)).copy()
    if result.empty:
        raise RuntimeError(f"No forecast rows were generated for {site}.")

    # Keep the existing output convention and save the requested result.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{site.lower()}_mcp_forecast.csv"
    result.to_csv(output_path, index=False)

    return json.dumps({
        "site": site,
        "requested_horizon_days": int(horizon_days),
        "returned_days": len(result),
        "rainfall_method": "Historical monthly median rainfall assumption",
        "is_live_weather_forecast": False,
        "saved_file": str(output_path.relative_to(PROJECT_ROOT)),
        "forecast": json.loads(_json_records(result)),
    })


@mcp.tool()
def get_saved_forecast(site: str) -> str:
    """Read the latest saved MCP forecast CSV for a plant/site."""
    site = str(site).strip().upper()
    path = OUTPUT_DIR / f"{site.lower()}_mcp_forecast.csv"
    if not path.exists():
        return json.dumps({
            "site": site,
            "found": False,
            "message": "No MCP forecast has been saved for this site yet."
        })

    df = pd.read_csv(path)
    return json.dumps({
        "site": site,
        "found": True,
        "forecast": json.loads(_json_records(df)),
    })


@mcp.tool()
def get_database_status() -> str:
    """Check whether the existing PostgreSQL database connection is available."""
    try:
        import db
        return json.dumps({
            "database_status": db.database_status(),
            "database_configured": db.db_available(),
        })
    except Exception as exc:
        return json.dumps({
            "database_status": "Error",
            "error": str(exc),
        })


if __name__ == "__main__":
    # Local development: use stdio (the MCP Inspector can launch this process).
    mcp.run()
