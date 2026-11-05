from __future__ import annotations
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from api.service import build_dashboard, load_data

app = FastAPI(title="Hydro Intelligence Agent API", version="1.0.0")
origins = [x.strip() for x in os.getenv("FRONTEND_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "Hydro Intelligence Agent API"}

@app.get("/api/plants")
def plants():
    try:
        data = load_data()
        return {"sites": sorted(data["Site"].dropna().astype(str).str.upper().unique().tolist())}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/api/dashboard")
def dashboard(site: str = "BBO", forecast_days: int = 30, validation_days: int = 30):
    try:
        return build_dashboard(site, forecast_days, validation_days)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Forecast preparation failed: {exc}")

@app.get("/api/live-agent")
def live_agent(site: str = "BBO"):
    """Run the original OpenWeather + XGBoost short-range hydro agent."""
    try:
        from agent.hydro_agent import forecast as run_agent_forecast
        import pandas as pd
        result, plant, raw_weather = run_agent_forecast(site.strip().upper())
        def records(frame):
            if frame is None or frame.empty:
                return []
            frame = frame.copy()
            for col in frame.columns:
                if pd.api.types.is_datetime64_any_dtype(frame[col]):
                    frame[col] = frame[col].dt.strftime("%Y-%m-%d %H:%M:%S")
            return frame.where(pd.notna(frame), None).to_dict(orient="records")
        return {"site": site.strip().upper(), "plant": plant, "forecast": records(result), "weather": records(raw_weather)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Live weather agent failed: {exc}")
