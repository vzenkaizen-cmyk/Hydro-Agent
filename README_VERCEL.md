# Hydro Agent — Vercel web migration

This migration keeps the original Streamlit app and Python forecasting modules in the repository. The new website is a Next.js frontend; the forecasting API is FastAPI.

## Deployment architecture

- **Frontend:** Vercel, using the `frontend/` directory as the project root.
- **Python API:** Render (or another persistent Python host), using the repository root.
- **Forecasting:** the API reuses the existing chronological training, validation, feature engineering and recursive forecasting functions extracted from `agent_app.py`.
- **Data/models:** existing `data/` and `models/` directories remain in place.

The Python API is intentionally separated from Vercel Functions because this repository includes pandas, scikit-learn, XGBoost, a ~12 MB dataset and multiple model artifacts. Running model training and inference in short-lived serverless functions can hit package-size and execution limits.

## 1. Deploy the Python API

1. Push this migration branch to GitHub.
2. In Render, create a new **Web Service** and connect the same GitHub repository.
3. Use the `vercel-migration` branch.
4. Set the root directory to the repository root.
5. Build command: `pip install -r backend-requirements.txt`
6. Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
7. Set environment variables:
   - `FRONTEND_ORIGINS` = your Vercel website origin (for example `https://your-project.vercel.app`)
   - `OPENWEATHER_API_KEY` = only if using the separate live-weather agent tools
   - `DATABASE_URL` = only if database features are used by the API
8. Deploy and confirm `https://YOUR-API.onrender.com/api/health` returns `{"status":"ok",...}`.

## 2. Deploy the website to Vercel

1. Import the GitHub repository into Vercel.
2. Select branch `vercel-migration`.
3. Set **Root Directory** to `frontend`.
4. Framework preset: Next.js.
5. Add environment variable:
   - `NEXT_PUBLIC_API_BASE_URL` = `https://YOUR-API.onrender.com`
6. Deploy.

## 3. Local testing

Backend:
```bash
pip install -r backend-requirements.txt
uvicorn api.main:app --reload
```

Frontend:
```bash
cd frontend
npm install
# create .env.local with NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
npm run dev
```

Open `http://localhost:3000`.

## Important preservation notes

- `agent_app.py` is retained unchanged; the existing Streamlit app remains available for rollback/testing.
- `forecast.py`, `db.py`, `agent/`, `models/`, `data/`, and `outputs/` are retained.
- The new API uses the same core `load_data`, `train_site_model`, `create_feature_row`, and `forecast_plant` functions from the original dashboard, removing only Streamlit caching decorators.
- The existing dashboard uses a historical same-month median rainfall baseline for 7–365 day forecasts. It does not use live weather rainfall for this dashboard. The separate `agent/` module still contains the OpenWeather 5-day agent flow.
- No secrets are included. Add API keys and database URLs only in the hosting provider's environment settings.
- The first request for a plant can take longer because the plant-specific model is trained and evaluated. Subsequent calls may be faster while the API process remains warm.
- This is a migration starter. Verify all plants, database-dependent workflows, live-weather agent flow, exports and production performance before switching users from Streamlit.
