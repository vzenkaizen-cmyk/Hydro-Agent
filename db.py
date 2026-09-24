"""
PostgreSQL persistence layer for the Hydro Forecasting Agent.

Required environment variable:
    DATABASE_URL=postgresql://USER:PASSWORD@HOST/DB?sslmode=require

Works with Neon PostgreSQL and other PostgreSQL providers.
"""

import json
import os
from datetime import datetime, timezone
from sqlalchemy.dialects.postgresql import insert

import pandas as pd
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Session


def _database_url():
    url = os.getenv("DATABASE_URL")
    if not url:
        try:
            import streamlit as st
            url = st.secrets.get("DATABASE_URL")
        except Exception:
            url = None

    if not url:
        return None

    url = str(url).strip()
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url


DATABASE_URL = _database_url()

engine = (
    create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=1800,
    )
    if DATABASE_URL
    else None
)


class Base(DeclarativeBase):
    pass


class Plant(Base):
    __tablename__ = "plants"

    id = Column(Integer, primary_key=True)
    site_code = Column(String(20), unique=True, nullable=False)
    plant_name = Column(String(200), nullable=False)
    country = Column(String(10))
    latitude = Column(Float)
    longitude = Column(Float)
    status = Column(String(200))


class WeatherForecast(Base):
    __tablename__ = "weather_forecasts"

    id = Column(Integer, primary_key=True)
    site_code = Column(String(20), nullable=False)
    forecast_date = Column(Date, nullable=False)
    rainfall_mm = Column(Float)
    temperature_c = Column(Float)
    weather_condition = Column(String(100))
    raw_json = Column(Text)
    fetched_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "site_code",
            "forecast_date",
            name="uq_weather_site_date",
        ),
    )


class GenerationForecast(Base):
    __tablename__ = "generation_forecasts"

    id = Column(Integer, primary_key=True)
    site_code = Column(String(20), nullable=False)
    forecast_date = Column(Date, nullable=False)
    horizon = Column(String(30), nullable=False)
    predicted_generation_kwh = Column(Float, nullable=False)
    forecast_rainfall_mm = Column(Float)
    forecast_source = Column(String(100))
    created_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "site_code",
            "forecast_date",
            "horizon",
            name="uq_generation_site_date_horizon",
        ),
    )


def db_available():
    return engine is not None


def init_db(plants):
    """Create tables and upsert plant registry."""
    if engine is None:
        return False

    Base.metadata.create_all(engine)

    with Session(engine) as session:
        for code, plant in plants.items():
            existing = session.scalar(
                select(Plant).where(Plant.site_code == code)
            )

            if existing is None:
                existing = Plant(site_code=code)
                session.add(existing)

            existing.plant_name = plant.get("name", code)
            existing.country = plant.get("country")
            existing.latitude = plant.get("lat")
            existing.longitude = plant.get("lon")
            existing.status = plant.get("status")

        session.commit()

    return True


def save_weather_forecast(site, weather_daily, raw_weather=None):
    """Store daily OpenWeather data. Existing site/date rows are updated."""
    if engine is None or weather_daily is None:
        return False

    df = pd.DataFrame(weather_daily).copy()
    if df.empty:
        return False

    df.columns = [str(c).strip() for c in df.columns]

    date_col = next(
        (c for c in ["Date", "date", "forecast_date"] if c in df.columns),
        None,
    )
    rain_col = next(
        (
            c
            for c in [
                "Forecast_Rainfall_mm",
                "Rainfall_mm",
                "rainfall_mm",
                "rain",
                "precipitation",
            ]
            if c in df.columns
        ),
        None,
    )
    temp_col = next(
        (
            c
            for c in [
                "Temperature_C",
                "temperature_c",
                "Temp_C",
                "temperature",
            ]
            if c in df.columns
        ),
        None,
    )
    condition_col = next(
        (
            c
            for c in [
                "Weather_Condition",
                "weather_condition",
                "Condition",
                "condition",
            ]
            if c in df.columns
        ),
        None,
    )

    if date_col is None:
        return False

    now = datetime.now(timezone.utc)

    raw_by_date = {}
    if isinstance(raw_weather, dict):
        # Keep raw payload once per forecast date when possible.
        raw_by_date = raw_weather

    with Session(engine) as session:
        for _, row in df.iterrows():
            dt = pd.to_datetime(row[date_col], errors="coerce")
            if pd.isna(dt):
                continue

            forecast_date = dt.date()

            existing = session.scalar(
                select(WeatherForecast).where(
                    WeatherForecast.site_code == str(site).upper(),
                    WeatherForecast.forecast_date == forecast_date,
                )
            )

            if existing is None:
                existing = WeatherForecast(
                    site_code=str(site).upper(),
                    forecast_date=forecast_date,
                )
                session.add(existing)

            existing.rainfall_mm = (
                float(row[rain_col]) if rain_col and pd.notna(row[rain_col]) else None
            )
            existing.temperature_c = (
                float(row[temp_col]) if temp_col and pd.notna(row[temp_col]) else None
            )
            existing.weather_condition = (
                str(row[condition_col]) if condition_col and pd.notna(row[condition_col]) else None
            )
            existing.raw_json = json.dumps(raw_by_date, default=str) if raw_by_date else None
            existing.fetched_at = now

        session.commit()

    return True


def save_generation_forecast(site, forecast_df, horizon, source):
    """
    Store generation predictions in PostgreSQL using an atomic UPSERT.

    The unique key is:
        site_code + forecast_date + horizon

    If a row already exists, its forecast values are updated instead of
    attempting a duplicate INSERT. This makes the 365-day refresh safe to
    run repeatedly and avoids SQLAlchemy autoflush/UniqueViolation errors.
    """
    if engine is None or forecast_df is None:
        return False

    df = pd.DataFrame(forecast_df).copy()

    if df.empty or "Date" not in df.columns:
        return False

    generation_col = "Forecast_Generation_kWh"

    if generation_col not in df.columns:
        return False

    rain_col = (
        "Forecast_Rainfall_mm"
        if "Forecast_Rainfall_mm" in df.columns
        else None
    )

    site = str(site).upper().strip()
    horizon = str(horizon).strip()
    source = str(source) if source is not None else None
    now = datetime.now(timezone.utc)

    # ---------------------------------------------------------
    # Prepare clean rows before touching PostgreSQL.
    # Duplicate dates inside the same forecast are reduced to the
    # last valid occurrence so one refresh cannot create conflicts.
    # ---------------------------------------------------------
    rows_to_upsert = []

    for _, row in df.iterrows():
        dt = pd.to_datetime(row["Date"], errors="coerce")

        if pd.isna(dt):
            continue

        generation = pd.to_numeric(
            row[generation_col],
            errors="coerce",
        )

        if pd.isna(generation):
            continue

        rainfall = None

        if rain_col and pd.notna(row[rain_col]):
            rainfall_value = pd.to_numeric(
                row[rain_col],
                errors="coerce",
            )

            if pd.notna(rainfall_value):
                rainfall = float(rainfall_value)

        rows_to_upsert.append(
            {
                "site_code": site,
                "forecast_date": dt.date(),
                "horizon": horizon,
                "predicted_generation_kwh": float(generation),
                "forecast_rainfall_mm": rainfall,
                "forecast_source": source,
                "created_at": now,
            }
        )

    if not rows_to_upsert:
        return False

    # ---------------------------------------------------------
    # Remove duplicate (site, date, horizon) rows inside the
    # incoming dataframe before sending them to PostgreSQL.
    # ---------------------------------------------------------
    unique_rows = {}

    for record in rows_to_upsert:
        key = (
            record["site_code"],
            record["forecast_date"],
            record["horizon"],
        )
        unique_rows[key] = record

    rows_to_upsert = list(unique_rows.values())

    # ---------------------------------------------------------
    # PostgreSQL atomic UPSERT.
    #
    # This avoids:
    #   1. SELECT -> INSERT race conditions
    #   2. SQLAlchemy query-triggered autoflush
    #   3. duplicate-key errors when Refresh is clicked again
    # ---------------------------------------------------------
    stmt = insert(GenerationForecast).values(rows_to_upsert)

    stmt = stmt.on_conflict_do_update(
        constraint="uq_generation_site_date_horizon",
        set_={
            "predicted_generation_kwh":
                stmt.excluded.predicted_generation_kwh,
            "forecast_rainfall_mm":
                stmt.excluded.forecast_rainfall_mm,
            "forecast_source":
                stmt.excluded.forecast_source,
            "created_at":
                stmt.excluded.created_at,
        },
    )

    try:
        with Session(engine) as session:
            session.execute(stmt)
            session.commit()

        return True

    except Exception:
        # The context manager closes the session; rollback explicitly
        # when possible so the transaction is never left pending.
        try:
            session.rollback()
        except Exception:
            pass

        raise


def load_latest_generation_forecast(site, horizon):
    """Return the latest stored forecast for a plant."""
    if engine is None:
        return pd.DataFrame()

    site = str(site).upper()

    with Session(engine) as session:
        rows = session.scalars(
            select(GenerationForecast)
            .where(
                GenerationForecast.site_code == site,
                GenerationForecast.horizon == horizon,
            )
            .order_by(
                GenerationForecast.created_at.desc(),
                GenerationForecast.forecast_date.asc(),
            )
        ).all()

    if not rows:
        return pd.DataFrame()

    # Keep only the newest creation batch.
    latest_created = max(r.created_at for r in rows)
    rows = [r for r in rows if r.created_at == latest_created]

    return pd.DataFrame(
        [
            {
                "Site": r.site_code,
                "Date": r.forecast_date,
                "Forecast_Generation_kWh": r.predicted_generation_kwh,
                "Forecast_Rainfall_mm": r.forecast_rainfall_mm,
                "Forecast_Source": r.forecast_source,
                "Created_At": r.created_at,
            }
            for r in rows
        ]
    )


def database_status():
    """Small status helper for the Streamlit UI."""
    if engine is None:
        return "Not configured"

    try:
        with engine.connect() as connection:
            connection.execute(select(Plant).limit(1))
        return "Connected"
    except Exception:
        return "Connection failed"
