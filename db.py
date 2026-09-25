"""
PostgreSQL persistence layer for the Hydro Forecasting Agent.

Supports:
- Neon PostgreSQL
- Streamlit Cloud
- Local development
- psycopg PostgreSQL driver

Required environment variable / Streamlit secret:

DATABASE_URL=postgresql://USER:PASSWORD@HOST/DB?sslmode=require
"""

import json
import os
from datetime import datetime, timezone

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

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import DeclarativeBase, Session


# ============================================================
# DATABASE URL
# ============================================================

def _database_url():
    """
    Get DATABASE_URL from:
    1. Environment variable
    2. Streamlit Secrets
    """

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

    # --------------------------------------------------------
    # Convert old postgres:// format
    # --------------------------------------------------------

    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]

    # --------------------------------------------------------
    # IMPORTANT:
    # Explicitly use psycopg 3.
    # --------------------------------------------------------

    if url.startswith("postgresql://"):
        url = url.replace(
            "postgresql://",
            "postgresql+psycopg://",
            1
        )

    return url


DATABASE_URL = _database_url()


# ============================================================
# DATABASE ENGINE
# ============================================================

engine = None

if DATABASE_URL:

    try:

        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            pool_recycle=1800,
        )

    except Exception as e:

        print(f"Database engine creation failed: {e}")

        engine = None


# ============================================================
# BASE
# ============================================================

class Base(DeclarativeBase):
    pass


# ============================================================
# PLANT TABLE
# ============================================================

class Plant(Base):

    __tablename__ = "plants"

    id = Column(
        Integer,
        primary_key=True
    )

    site_code = Column(
        String(20),
        unique=True,
        nullable=False
    )

    plant_name = Column(
        String(200),
        nullable=False
    )

    country = Column(
        String(10)
    )

    latitude = Column(
        Float
    )

    longitude = Column(
        Float
    )

    status = Column(
        String(200)
    )


# ============================================================
# WEATHER FORECAST TABLE
# ============================================================

class WeatherForecast(Base):

    __tablename__ = "weather_forecasts"

    id = Column(
        Integer,
        primary_key=True
    )

    site_code = Column(
        String(20),
        nullable=False
    )

    forecast_date = Column(
        Date,
        nullable=False
    )

    rainfall_mm = Column(
        Float
    )

    temperature_c = Column(
        Float
    )

    weather_condition = Column(
        String(100)
    )

    raw_json = Column(
        Text
    )

    fetched_at = Column(
        DateTime(timezone=True),
        nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "site_code",
            "forecast_date",
            name="uq_weather_site_date",
        ),
    )


# ============================================================
# GENERATION FORECAST TABLE
# ============================================================

class GenerationForecast(Base):

    __tablename__ = "generation_forecasts"

    id = Column(
        Integer,
        primary_key=True
    )

    site_code = Column(
        String(20),
        nullable=False
    )

    forecast_date = Column(
        Date,
        nullable=False
    )

    horizon = Column(
        String(30),
        nullable=False
    )

    predicted_generation_kwh = Column(
        Float,
        nullable=False
    )

    forecast_rainfall_mm = Column(
        Float
    )

    forecast_source = Column(
        String(100)
    )

    created_at = Column(
        DateTime(timezone=True),
        nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "site_code",
            "forecast_date",
            "horizon",
            name="uq_generation_site_date_horizon",
        ),
    )


# ============================================================
# DATABASE AVAILABILITY
# ============================================================

def db_available():

    return engine is not None


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def init_db(plants):

    if engine is None:
        return False

    Base.metadata.create_all(engine)

    with Session(engine) as session:

        for code, plant in plants.items():

            existing = session.scalar(
                select(Plant).where(
                    Plant.site_code == code
                )
            )

            if existing is None:

                existing = Plant(
                    site_code=code
                )

                session.add(existing)

            existing.plant_name = plant.get(
                "name",
                code
            )

            existing.country = plant.get(
                "country"
            )

            existing.latitude = plant.get(
                "lat"
            )

            existing.longitude = plant.get(
                "lon"
            )

            existing.status = plant.get(
                "status"
            )

        session.commit()

    return True


# ============================================================
# SAVE WEATHER FORECAST
# ============================================================

def save_weather_forecast(
    site,
    weather_daily,
    raw_weather=None
):

    if engine is None or weather_daily is None:
        return False

    df = pd.DataFrame(
        weather_daily
    ).copy()

    if df.empty:
        return False

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    date_col = next(
        (
            c
            for c in [
                "Date",
                "date",
                "forecast_date"
            ]
            if c in df.columns
        ),
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

    now = datetime.now(
        timezone.utc
    )

    raw_by_date = {}

    if isinstance(
        raw_weather,
        dict
    ):
        raw_by_date = raw_weather

    with Session(engine) as session:

        for _, row in df.iterrows():

            dt = pd.to_datetime(
                row[date_col],
                errors="coerce"
            )

            if pd.isna(dt):
                continue

            forecast_date = dt.date()

            existing = session.scalar(
                select(
                    WeatherForecast
                ).where(
                    WeatherForecast.site_code
                    == str(site).upper(),

                    WeatherForecast.forecast_date
                    == forecast_date,
                )
            )

            if existing is None:

                existing = WeatherForecast(
                    site_code=str(
                        site
                    ).upper(),

                    forecast_date=forecast_date,
                )

                session.add(existing)

            existing.rainfall_mm = (

                float(row[rain_col])
                if rain_col
                and pd.notna(
                    row[rain_col]
                )
                else None
            )

            existing.temperature_c = (

                float(row[temp_col])
                if temp_col
                and pd.notna(
                    row[temp_col]
                )
                else None
            )

            existing.weather_condition = (

                str(row[condition_col])
                if condition_col
                and pd.notna(
                    row[condition_col]
                )
                else None
            )

            existing.raw_json = (

                json.dumps(
                    raw_by_date,
                    default=str
                )
                if raw_by_date
                else None
            )

            existing.fetched_at = now

        session.commit()

    return True


# ============================================================
# SAVE GENERATION FORECAST
# ============================================================

def save_generation_forecast(
    site,
    forecast_df,
    horizon,
    source
):

    if engine is None or forecast_df is None:
        return False

    df = pd.DataFrame(
        forecast_df
    ).copy()

    if (
        df.empty
        or "Date" not in df.columns
    ):
        return False

    generation_col = (
        "Forecast_Generation_kWh"
    )

    if generation_col not in df.columns:
        return False

    rain_col = (

        "Forecast_Rainfall_mm"

        if "Forecast_Rainfall_mm"
        in df.columns

        else None
    )

    site = str(
        site
    ).upper().strip()

    horizon = str(
        horizon
    ).strip()

    source = (
        str(source)
        if source is not None
        else None
    )

    now = datetime.now(
        timezone.utc
    )

    rows_to_upsert = []

    for _, row in df.iterrows():

        dt = pd.to_datetime(
            row["Date"],
            errors="coerce"
        )

        if pd.isna(dt):
            continue

        generation = pd.to_numeric(
            row[generation_col],
            errors="coerce"
        )

        if pd.isna(generation):
            continue

        rainfall = None

        if (
            rain_col
            and pd.notna(
                row[rain_col]
            )
        ):

            rainfall_value = pd.to_numeric(
                row[rain_col],
                errors="coerce"
            )

            if pd.notna(
                rainfall_value
            ):

                rainfall = float(
                    rainfall_value
                )

        rows_to_upsert.append(
            {
                "site_code": site,

                "forecast_date":
                    dt.date(),

                "horizon":
                    horizon,

                "predicted_generation_kwh":
                    float(generation),

                "forecast_rainfall_mm":
                    rainfall,

                "forecast_source":
                    source,

                "created_at":
                    now,
            }
        )

    if not rows_to_upsert:
        return False

    # --------------------------------------------------------
    # Remove duplicate dates
    # --------------------------------------------------------

    unique_rows = {}

    for record in rows_to_upsert:

        key = (
            record["site_code"],
            record["forecast_date"],
            record["horizon"],
        )

        unique_rows[key] = record

    rows_to_upsert = list(
        unique_rows.values()
    )

    # --------------------------------------------------------
    # PostgreSQL UPSERT
    # --------------------------------------------------------

    stmt = insert(
        GenerationForecast
    ).values(
        rows_to_upsert
    )

    stmt = stmt.on_conflict_do_update(
        constraint=
        "uq_generation_site_date_horizon",

        set_={
            "predicted_generation_kwh":
                stmt.excluded
                .predicted_generation_kwh,

            "forecast_rainfall_mm":
                stmt.excluded
                .forecast_rainfall_mm,

            "forecast_source":
                stmt.excluded
                .forecast_source,

            "created_at":
                stmt.excluded
                .created_at,
        },
    )

    with Session(engine) as session:

        try:

            session.execute(
                stmt
            )

            session.commit()

            return True

        except Exception:

            session.rollback()

            raise


# ============================================================
# LOAD LATEST GENERATION FORECAST
# ============================================================

def load_latest_generation_forecast(
    site,
    horizon
):

    if engine is None:
        return pd.DataFrame()

    site = str(
        site
    ).upper()

    with Session(engine) as session:

        rows = session.scalars(

            select(
                GenerationForecast
            )

            .where(

                GenerationForecast.site_code
                == site,

                GenerationForecast.horizon
                == horizon,

            )

            .order_by(

                GenerationForecast.created_at.desc(),

                GenerationForecast.forecast_date.asc(),

            )

        ).all()

    if not rows:
        return pd.DataFrame()

    latest_created = max(
        r.created_at
        for r in rows
    )

    rows = [
        r
        for r in rows
        if r.created_at
        == latest_created
    ]

    return pd.DataFrame(

        [

            {
                "Site":
                    r.site_code,

                "Date":
                    r.forecast_date,

                "Forecast_Generation_kWh":
                    r.predicted_generation_kwh,

                "Forecast_Rainfall_mm":
                    r.forecast_rainfall_mm,

                "Forecast_Source":
                    r.forecast_source,

                "Created_At":
                    r.created_at,
            }

            for r in rows

        ]
    )


# ============================================================
# DATABASE STATUS
# ============================================================

def database_status():

    if engine is None:
        return "Not configured"

    try:

        with engine.connect() as connection:

            connection.execute(
                select(
                    Plant
                ).limit(1)
            )

        return "Connected"

    except Exception as e:

        print(
            f"Database connection failed: {e}"
        )

        return "Connection failed"
