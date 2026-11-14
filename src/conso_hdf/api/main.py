"""API REST (FastAPI) qui sert les prévisions du modèle champion.

Lancement : uvicorn conso_hdf.api.main:app --reload
Documentation interactive : http://localhost:8000/docs
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from conso_hdf import __version__
from conso_hdf.config import settings
from conso_hdf.ingestion import data_freshness
from conso_hdf.model_store import champion_exists, load_champion
from conso_hdf.predict import StaleDataError, forecast

app = FastAPI(
    title="API Prévision conso électrique – Hauts-de-France",
    description="Prévision horaire de la consommation électrique régionale (données RTE éCO2mix).",
    version=__version__,
)


class ForecastPoint(BaseModel):
    target_time: datetime
    prediction_mw: float


class ForecastResponse(BaseModel):
    region: str
    model_version: int
    generated_at: datetime
    last_observation: datetime
    data_age_hours: float
    unit: str = "MW"
    forecast: list[ForecastPoint]


@app.get("/health", tags=["monitoring"])
def health() -> dict:
    """État du service : modèle chargé et fraîcheur des données (`degraded` si données obsolètes)."""
    model_loaded = champion_exists(settings)
    freshness = None
    if settings.conso_path.exists():
        freshness = data_freshness(pd.read_parquet(settings.conso_path), settings)
    ok = model_loaded and freshness is not None and not freshness["stale"]
    return {"status": "ok" if ok else "degraded", "model_loaded": model_loaded, "data": freshness}


@app.get("/model", tags=["modèle"])
def model_info() -> dict:
    """Métadonnées du modèle champion (version, métriques, période d'entraînement...)."""
    try:
        _, meta = load_champion(settings)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return meta


@app.get("/forecast", response_model=ForecastResponse, tags=["prévision"])
def get_forecast(hours: int = Query(24, ge=1, le=24, description="Horizon de prévision (h)")) -> ForecastResponse:
    """Prévision des prochaines heures, calculée à partir des dernières données ingérées."""
    try:
        df = forecast(settings, hours=hours)
    except (FileNotFoundError, StaleDataError) as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    fresh = df.attrs["freshness"]
    return ForecastResponse(
        region=settings.region,
        model_version=int(df.model_version.iloc[0]),
        generated_at=df.generated_at.iloc[0],
        last_observation=fresh["last_observation"],
        data_age_hours=fresh["data_age_hours"],
        forecast=[
            ForecastPoint(target_time=t, prediction_mw=p)
            for t, p in zip(df.target_time, df.prediction_mw, strict=True)
        ],
    )


@app.get("/history", tags=["prévision"])
def history(days: int = Query(7, ge=1, le=90)) -> list[dict]:
    """Prévisions passées comparées à la consommation réelle."""
    if not settings.predictions_path.exists() or not settings.conso_path.exists():
        return []
    preds = pd.read_parquet(settings.predictions_path).set_index("target_time")
    conso = pd.read_parquet(settings.conso_path)
    df = preds.join(conso, how="left")
    df = df[df.index > df.index.max() - pd.Timedelta(days=days)].reset_index()
    df = df.rename(columns={"index": "target_time", "consommation": "actual_mw"})
    if "simulated" not in df.columns:
        df["simulated"] = False
    df = df.astype(object).where(df.notna(), None)
    cols = ["target_time", "prediction_mw", "actual_mw", "model_version", "simulated"]
    return df[cols].to_dict(orient="records")


@app.get("/monitoring", tags=["monitoring"])
def monitoring() -> dict:
    """Dernier rapport de monitoring (performance + dérive)."""
    path = settings.reports_dir / "monitoring_latest.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Aucun monitoring exécuté pour l'instant.")
    return json.loads(path.read_text())
