"""Prévision des prochaines heures avec le modèle champion."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from conso_hdf.config import Settings, settings
from conso_hdf.features import build_features
from conso_hdf.ingestion import data_freshness, load_data
from conso_hdf.model_store import load_champion

log = logging.getLogger(__name__)


class StaleDataError(RuntimeError):
    """Les données ingérées sont trop anciennes pour produire une prévision utile."""


def forecast(cfg: Settings = settings, hours: int | None = None, allow_stale: bool = False) -> pd.DataFrame:
    """Calcule la prévision des `hours` heures qui suivent la dernière donnée (sans l'enregistrer).

    Lève `StaleDataError` si la dernière consommation observée a plus de `MAX_DATA_AGE_HOURS` :
    la prévision porterait alors sur des heures déjà passées.
    """
    hours = hours or cfg.forecast_horizon
    if not 1 <= hours <= 24:
        raise ValueError("L'horizon doit être compris entre 1 et 24 h (lags >= 24 h).")
    model, meta = load_champion(cfg)
    conso, weather = load_data(cfg)
    fresh = data_freshness(conso, cfg)
    if fresh["stale"] and not allow_stale:
        raise StaleDataError(
            f"Données obsolètes : dernière observation {fresh['last_observation']} "
            f"({fresh['data_age_hours']} h > {cfg.max_data_age_hours} h). Vérifiez l'ingestion."
        )
    if not meta["use_weather"]:
        weather = None
    feats = build_features(conso, weather, future_hours=hours).tail(hours)
    pred = model.predict(feats[meta["features"]])
    out = pd.DataFrame(
        {
            "target_time": feats.index,
            "prediction_mw": pred.astype(float).round(1),
            "model_version": meta["model_version"],
            "generated_at": datetime.now(timezone.utc),
            "simulated": False,
        }
    )
    out.attrs["freshness"] = fresh
    return out


def predict_and_store(cfg: Settings = settings) -> pd.DataFrame:
    """Calcule la prévision et l'ajoute à l'historique des prévisions (pour le monitoring)."""
    cfg.ensure_dirs()
    new = forecast(cfg)
    if cfg.predictions_path.exists():
        old = pd.read_parquet(cfg.predictions_path)
        # Une seule prévision par heure cible : on garde la plus récente
        new_all = pd.concat([old, new]).drop_duplicates("target_time", keep="last")
    else:
        new_all = new
    new_all.sort_values("target_time").to_parquet(cfg.predictions_path, index=False)
    log.info("Prévision enregistrée : %s → %s", new.target_time.min(), new.target_time.max())
    return new


def backfill(cfg: Settings = settings, days: int = 7) -> pd.DataFrame:
    """Simule les prévisions des `days` derniers jours (utile au 1er lancement pour le monitoring).

    Comme toutes les variables de retard ont un décalage >= 24 h, la prévision d'une heure t
    n'utilise que des consommations connues à t-24 h : c'est équivalent à une prévision émise
    la veille. Seule différence : la température utilisée est l'observée (et non la prévue),
    ce qui rend ces prévisions simulées légèrement optimistes.

    Le modèle de production est réentraîné sur toutes les données (y compris ces derniers jours) :
    il les a donc déjà vues. On utilise ici le **modèle d'évaluation** du champion, entraîné avant
    le jeu de test, pour que les prévisions simulées restent hors échantillon.
    Elles sont marquées `simulated=True` et le monitoring les distingue des vraies prévisions.
    """
    cfg.ensure_dirs()
    model, meta = load_champion(cfg, kind="eval")
    conso, weather = load_data(cfg)
    feats = build_features(conso, weather if meta["use_weather"] else None)
    feats = feats[feats.index > conso.index.max() - pd.Timedelta(days=days)]
    out = pd.DataFrame(
        {
            "target_time": feats.index,
            "prediction_mw": model.predict(feats[meta["features"]]).astype(float).round(1),
            "model_version": meta["model_version"],
            "generated_at": feats.index - pd.Timedelta(hours=24),
            "simulated": True,
        }
    )
    if cfg.predictions_path.exists():
        old = pd.read_parquet(cfg.predictions_path)
        out = pd.concat([out, old]).drop_duplicates("target_time", keep="last")
    out.sort_values("target_time").to_parquet(cfg.predictions_path, index=False)
    log.info("Backfill : %d prévisions simulées", len(feats))
    return out
