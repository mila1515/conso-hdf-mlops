"""Ingestion des données : consommation éCO2mix (RTE / ODRÉ) et météo (Open-Meteo).

- Historique consolidé/définitif : dataset `eco2mix-regional-cons-def` (depuis 2013, pas 30 min)
- Données récentes temps réel  : dataset `eco2mix-regional-tr` (pas 15 min, ~2-3 derniers mois)

Les deux sources sont fusionnées (priorité au consolidé), puis agrégées au pas horaire (MW moyens).
L'ingestion est incrémentale : on ne re-télécharge que ce qui manque.
"""

from __future__ import annotations

import io
import json
import logging
from datetime import date, timedelta

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from conso_hdf.config import Settings, settings

log = logging.getLogger(__name__)
TIMEOUT = 300


def _session() -> requests.Session:
    """Session HTTP avec relances automatiques (erreurs 429/5xx, coupures réseau)."""
    retry = Retry(total=5, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET",), respect_retry_after_header=False)
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "conso-hdf-mlops/0.1 (projet portfolio)"
    return s


HTTP = _session()


# --------------------------------------------------------------------------- #
# éCO2mix
# --------------------------------------------------------------------------- #
def _fetch_odre(cfg: Settings, dataset: str, start: str | None = None) -> pd.DataFrame:
    """Télécharge (export CSV) la consommation d'une région depuis ODRÉ."""
    where = f'libelle_region="{cfg.region}" and consommation is not null'
    if start:
        where += f" and date_heure >= date'{start}'"
    url = f"{cfg.odre_base_url}/{dataset}/exports/csv"
    params = {"select": "date_heure,consommation", "where": where, "order_by": "date_heure"}
    log.info("Téléchargement %s (depuis %s)...", dataset, start or "le début")
    r = HTTP.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.content.decode("utf-8-sig")), sep=";")
    if df.empty:
        return pd.DataFrame(columns=["date_heure", "consommation"])
    df["date_heure"] = pd.to_datetime(df["date_heure"], utc=True)
    df["consommation"] = pd.to_numeric(df["consommation"], errors="coerce")
    df["source"] = dataset
    log.info("  -> %d lignes (%s → %s)", len(df), df.date_heure.min(), df.date_heure.max())
    return df.dropna(subset=["consommation"])


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Agrège au pas horaire (moyenne des MW) avec un index UTC continu."""
    s = (
        df.sort_values("date_heure")
        .drop_duplicates("date_heure", keep="first")
        .set_index("date_heure")["consommation"]
        .resample("1h")
        .mean()
    )
    # On comble uniquement les petits trous (<= 3h) par interpolation
    s = s.interpolate(limit=3, limit_area="inside")
    out = s.to_frame("consommation")
    out.index.name = "date_heure"
    return out.dropna()


def ingest_consumption(cfg: Settings = settings) -> pd.DataFrame:
    """Met à jour le fichier local de consommation horaire et le renvoie."""
    cfg.ensure_dirs()
    existing = pd.read_parquet(cfg.conso_path) if cfg.conso_path.exists() else None
    state_path = cfg.conso_path.with_name("ingestion_state.json")
    state = json.loads(state_path.read_text()) if (existing is not None and state_path.exists()) else {}
    start_def = state.get("last_consolidated", cfg.history_start)
    consolidated = _fetch_odre(cfg, cfg.dataset_consolidated, start_def)
    realtime = _fetch_odre(cfg, cfg.dataset_realtime)

    last_def = (
        consolidated.date_heure.max().strftime("%Y-%m-%d")
        if not consolidated.empty
        else start_def
    )
    # Priorité au consolidé : on ne garde le temps réel qu'après la fin du consolidé
    if not consolidated.empty:
        realtime = realtime[realtime.date_heure > consolidated.date_heure.max()]
    fresh = to_hourly(pd.concat([consolidated, realtime], ignore_index=True))

    if existing is not None:
        # Les nouvelles valeurs remplacent les anciennes (ex. temps réel -> consolidé)
        merged = pd.concat([existing[~existing.index.isin(fresh.index)], fresh]).sort_index()
    else:
        merged = fresh
    merged = merged[merged.index >= pd.Timestamp(cfg.history_start, tz="UTC")]
    merged.to_parquet(cfg.conso_path)
    state_path.write_text(json.dumps({"last_consolidated": last_def}))
    log.info("Consommation : %d heures (%s → %s)", len(merged), merged.index.min(), merged.index.max())
    return merged


# --------------------------------------------------------------------------- #
# Météo
# --------------------------------------------------------------------------- #
def _open_meteo(url: str, params: dict) -> pd.DataFrame:
    r = HTTP.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame({"date_heure": pd.to_datetime(h["time"], utc=True), "temperature": h["temperature_2m"]})
    return df.dropna().set_index("date_heure")


def ingest_weather(cfg: Settings = settings) -> pd.DataFrame:
    """Températures horaires à Lille : historique (archive) + 14 jours récents + prévision J+2."""
    cfg.ensure_dirs()
    existing = pd.read_parquet(cfg.weather_path) if cfg.weather_path.exists() else None
    base = {"latitude": cfg.latitude, "longitude": cfg.longitude, "hourly": "temperature_2m", "timezone": "UTC"}

    frames = []
    archive_end = date.today() - timedelta(days=6)  # l'archive a ~5 jours de retard
    if existing is not None:
        archive_start = existing.index.max().date() - timedelta(days=7)
    else:
        archive_start = date.fromisoformat(cfg.history_start)
    if archive_start < archive_end:
        log.info("Météo archive %s → %s", archive_start, archive_end)
        frames.append(
            _open_meteo(
                "https://archive-api.open-meteo.com/v1/archive",
                {**base, "start_date": archive_start.isoformat(), "end_date": archive_end.isoformat()},
            )
        )
    log.info("Météo récente + prévision")
    try:
        frames.append(
            _open_meteo("https://api.open-meteo.com/v1/forecast", {**base, "past_days": 14, "forecast_days": 3})
        )
    except requests.RequestException as e:
        # Non bloquant : XGBoost sait gérer les températures manquantes (NaN)
        log.warning("Prévision météo indisponible (%s) : on continue sans.", e)

    if not frames:
        log.warning("Aucune nouvelle donnée météo récupérée.")
        return existing if existing is not None else pd.DataFrame(columns=["temperature"])
    fresh = pd.concat(frames)
    fresh = fresh[~fresh.index.duplicated(keep="last")]
    merged = fresh if existing is None else pd.concat([existing[~existing.index.isin(fresh.index)], fresh])
    merged = merged.sort_index()
    merged.to_parquet(cfg.weather_path)
    log.info("Météo : %d heures (%s → %s)", len(merged), merged.index.min(), merged.index.max())
    return merged


def load_data(cfg: Settings = settings) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Charge les données locales (sans appel réseau)."""
    if not cfg.conso_path.exists():
        raise FileNotFoundError("Pas de données : lancez d'abord `python -m conso_hdf.pipeline ingest`.")
    conso = pd.read_parquet(cfg.conso_path)
    weather = pd.read_parquet(cfg.weather_path) if (cfg.use_weather and cfg.weather_path.exists()) else None
    return conso, weather


def data_freshness(conso: pd.DataFrame, cfg: Settings = settings) -> dict:
    """Âge de la dernière consommation observée.

    La prévision part de la dernière donnée ingérée : si l'ingestion est bloquée depuis
    plusieurs jours, la « prévision des 24 prochaines heures » porterait sur des heures
    déjà passées. Ce contrôle permet de le détecter (API, monitoring, pipeline).
    """
    last = conso.index.max()
    age_h = (pd.Timestamp.now(tz="UTC") - last).total_seconds() / 3600
    return {
        "last_observation": last.isoformat(),
        "data_age_hours": round(age_h, 1),
        "max_data_age_hours": cfg.max_data_age_hours,
        "stale": bool(age_h > cfg.max_data_age_hours),
    }


def ingest_all(cfg: Settings = settings) -> None:
    ingest_consumption(cfg)
    if cfg.use_weather:
        ingest_weather(cfg)
