"""Configuration centralisée, lue depuis le fichier .env (ou les variables d'environnement)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))


def _path(env_name: str, default: str) -> Path:
    p = Path(os.getenv(env_name, default))
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass(frozen=True)
class Settings:
    # --- Données ---
    region: str = os.getenv("REGION", "Hauts-de-France")
    history_start: str = os.getenv("HISTORY_START", "2019-01-01")
    odre_base_url: str = os.getenv(
        "ODRE_BASE_URL", "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets"
    )
    dataset_consolidated: str = os.getenv("DATASET_CONSOLIDATED", "eco2mix-regional-cons-def")
    dataset_realtime: str = os.getenv("DATASET_REALTIME", "eco2mix-regional-tr")

    # --- Météo (Open-Meteo, gratuit, sans clé) ---
    use_weather: bool = os.getenv("USE_WEATHER", "true").lower() == "true"
    latitude: float = float(os.getenv("LATITUDE", "50.63"))  # Lille
    longitude: float = float(os.getenv("LONGITUDE", "3.06"))

    # --- Modèle ---
    forecast_horizon: int = int(os.getenv("FORECAST_HORIZON", "24"))
    test_days: int = int(os.getenv("TEST_DAYS", "30"))
    model_name: str = os.getenv("MODEL_NAME", "conso-hdf-xgboost")

    # --- MLflow ---
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow_experiment: str = os.getenv("MLFLOW_EXPERIMENT", "conso-hdf")

    # --- Monitoring ---
    mape_alert_threshold: float = float(os.getenv("MAPE_ALERT_THRESHOLD", "5.0"))
    drift_share_threshold: float = float(os.getenv("DRIFT_SHARE_THRESHOLD", "0.5"))
    wasserstein_threshold: float = float(os.getenv("WASSERSTEIN_THRESHOLD", "0.5"))
    monitoring_days: int = int(os.getenv("MONITORING_DAYS", "7"))
    # Notifications (voir conso_hdf/notify.py) : laisser vide pour désactiver
    discord_webhook_url: str = os.getenv("DISCORD_WEBHOOK_URL", "")
    slack_webhook_url: str = os.getenv("SLACK_WEBHOOK_URL", "")
    alert_webhook_url: str = os.getenv("ALERT_WEBHOOK_URL", "")  # ancien réglage (Discord/Slack auto-détecté)
    # Résumé après chaque exécution daily/weekly (MAPE, dérive, pic prévu...), en plus des alertes
    notify_daily_summary: bool = os.getenv("NOTIFY_DAILY_SUMMARY", "false").lower() == "true"
    # Âge maximal (h) de la dernière consommation ingérée avant de considérer les données obsolètes
    max_data_age_hours: float = float(os.getenv("MAX_DATA_AGE_HOURS", "36"))
    # Nombre de rapports Evidently HTML conservés dans reports/
    reports_keep: int = int(os.getenv("REPORTS_KEEP", "30"))

    # --- Planificateur (docker compose) ---
    schedule_hour_utc: int = int(os.getenv("SCHEDULE_HOUR_UTC", "6"))

    # --- Chemins ---
    data_dir: Path = field(default_factory=lambda: _path("DATA_DIR", "data"))
    artifacts_dir: Path = field(default_factory=lambda: _path("ARTIFACTS_DIR", "artifacts"))
    reports_dir: Path = field(default_factory=lambda: _path("REPORTS_DIR", "reports"))
    # Artefacts MLflow quand le suivi est local (sqlite) : emplacement explicite et donc
    # identique en local, en CI et dans le cache GitHub Actions.
    mlflow_artifacts_dir: Path = field(default_factory=lambda: _path("MLFLOW_ARTIFACTS_DIR", "mlartifacts"))

    @property
    def conso_path(self) -> Path:
        return self.data_dir / "raw" / "eco2mix_hdf.parquet"

    @property
    def weather_path(self) -> Path:
        return self.data_dir / "raw" / "weather_lille.parquet"

    @property
    def predictions_path(self) -> Path:
        return self.data_dir / "predictions" / "predictions.parquet"

    @property
    def champion_dir(self) -> Path:
        return self.artifacts_dir / "champion"

    def ensure_dirs(self) -> None:
        for p in (
            self.conso_path.parent,
            self.weather_path.parent,
            self.predictions_path.parent,
            self.champion_dir,
            self.reports_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()
