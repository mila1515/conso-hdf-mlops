"""Fixtures : données synthétiques + configuration isolée dans un dossier temporaire."""

import numpy as np
import pandas as pd
import pytest

from conso_hdf.config import Settings


def make_synthetic_conso(days: int = 90, seed: int = 0, end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Consommation horaire synthétique. Par défaut, la série se termine à l'heure courante
    (données « fraîches ») pour que le contrôle de fraîcheur ne bloque pas les tests."""
    rng = np.random.default_rng(seed)
    end = end if end is not None else pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=1)
    idx = pd.date_range(end=end, periods=days * 24, freq="1h", name="date_heure")
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    y = 6000 + 800 * np.sin(2 * np.pi * (hour - 6) / 24) - 400 * (dow >= 5) + rng.normal(0, 60, len(idx))
    return pd.DataFrame({"consommation": y}, index=idx)


@pytest.fixture
def cfg(tmp_path) -> Settings:
    c = Settings(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "reports",
        mlflow_artifacts_dir=tmp_path / "mlartifacts",
        mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        mlflow_experiment="test",
        model_name="test-model",
        use_weather=False,
        test_days=7,
        monitoring_days=3,
        history_start="2020-01-01",
        # jamais de vraie notification pendant les tests, même si .env définit des webhooks
        alert_webhook_url="",
        discord_webhook_url="",
        slack_webhook_url="",
        notify_daily_summary=False,
    )
    c.ensure_dirs()
    make_synthetic_conso().to_parquet(c.conso_path)
    return c
