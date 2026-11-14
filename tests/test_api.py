import pandas as pd
from fastapi.testclient import TestClient

import conso_hdf.api.main as api
from conso_hdf.train import train
from tests.conftest import make_synthetic_conso


def test_api_endpoints(cfg, monkeypatch):
    monkeypatch.setattr(api, "settings", cfg)
    client = TestClient(api.app)
    health = client.get("/health").json()
    assert health["model_loaded"] is False and health["status"] == "degraded"
    assert client.get("/forecast").status_code == 503

    train(cfg)
    health = client.get("/health").json()
    assert health["model_loaded"] is True and health["status"] == "ok"
    assert health["data"]["stale"] is False
    assert client.get("/model").json()["model_version"] == 1

    r = client.get("/forecast", params={"hours": 12})
    assert r.status_code == 200
    body = r.json()
    assert len(body["forecast"]) == 12 and body["unit"] == "MW"
    assert body["data_age_hours"] < cfg.max_data_age_hours

    assert client.get("/forecast", params={"hours": 48}).status_code == 422


def test_api_refuses_forecast_on_stale_data(cfg, monkeypatch):
    monkeypatch.setattr(api, "settings", cfg)
    train(cfg)
    # L'ingestion « s'est arrêtée » il y a 5 jours
    stale = make_synthetic_conso(end=make_synthetic_conso().index.max() - pd.Timedelta(days=5))
    stale.to_parquet(cfg.conso_path)
    client = TestClient(api.app)
    assert client.get("/health").json()["status"] == "degraded"
    r = client.get("/forecast")
    assert r.status_code == 503 and "obsolètes" in r.json()["detail"]
