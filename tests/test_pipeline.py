import pandas as pd
import pytest
import xgboost as xgb

import conso_hdf.pipeline as pipeline
from conso_hdf.features import TARGET, training_frame
from conso_hdf.model_store import load_champion, save_champion
from conso_hdf.monitoring import run_monitoring
from conso_hdf.predict import StaleDataError, backfill, forecast, predict_and_store
from conso_hdf.train import train
from tests.conftest import make_synthetic_conso


def test_train_predict_monitor_end_to_end(cfg):
    result = train(cfg)
    assert result["promoted"] is True
    # Le modèle doit battre la baseline naïve sur ces données très régulières
    assert result["metrics"]["mape"] < result["baseline"]["mape"]

    _, meta = load_champion(cfg)
    assert meta["model_version"] == 1

    preds = predict_and_store(cfg)
    assert len(preds) == cfg.forecast_horizon
    assert preds.prediction_mw.between(3000, 9000).all()
    assert not preds.simulated.any()

    summary = run_monitoring(cfg)
    assert summary["model_version"] == 1
    assert summary["freshness"]["stale"] is False
    assert "drift" in summary


def test_production_model_is_refit_on_all_data(cfg):
    train(cfg)
    _, meta = load_champion(cfg)
    conso = pd.read_parquet(cfg.conso_path)
    # Le modèle de production a vu les données jusqu'à la fin, au-delà du jeu de test
    assert meta["refit_on_full_data"] is True
    assert pd.Timestamp(meta["data_end"]) == conso.index.max()
    assert pd.Timestamp(meta["train_end"]) < pd.Timestamp(meta["test_start"])
    # Le modèle d'évaluation (hors échantillon) est conservé séparément
    assert (cfg.champion_dir / "model_eval.json").exists()
    prod, _ = load_champion(cfg)
    ev, _ = load_champion(cfg, kind="eval")
    assert prod.get_booster().num_boosted_rounds() == meta["refit_n_estimators"]
    assert ev.get_booster().num_boosted_rounds() >= meta["refit_n_estimators"]


def test_challenger_not_promoted_if_worse(cfg):
    first = train(cfg)
    second = train(cfg)  # mêmes données -> même score, donc pas strictement meilleur
    assert second["champion_mape"] is not None
    assert second["champion_mape"] == pytest.approx(first["metrics"]["mape"])
    assert second["promoted"] is False
    assert second["version"] is None


def test_champion_still_compared_when_features_change(cfg):
    train(cfg)
    _, meta = load_champion(cfg)
    # Simule un champion entraîné avec une liste de variables différente (une variable en moins)
    old_features = [f for f in meta["features"] if f != "roll_std_24h"]
    df = training_frame(pd.read_parquet(cfg.conso_path), None)
    old_model = xgb.XGBRegressor(n_estimators=50).fit(df[old_features], df[TARGET])
    save_champion(old_model, {**meta, "features": old_features}, cfg, eval_model=old_model)

    result = train(cfg)
    # Avant : champion_mape = None -> promotion automatique sans comparaison
    assert result["champion_mape"] is not None


def test_backfill_uses_out_of_sample_eval_model(cfg):
    train(cfg)
    out = backfill(cfg, days=3)
    assert out.simulated.all()
    summary = run_monitoring(cfg)
    assert summary["performance"]["status"] == "ok"
    assert summary["performance"]["includes_simulated"] is True


def test_forecast_refused_on_stale_data(cfg):
    train(cfg)
    end = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(days=4)
    make_synthetic_conso(end=end).to_parquet(cfg.conso_path)
    with pytest.raises(StaleDataError):
        forecast(cfg)
    assert len(forecast(cfg, allow_stale=True)) == cfg.forecast_horizon
    summary = run_monitoring(cfg)
    assert any(a["type"] == "data" for a in summary["alerts"])
    assert summary["retrain_recommended"] is False


def test_pipeline_failure_sends_alert(cfg, monkeypatch):
    sent = []
    monkeypatch.setattr(pipeline, "settings", cfg)
    monkeypatch.setattr(pipeline, "send_alert", lambda _cfg, msg: sent.append(msg))

    def boom(_cfg):
        raise ConnectionError("API ODRÉ indisponible")

    monkeypatch.setattr(pipeline, "ingest_all", boom)
    with pytest.raises(ConnectionError):
        pipeline.run_step("daily")
    assert len(sent) == 1 and "daily" in sent[0] and "ODRÉ" in sent[0]


def test_ingestion_with_stale_source_fails(cfg, monkeypatch):
    monkeypatch.setattr(pipeline, "settings", cfg)
    monkeypatch.setattr(pipeline, "ingest_all", lambda _cfg: None)
    make_synthetic_conso(end=pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=3)).to_parquet(cfg.conso_path)
    with pytest.raises(RuntimeError, match="obsolètes"):
        pipeline.ingest_checked()

