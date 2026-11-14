"""Point d'entrée en ligne de commande.

Exemples :
    python -m conso_hdf.pipeline ingest     # récupère les nouvelles données
    python -m conso_hdf.pipeline train      # entraîne + compare au champion
    python -m conso_hdf.pipeline predict    # prévision des 24 prochaines heures
    python -m conso_hdf.pipeline backfill   # simule les prévisions des 7 derniers jours (1er lancement)
    python -m conso_hdf.pipeline monitor    # fraîcheur + performance réelle + dérive
    python -m conso_hdf.pipeline daily      # ingest + monitor + predict (+ retrain si alerte)
    python -m conso_hdf.pipeline weekly     # ingest + train (+ predict)
    python -m conso_hdf.pipeline schedule   # boucle : daily chaque jour, weekly le lundi (docker compose)

Toute étape qui échoue (API source indisponible, format inattendu...) déclenche une alerte
webhook avant de remonter l'erreur : la panne ne passe pas inaperçue.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from conso_hdf.config import settings
from conso_hdf.ingestion import data_freshness, ingest_all, load_data
from conso_hdf.model_store import champion_exists
from conso_hdf.monitoring import run_monitoring, send_alert
from conso_hdf.predict import backfill, predict_and_store
from conso_hdf.train import train

log = logging.getLogger("conso_hdf.pipeline")


def ingest_checked() -> None:
    """Ingestion + contrôle de fraîcheur. Lève une erreur si les données restent obsolètes."""
    ingest_all(settings)
    conso, _ = load_data(settings)
    fresh = data_freshness(conso, settings)
    if fresh["stale"]:
        raise RuntimeError(
            f"Ingestion terminée mais données obsolètes : dernière observation {fresh['last_observation']} "
            f"({fresh['data_age_hours']} h). La source RTE est peut-être en retard ou en panne."
        )


def daily(auto_retrain: bool = True) -> None:
    ingest_checked()
    if not champion_exists(settings):
        train(settings)
        backfill(settings, days=settings.monitoring_days)
    summary = run_monitoring(settings)  # évalue les prévisions d'hier sur les données fraîches
    if auto_retrain and summary["retrain_recommended"]:
        log.info("Alerte -> réentraînement automatique")
        train(settings)
    predict_and_store(settings)


def weekly() -> None:
    ingest_checked()
    train(settings)
    predict_and_store(settings)


def _next_run(now: datetime, hour: int) -> datetime:
    run = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    return run if run > now else run + timedelta(days=1)


def schedule() -> None:
    """Planificateur minimal pour docker compose : `weekly` le lundi, `daily` les autres jours,
    à `SCHEDULE_HOUR_UTC`. Une exécution en échec est alertée puis la boucle continue."""
    while True:
        now = datetime.now(timezone.utc)
        nxt = _next_run(now, settings.schedule_hour_utc)
        log.info("Prochaine exécution : %s", nxt.isoformat())
        time.sleep((nxt - now).total_seconds())
        step = "weekly" if nxt.weekday() == 0 else "daily"
        try:
            run_step(step)
        except Exception:  # noqa: BLE001 - déjà alerté dans run_step ; on attend le prochain créneau
            log.exception("Échec de l'exécution planifiée (%s)", step)


def run_step(step: str, force_promote: bool = False, auto_retrain: bool = True) -> None:
    try:
        if step == "ingest":
            ingest_checked()
        elif step == "train":
            print(json.dumps(train(settings, force_promote=force_promote), indent=2))
        elif step == "predict":
            print(predict_and_store(settings).to_string(index=False))
        elif step == "backfill":
            backfill(settings, days=settings.monitoring_days)
        elif step == "monitor":
            print(json.dumps(run_monitoring(settings), indent=2, ensure_ascii=False, default=str))
        elif step == "daily":
            daily(auto_retrain=auto_retrain)
        elif step == "weekly":
            weekly()
    except Exception as e:
        send_alert(settings, f"[conso-hdf][critique] Échec de l'étape « {step} » : {type(e).__name__}: {e}")
        raise


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    parser = argparse.ArgumentParser(description="Pipeline MLOps – conso électrique Hauts-de-France")
    parser.add_argument(
        "step", choices=["ingest", "train", "predict", "backfill", "monitor", "daily", "weekly", "schedule"]
    )
    parser.add_argument("--force-promote", action="store_true", help="Promouvoir même si pas meilleur")
    parser.add_argument("--no-auto-retrain", action="store_true", help="(daily) ne pas réentraîner en cas d'alerte")
    args = parser.parse_args()

    if args.step == "schedule":
        schedule()
    else:
        run_step(args.step, force_promote=args.force_promote, auto_retrain=not args.no_auto_retrain)


if __name__ == "__main__":
    main()
