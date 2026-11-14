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
    python -m conso_hdf.pipeline notify-test  # envoie un message de test sur Discord / Slack

Toute étape qui échoue (API source indisponible, format inattendu...) déclenche une alerte
Discord / Slack avant de remonter l'erreur : la panne ne passe pas inaperçue.
Avec `NOTIFY_DAILY_SUMMARY=true`, un résumé est aussi envoyé après chaque exécution daily / weekly.
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
from conso_hdf.monitoring import run_monitoring
from conso_hdf.notify import notify, run_url, send_alert, webhook_targets
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


def _forecast_fields(preds) -> dict:
    peak = preds.loc[preds.prediction_mw.idxmax()]
    return {
        "Pic prévu": f"{peak.prediction_mw:,.0f} MW".replace(",", " "),
        "Heure du pic": f"{peak.target_time.tz_convert('Europe/Paris'):%d/%m %Hh} (Paris)",
        "Moyenne prévue": f"{preds.prediction_mw.mean():,.0f} MW".replace(",", " "),
    }


def _train_fields(result: dict) -> dict:
    fields = {
        "MAPE test": f"{result['metrics']['mape']:.2f} %",
        "Baseline": f"{result['baseline']['mape']:.2f} %",
        "Promu": "oui" if result["promoted"] else "non",
    }
    if result.get("champion_mape") is not None:
        fields["Champion"] = f"{result['champion_mape']:.2f} %"
    if result.get("version"):
        fields["Nouvelle version"] = f"v{result['version']}"
    return fields


def daily(auto_retrain: bool = True) -> None:
    ingest_checked()
    if not champion_exists(settings):
        train(settings)
        backfill(settings, days=settings.monitoring_days)
    summary = run_monitoring(settings)  # évalue les prévisions d'hier sur les données fraîches
    retrained = None
    if auto_retrain and summary["retrain_recommended"]:
        log.info("Alerte -> réentraînement automatique")
        retrained = train(settings)
    preds = predict_and_store(settings)

    if settings.notify_daily_summary:
        perf, drift = summary["performance"], summary["drift"]
        fields = {"Modèle": f"v{summary['model_version']}"}
        if perf.get("status") == "ok":
            label = "MAPE simulé" if perf.get("includes_simulated") else "MAPE réel"
            fields[f"{label} {settings.monitoring_days} j"] = f"{perf['mape']:.2f} %"
        if drift.get("status") == "ok":
            fields["Dérive"] = f"{drift['drift_share']:.0%} des variables"
        fields.update(_forecast_fields(preds))
        if retrained:
            fields["Réentraînement"] = "promu" if retrained["promoted"] else "non promu"
        severity = "avertissement" if summary["alerts"] else "succès"
        msg = f"{len(summary['alerts'])} alerte(s)" if summary["alerts"] else "Aucune alerte"
        notify(settings, "conso-hdf : exécution quotidienne", f"{msg}. Prévision des 24 prochaines heures publiée.",
               severity, fields)


def weekly() -> None:
    ingest_checked()
    result = train(settings)
    preds = predict_and_store(settings)
    if settings.notify_daily_summary:
        verdict = (
            "Nouveau modèle promu en production."
            if result["promoted"]
            else "Le champion actuel reste en production (challenger pas meilleur)."
        )
        notify(settings, "conso-hdf : réentraînement hebdomadaire", verdict, "succès",
               {**_train_fields(result), **_forecast_fields(preds)})


def notify_test() -> None:
    targets = webhook_targets(settings)
    if not targets:
        raise SystemExit("Aucun webhook configuré : définissez DISCORD_WEBHOOK_URL et/ou SLACK_WEBHOOK_URL.")
    sent = notify(settings, "conso-hdf : test de notification",
                  "Les alertes du pipeline arriveront ici.", "info", {"Cibles": ", ".join(k for k, _ in targets)})
    print(f"{sent}/{len(targets)} notification(s) envoyée(s)")
    if sent < len(targets):
        raise SystemExit(1)


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
        fields = {"Étape": step, "Erreur": type(e).__name__}
        if run_url():
            fields["Exécution"] = run_url()
        send_alert(settings, f"{type(e).__name__}: {e}"[:1500], title=f"conso-hdf : échec de l'étape « {step} »",
                   fields=fields)
        raise


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    parser = argparse.ArgumentParser(description="Pipeline MLOps – conso électrique Hauts-de-France")
    parser.add_argument(
        "step",
        choices=["ingest", "train", "predict", "backfill", "monitor", "daily", "weekly", "schedule", "notify-test"],
    )
    parser.add_argument("--force-promote", action="store_true", help="Promouvoir même si pas meilleur")
    parser.add_argument("--no-auto-retrain", action="store_true", help="(daily) ne pas réentraîner en cas d'alerte")
    args = parser.parse_args()

    if args.step == "schedule":
        schedule()
    elif args.step == "notify-test":
        notify_test()
    else:
        run_step(args.step, force_promote=args.force_promote, auto_retrain=not args.no_auto_retrain)


if __name__ == "__main__":
    main()
