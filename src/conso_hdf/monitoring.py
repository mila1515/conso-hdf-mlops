"""Monitoring du modèle en production.

Trois contrôles :
1. **Fraîcheur des données** : la dernière consommation ingérée doit être récente, sinon les
   prévisions portent sur des heures déjà passées (ingestion bloquée, API source en panne...).
2. **Performance réelle** : les prévisions passées sont comparées aux consommations observées
   (MAPE sur les `MONITORING_DAYS` derniers jours).
3. **Dérive des données (data drift)** : la distribution des variables récentes est comparée à la
   même période de l'année dans l'historique (années précédentes), par la distance de Wasserstein
   normalisée — la même mesure que la méthode `wasserstein` d'Evidently. Evidently produit le
   rapport HTML lisible ; le calcul de la décision est fait ici, sans dépendre du format interne
   des résultats d'Evidently (qui change d'une version à l'autre).

Si un seuil est dépassé -> alerte (log + webhook Slack/Discord optionnel) et
recommandation de réentraînement.
"""

from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from conso_hdf.config import Settings, settings
from conso_hdf.features import CALENDAR_FEATURES, build_features
from conso_hdf.ingestion import data_freshness, load_data
from conso_hdf.metrics import regression_metrics
from conso_hdf.model_store import load_champion
from conso_hdf.notify import send_alert

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", module=r"evidently.*")
warnings.filterwarnings("ignore", message=".*PeriodArray.*")

REPORT_GLOB = "drift_report_*.html"


def performance_check(cfg: Settings, conso: pd.DataFrame) -> dict:
    """MAPE des prévisions passées vs consommation observée.

    Les prévisions réellement émises par le pipeline sont utilisées en priorité ; les prévisions
    simulées (backfill, 1er lancement) ne servent que s'il n'y a pas encore assez de vraies prévisions.
    """
    if not cfg.predictions_path.exists():
        return {"status": "no_predictions"}
    preds = pd.read_parquet(cfg.predictions_path)
    if "simulated" not in preds.columns:
        preds["simulated"] = False
    preds["simulated"] = preds["simulated"].fillna(False).astype(bool)
    since = conso.index.max() - pd.Timedelta(days=cfg.monitoring_days)
    joined = preds.set_index("target_time").join(conso, how="inner")
    joined = joined[joined.index > since]
    real = joined[~joined["simulated"]]
    used = real if len(real) >= 12 else joined
    if len(used) < 12:
        return {"status": "not_enough_data", "n_hours": int(len(used))}
    m = regression_metrics(used["consommation"], used["prediction_mw"])
    n_sim = int(used["simulated"].sum())
    return {"status": "ok", "n_hours": int(len(used)), "n_simulated": n_sim, "includes_simulated": n_sim > 0, **m}


def seasonal_reference(feats: pd.DataFrame, current: pd.DataFrame, window_days: int = 21) -> pd.DataFrame:
    """Référence = mêmes semaines de l'année, **années précédentes uniquement**.

    La consommation électrique est très saisonnière : comparer novembre à octobre détecterait
    une « dérive » à chaque changement de saison. On compare donc la période actuelle à la même
    période des années passées : une vraie anomalie (vague de froid, crise...) ressort clairement.
    L'année en cours est exclue pour que la référence ne contienne pas la période observée elle-même.
    """
    center = current.index.min() + (current.index.max() - current.index.min()) / 2
    hist = feats[feats.index < current.index.min()]
    delta = pd.Timedelta(days=window_days)
    mask = np.zeros(len(hist), dtype=bool)
    for year in sorted(set(hist.index.year)):
        if year >= center.year:
            continue
        try:
            c = center.replace(year=year)
        except ValueError:  # 29 février
            c = center.replace(year=year, day=28)
        mask |= (hist.index >= c - delta) & (hist.index <= c + delta)
    return hist[mask]


def wasserstein_norm(reference: pd.Series, current: pd.Series) -> float:
    """Distance de Wasserstein normalisée par l'écart-type de la référence (méthode Evidently)."""
    norm = max(float(np.std(reference)), 0.001)
    return float(wasserstein_distance(reference, current) / norm)


def _evidently_project(cfg: Settings, columns: list[str]):
    """Workspace + projet Evidently (créés au besoin, avec leurs graphiques d'évolution).

    Serveur Evidently UI (EVIDENTLY_SERVER_URL) s'il est configuré : l'UI garde le workspace en mémoire,
    il faut donc passer par son API pour que les nouveaux rapports apparaissent sans la redémarrer.
    Sinon, workspace local (dossier EVIDENTLY_WORKSPACE, lu par `evidently ui` au démarrage)."""
    from evidently.sdk.models import PanelMetric
    from evidently.sdk.panels import DashboardPanelPlot
    from evidently.ui.workspace import RemoteWorkspace, Workspace

    if cfg.evidently_server_url:
        ws = RemoteWorkspace(cfg.evidently_server_url)
    else:
        ws = Workspace.create(str(cfg.evidently_workspace))
    found = ws.search_project(cfg.evidently_project)
    if found:
        return ws, found[0]
    project = ws.create_project(
        cfg.evidently_project,
        description="Dérive des variables (7 derniers jours) vs même période des années précédentes. "
                    "Distance de Wasserstein normalisée, seuil "
                    f"{cfg.wasserstein_threshold}.",
    )
    project.dashboard.add_panel(DashboardPanelPlot(
        title="Part des variables en dérive", subtitle="0 = aucune, 1 = toutes", size="half",
        values=[PanelMetric(legend="Part", metric="DriftedColumnsCount", metric_labels={"value_type": "share"})],
        plot_params={"plot_type": "line"},
    ))
    project.dashboard.add_panel(DashboardPanelPlot(
        title="Nombre de variables en dérive", size="half",
        values=[PanelMetric(legend="Nombre", metric="DriftedColumnsCount", metric_labels={"value_type": "count"})],
        plot_params={"plot_type": "bar"},
    ))
    project.dashboard.add_panel(DashboardPanelPlot(
        title="Distance de Wasserstein par variable", subtitle=f"Dérive au-delà de {cfg.wasserstein_threshold}",
        size="full",
        values=[PanelMetric(legend=c, metric="ValueDrift", metric_labels={"column": c}) for c in columns],
        plot_params={"plot_type": "line"},
    ))
    project.save()
    return ws, project


def _save_evidently_report(cfg: Settings, current: pd.DataFrame, reference: pd.DataFrame) -> str | None:
    """Rapport Evidently : HTML (dashboard Streamlit) + run dans le workspace (Evidently UI).
    Non bloquant : renvoie le nom du fichier HTML, ou None."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        preset = DataDriftPreset(num_method="wasserstein", num_threshold=cfg.wasserstein_threshold)
        snapshot = Report([preset]).run(current_data=current, reference_data=reference)
        name = f"drift_report_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.html"
        snapshot.save_html(str(cfg.reports_dir / name))
    except Exception as e:  # noqa: BLE001 - le rapport visuel ne doit pas bloquer le monitoring
        log.warning("Rapport Evidently non généré : %s", e)
        return None
    try:
        ws, project = _evidently_project(cfg, list(current.columns))
        ws.add_run(project.id, snapshot, include_data=False, name=name.removesuffix(".html"))
    except Exception as e:  # noqa: BLE001
        log.warning("Rapport non ajouté au workspace Evidently : %s", e)
    return name


def purge_old_reports(cfg: Settings) -> None:
    """Ne conserve que les `REPORTS_KEEP` rapports HTML les plus récents."""
    reports = sorted(cfg.reports_dir.glob(REPORT_GLOB))
    for old in reports[: max(len(reports) - cfg.reports_keep, 0)]:
        old.unlink(missing_ok=True)


def drift_check(cfg: Settings, conso, weather, meta: dict) -> dict:
    feats = build_features(conso, weather if meta["use_weather"] else None)
    # Les variables purement calendaires ne « dérivent » pas au sens utile : on les exclut
    monitored = [c for c in meta["features"] if c not in CALENDAR_FEATURES and c in feats.columns]
    current = feats[feats.index > conso.index.max() - pd.Timedelta(days=cfg.monitoring_days)][monitored].dropna()
    if current.empty:
        return {"status": "not_enough_data"}
    reference = seasonal_reference(feats, current)[monitored].dropna()
    if len(reference) < 100:
        return {"status": "not_enough_reference"}

    # Distance de Wasserstein normalisée : plus robuste que les tests statistiques (K-S),
    # qui signalent des écarts insignifiants dès que les échantillons sont grands.
    scores = {c: round(wasserstein_norm(reference[c], current[c]), 3) for c in monitored}
    drifted = [c for c, v in scores.items() if v >= cfg.wasserstein_threshold]
    report = _save_evidently_report(cfg, current, reference)
    purge_old_reports(cfg)
    return {
        "status": "ok",
        "method": "wasserstein_norm",
        "threshold": cfg.wasserstein_threshold,
        "reference_hours": int(len(reference)),
        "current_hours": int(len(current)),
        "drift_share": len(drifted) / len(monitored),
        "drifted_columns": drifted,
        "scores": scores,
        "report": report,  # nom de fichier, relatif à REPORTS_DIR
    }


def report_path(cfg: Settings, report: str | None) -> Path | None:
    """Chemin d'un rapport à partir de ce qui est stocké dans le JSON (nom relatif ou ancien chemin absolu)."""
    return cfg.reports_dir / Path(report).name if report else None


def run_monitoring(cfg: Settings = settings) -> dict:
    cfg.ensure_dirs()
    conso, weather = load_data(cfg)
    _, meta = load_champion(cfg)
    freshness = data_freshness(conso, cfg)
    perf = performance_check(cfg, conso)
    drift = drift_check(cfg, conso, weather, meta)

    alerts: list[dict] = []
    if freshness["stale"]:
        alerts.append({"severity": "critique", "type": "data",
                       "message": f"Données obsolètes : dernière observation {freshness['last_observation']} "
                                  f"({freshness['data_age_hours']} h, seuil {cfg.max_data_age_hours} h). "
                                  "Vérifier l'ingestion."})
    perf_ok = perf.get("status") == "ok"
    if perf_ok and perf["mape"] > cfg.mape_alert_threshold:
        alerts.append({"severity": "critique", "type": "performance",
                       "message": f"MAPE des {cfg.monitoring_days} derniers jours = {perf['mape']:.2f} % "
                                  f"(seuil {cfg.mape_alert_threshold} %)"})
    drift_alert = drift.get("status") == "ok" and drift["drift_share"] >= cfg.drift_share_threshold
    if drift_alert:
        alerts.append({"severity": "avertissement", "type": "drift",
                       "message": f"Dérive sur {drift['drift_share']:.0%} des variables "
                                  f"vs même période des années passées : {drift['drifted_columns']}"})

    # Règle de décision : la dérive seule est un signal d'attention (ex. semaine plus froide que
    # d'habitude) ; on ne réentraîne que si la performance se dégrade réellement, ou si on ne peut
    # pas la mesurer alors que les données dérivent. Des données obsolètes ne justifient pas un
    # réentraînement (il faut d'abord réparer l'ingestion).
    retrain = any(a["type"] == "performance" for a in alerts) or (drift_alert and not perf_ok)

    summary = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "model_version": meta["model_version"],
        "freshness": freshness,
        "performance": perf,
        "drift": drift,
        "alerts": alerts,
        "retrain_recommended": retrain,
    }
    (cfg.reports_dir / "monitoring_latest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    titles = {"data": "Données obsolètes", "performance": "Performance dégradée", "drift": "Dérive des données"}
    for a in alerts:
        fields = {"Modèle": f"v{meta['model_version']}"}
        if perf_ok:
            fields[f"MAPE {cfg.monitoring_days} j"] = f"{perf['mape']:.2f} %"
        if a["type"] == "performance":
            fields["Réentraînement"] = "oui" if retrain else "non"
        send_alert(cfg, a["message"], severity=a["severity"], title=f"conso-hdf : {titles[a['type']]}", fields=fields)
    if not alerts:
        log.info("Monitoring OK : %s", {k: perf.get(k) for k in ("mape", "n_hours")})
    return summary
