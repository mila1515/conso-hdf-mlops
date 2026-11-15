"""Dashboard Streamlit : prévision vs réalité + santé du modèle.

Lancement : streamlit run dashboard/app.py
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from conso_hdf.config import settings
from conso_hdf.ingestion import data_freshness
from conso_hdf.model_store import champion_exists, load_champion
from conso_hdf.monitoring import report_path
from conso_hdf.predict import forecast

# Noms lisibles des variables du modèle (le monitoring travaille avec les noms techniques)
FEATURE_LABELS = {
    "lag_24h": "Conso il y a 24 h",
    "lag_48h": "Conso il y a 48 h",
    "lag_168h": "Conso il y a 7 jours",
    "lag_336h": "Conso il y a 14 jours",
    "roll_mean_24h": "Conso moyenne sur 24 h",
    "roll_std_24h": "Variabilité de la conso sur 24 h",
    "roll_mean_168h": "Conso moyenne sur 7 jours",
    "temperature": "Température",
    "heating_degree": "Besoin de chauffage (degrés sous 15 °C)",
    "temp_mean_24h": "Température moyenne sur 24 h",
}


def label(col: str) -> str:
    return FEATURE_LABELS.get(col, col)


def fmt_pct(x: float) -> str:
    return f"{x:.2f} %".replace(".", ",")


def fmt_mw(x: float) -> str:
    return f"{x:,.0f} MW".replace(",", " ")


st.set_page_config(page_title="Conso électrique Hauts-de-France", page_icon="⚡", layout="wide")
st.title(":material/bolt: Prévision de la consommation électrique – Hauts-de-France")
st.caption("Données RTE éCO2mix (ODRÉ) · météo Open-Meteo · modèle XGBoost suivi avec MLflow")

if not settings.conso_path.exists() or not champion_exists(settings):
    st.warning("Lancez : `python -m conso_hdf.pipeline weekly`", title="Pas encore de données ou de modèle")
    st.stop()


def _mtime(path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(ttl=600, show_spinner=False)
def cached_forecast(conso_mtime: float, model_mtime: float) -> pd.DataFrame:
    """Prévision mise en cache : recalculée seulement si les données ou le modèle changent (ou après 10 min)."""
    return forecast(settings)


conso = pd.read_parquet(settings.conso_path)
freshness = data_freshness(conso, settings)
_, meta = load_champion(settings)
preds = pd.read_parquet(settings.predictions_path) if settings.predictions_path.exists() else pd.DataFrame()
monitoring_path = settings.reports_dir / "monitoring_latest.json"
monitoring = json.loads(monitoring_path.read_text()) if monitoring_path.exists() else None

days = st.sidebar.slider("Historique affiché (jours)", 3, 60, 14)
st.sidebar.markdown(f"**Modèle champion :** v{meta['model_version']}")
st.sidebar.markdown(f"Entraîné le {meta['trained_at'][:16].replace('T', ' ')} UTC")
st.sidebar.markdown(f"Données jusqu'au {conso.index.max():%d/%m/%Y %H:%M} UTC")
if freshness["stale"]:
    st.error(
        f"Dernière observation il y a {freshness['data_age_hours']:.0f} h "
        f"(seuil {settings.max_data_age_hours:.0f} h). Vérifiez l'ingestion.",
        title="Données obsolètes",
        icon=":material/error:",
    )

try:
    fc = cached_forecast(_mtime(settings.conso_path), _mtime(settings.champion_dir / "model.json"))
except Exception as e:  # noqa: BLE001
    fc = None
    st.info(str(e), title="Prévision indisponible")

# --- Indicateurs du modèle ---
with st.grid(4, min_column_width=200, gap="small"):
    st.metric(
        "MAPE test (modèle)",
        fmt_pct(meta["metrics_test"]["mape"]),
        border=True,
        height="stretch",
        help="30 derniers jours, jamais vus à l'entraînement, avec la température observée (légèrement optimiste).",
    )
    st.metric(
        "MAPE baseline naïve",
        fmt_pct(meta["baseline_test"]["mape"]),
        border=True,
        height="stretch",
        help="Même heure, une semaine avant. Le modèle doit faire nettement mieux.",
    )
    if fc is not None and not fc.empty:
        peak = fc.loc[fc.prediction_mw.idxmax()]
        st.metric(
            "Pic prévu (24 h)",
            fmt_mw(peak.prediction_mw),
            border=True,
            height="stretch",
            help=f"Attendu le {peak.target_time.tz_convert('Europe/Paris'):%d/%m à %Hh} (heure de Paris).",
        )
    else:
        st.metric("Pic prévu (24 h)", "—", border=True, height="stretch")
    st.metric("Dernière conso observée", fmt_mw(conso["consommation"].iloc[-1]), border=True, height="stretch")

# --- Graphique principal ---
since = conso.index.max() - pd.Timedelta(days=days)
fig = go.Figure()
recent = conso[conso.index > since]
fig.add_trace(go.Scatter(x=recent.index, y=recent["consommation"], name="Consommation réelle", line=dict(width=2)))
if not preds.empty:
    p = preds[preds.target_time > since]
    fig.add_trace(go.Scatter(x=p.target_time, y=p.prediction_mw, name="Prévisions passées", line=dict(dash="dot")))
if fc is not None:
    fig.add_trace(go.Scatter(x=fc.target_time, y=fc.prediction_mw, name="Prévision 24 h", line=dict(width=3)))
fig.update_layout(height=460, yaxis_title="MW", hovermode="x unified", legend=dict(orientation="h", y=1.08))
st.plotly_chart(
    fig, width="stretch", alt="Consommation réelle, prévisions passées et prévision des 24 prochaines heures"
)

# --- Santé du modèle ---
st.subheader(":material/monitor_heart: Santé du modèle")
if monitoring is None:
    st.info("Lancez : `python -m conso_hdf.pipeline monitor`", title="Aucun monitoring exécuté")
    st.stop()

perf = monitoring.get("performance", {})
drift = monitoring.get("drift", {})
fresh = monitoring.get("freshness") or freshness
alerts = monitoring["alerts"]
retrain = monitoring["retrain_recommended"]
drift_ok = drift.get("status") == "ok"
scores: dict[str, float] = drift.get("scores", {}) if drift_ok else {}
drifted = drift.get("drifted_columns", []) if drift_ok else []
threshold = drift.get("threshold", settings.wasserstein_threshold)
perf_ok = perf.get("status") == "ok"

# Un seul bandeau d'état, rédigé pour un lecteur non technique
critical = [a for a in alerts if a["severity"] == "critique"]
if critical:
    st.error(
        "\n\n".join(a["message"] for a in critical)
        + ("\n\nRéentraînement automatique déclenché." if retrain else ""),
        title="Action requise",
        icon=":material/error:",
    )
elif drifted:
    top = sorted(drifted, key=lambda c: scores[c], reverse=True)[:2]
    names = " et ".join(f"« {label(c)} »" for c in top)
    verdict = (
        f"La performance reste dans les seuils (MAPE {fmt_pct(perf['mape'])} < "
        f"{settings.mape_alert_threshold:.0f} %) : pas de réentraînement, on surveille."
        if perf_ok and not retrain
        else "Réentraînement recommandé."
    )
    st.warning(
        f"{len(drifted)} variables sur {len(scores)} s'écartent de la même période des années passées, "
        f"surtout {names}. {verdict}",
        title="Les conditions sont inhabituelles",
        icon=":material/warning:",
    )
else:
    st.success(
        "Données à jour, performance et distribution des variables dans les seuils.",
        title="Modèle en bonne santé",
        icon=":material/check_circle:",
    )

with st.grid(4, min_column_width=200, gap="small"):
    st.metric(
        "Fraîcheur des données",
        f"{fresh['data_age_hours']:.1f} h".replace(".", ","),
        border=True,
        height="stretch",
        help=f"Âge de la dernière consommation observée. Alerte au-delà de {settings.max_data_age_hours:.0f} h.",
    )
    st.metric(
        f"MAPE {'simulé' if perf.get('includes_simulated') else 'réel'} ({settings.monitoring_days} j)",
        fmt_pct(perf["mape"]) if perf_ok else "—",
        border=True,
        height="stretch",
        help=f"Erreur moyenne des prévisions passées. Alerte au-delà de {settings.mape_alert_threshold:.0f} %.",
    )
    st.metric(
        "Variables en dérive",
        f"{len(drifted)} / {len(scores)}" if drift_ok else "—",
        border=True,
        height="stretch",
        help="Comparées à la même période des années passées. Avertissement à partir de "
        f"{settings.drift_share_threshold:.0%} des variables.",
    )
    st.metric(
        "Réentraînement",
        "Recommandé" if retrain else "Non nécessaire",
        border=True,
        height="stretch",
        help="Déclenché uniquement si la performance se dégrade (une dérive seule ne suffit pas).",
    )

if drift_ok:
    with st.container(border=True):
        st.markdown("**Dérive par variable**")
        st.caption(
            "Distance de Wasserstein normalisée entre les 7 derniers jours et la même période des années "
            f"passées. Au-delà de {str(threshold).replace('.', ',')} : la variable est considérée en dérive."
        )
        table = pd.DataFrame(
            [
                {"Variable": label(col), "Distance": score, "État": "🟠 Dérive" if score >= threshold else "🟢 Stable"}
                for col, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            ]
        )
        st.dataframe(
            table,
            hide_index=True,
            column_config={
                "Variable": st.column_config.TextColumn(width="large"),
                "Distance": st.column_config.ProgressColumn(
                    format="%.2f", min_value=0.0, max_value=max(1.0, float(table["Distance"].max()))
                ),
                "État": st.column_config.TextColumn(width="small"),
            },
            alt="Distance de dérive de chaque variable, de la plus forte à la plus faible",
        )

checked_at = pd.Timestamp(monitoring["checked_at"]).tz_convert("Europe/Paris")
with st.container(horizontal=True, vertical_alignment="center"):
    st.caption(f"Dernier contrôle : {checked_at:%d/%m/%Y à %H:%M} (heure de Paris)")
    st.space("stretch")
    st.link_button(
        "Historique dans Evidently UI", settings.evidently_ui_url, icon=":material/open_in_new:", type="tertiary"
    )

if drift_ok:
    report = st.expander(
        "Rapport Evidently détaillé", icon=":material/description:", on_change="rerun", key="evidently_report"
    )
    if report.open:  # chargé uniquement quand l'expander est ouvert
        with report:
            path = report_path(settings, drift.get("report"))
            if path is not None and path.exists():
                st.iframe(path, height=900, alt="Rapport de dérive Evidently")
            else:
                st.info("Le fichier du rapport n'existe plus (rapports anciens purgés).", title="Rapport introuvable")
