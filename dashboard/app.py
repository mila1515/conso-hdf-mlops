"""Dashboard Streamlit : prévision vs réalité + santé du modèle.

Lancement : streamlit run dashboard/app.py
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from conso_hdf.config import settings
from conso_hdf.ingestion import data_freshness
from conso_hdf.model_store import champion_exists, load_champion
from conso_hdf.monitoring import report_path
from conso_hdf.predict import forecast

st.set_page_config(page_title="Conso électrique Hauts-de-France", page_icon="⚡", layout="wide")
st.title("⚡ Prévision de la consommation électrique – Hauts-de-France")
st.caption("Données RTE éCO2mix (ODRÉ) · météo Open-Meteo · modèle XGBoost suivi avec MLflow")

if not settings.conso_path.exists() or not champion_exists(settings):
    st.warning("Pas encore de données ou de modèle. Lancez : `python -m conso_hdf.pipeline weekly`")
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
        f"Données obsolètes : dernière observation il y a {freshness['data_age_hours']:.0f} h "
        f"(seuil {settings.max_data_age_hours:.0f} h). Vérifiez l'ingestion."
    )

# --- KPIs ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("MAPE test (modèle)", f"{meta['metrics_test']['mape']:.2f} %",
          help="30 derniers jours, jamais vus à l'entraînement, avec la température observée (légèrement optimiste).")
c2.metric(
    "MAPE baseline naïve",
    f"{meta['baseline_test']['mape']:.2f} %",
    delta=f"{meta['metrics_test']['mape'] - meta['baseline_test']['mape']:.2f} pts vs modèle",
    delta_color="off",
)
if monitoring and monitoring["performance"].get("status") == "ok":
    perf = monitoring["performance"]
    label = "MAPE simulé" if perf.get("includes_simulated") else "MAPE réel"
    c3.metric(f"{label} ({settings.monitoring_days} j)", f"{perf['mape']:.2f} %",
              help="Inclut des prévisions simulées (backfill)." if perf.get("includes_simulated") else None)
else:
    c3.metric(f"MAPE réel ({settings.monitoring_days} j)", "—")
c4.metric("Dernière conso observée", f"{conso['consommation'].iloc[-1]:,.0f} MW".replace(",", " "))

# --- Graphique principal ---
since = conso.index.max() - pd.Timedelta(days=days)
fig = go.Figure()
recent = conso[conso.index > since]
fig.add_trace(go.Scatter(x=recent.index, y=recent["consommation"], name="Consommation réelle", line=dict(width=2)))
if not preds.empty:
    p = preds[preds.target_time > since]
    fig.add_trace(go.Scatter(x=p.target_time, y=p.prediction_mw, name="Prévisions passées", line=dict(dash="dot")))
try:
    fc = cached_forecast(_mtime(settings.conso_path), _mtime(settings.champion_dir / "model.json"))
    fig.add_trace(go.Scatter(x=fc.target_time, y=fc.prediction_mw, name="Prévision 24 h", line=dict(width=3)))
except Exception as e:  # noqa: BLE001
    st.info(f"Prévision indisponible : {e}")
fig.update_layout(height=460, yaxis_title="MW", hovermode="x unified", legend=dict(orientation="h", y=1.08))
st.plotly_chart(fig, use_container_width=True)

# --- Monitoring ---
st.subheader("🩺 Santé du modèle")
if monitoring is None:
    st.info("Aucun monitoring exécuté : `python -m conso_hdf.pipeline monitor`")
else:
    if monitoring["alerts"]:
        for a in monitoring["alerts"]:
            (st.error if a["severity"] == "critique" else st.warning)(a["message"])
        st.write("Réentraînement recommandé : **" + ("oui" if monitoring["retrain_recommended"] else "non") + "**")
    else:
        st.success("Aucune alerte : performance et distribution des données dans les seuils.")
    drift = monitoring.get("drift", {})
    if drift.get("status") == "ok":
        st.write(
            f"Part des variables en dérive : **{drift['drift_share']:.0%}** — {drift['drifted_columns'] or 'aucune'}"
        )
        st.caption(
            f"Historique de tous les rapports et évolution de la dérive : [Evidently UI]({settings.evidently_ui_url})"
        )
        with st.expander("Voir le rapport Evidently complet"):
            path = report_path(settings, drift.get("report"))
            if path is not None and path.exists():
                components.html(path.read_text(encoding="utf-8"), height=900, scrolling=True)
            else:
                st.write("Rapport introuvable.")
