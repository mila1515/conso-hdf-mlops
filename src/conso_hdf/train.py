"""Entraînement, suivi MLflow et promotion du modèle « champion ».

Étapes :
1. Découpage temporel : les `TEST_DAYS` derniers jours servent de jeu de test,
   les `TEST_DAYS` précédents de jeu de validation (early stopping).
2. Baseline « naïve saisonnière » (= même heure la semaine précédente) pour avoir un point de comparaison.
3. Entraînement du modèle d'**évaluation** XGBoost (train seul, early stopping sur la validation).
4. Journalisation dans MLflow : paramètres, métriques, graphiques.
5. Comparaison « challenger vs champion » sur le même jeu de test, avec les modèles d'évaluation
   (aucun des deux n'a vu ce jeu de test) : le nouveau n'est promu que s'il est meilleur.
6. Si promu : réentraînement du modèle de **production** sur 100 % des données (train + val + test)
   avec le nombre d'arbres trouvé à l'étape 3, puis enregistrement dans le registre MLflow.
   Sans cette étape, le modèle servi ignorerait les ~60 derniers jours, les plus utiles
   pour une série aussi saisonnière.
"""

from __future__ import annotations

import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import xgboost as xgb
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from conso_hdf.config import Settings, settings
from conso_hdf.features import TARGET, build_features, feature_columns, training_frame
from conso_hdf.ingestion import load_data
from conso_hdf.metrics import regression_metrics
from conso_hdf.model_store import champion_exists, load_champion, save_champion

log = logging.getLogger(__name__)

XGB_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 7,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "early_stopping_rounds": 100,
    "tree_method": "hist",
    "random_state": 42,
}


def split(df: pd.DataFrame, test_days: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cutoff = df.index.max() - pd.Timedelta(days=test_days)
    train_full, test = df[df.index <= cutoff], df[df.index > cutoff]
    val_cut = train_full.index.max() - pd.Timedelta(days=test_days)
    return train_full[train_full.index <= val_cut], train_full[train_full.index > val_cut], test


def setup_mlflow(cfg: Settings) -> None:
    """Configure MLflow. En suivi local (sqlite / fichiers), l'emplacement des artefacts est fixé
    explicitement (`MLFLOW_ARTIFACTS_DIR`) pour être identique en local, en CI et dans le cache."""
    uri = cfg.mlflow_tracking_uri
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(cfg.mlflow_experiment) is None:
        kwargs = {}
        if uri.startswith(("sqlite:", "file:")) or "://" not in uri:
            cfg.mlflow_artifacts_dir.mkdir(parents=True, exist_ok=True)
            kwargs["artifact_location"] = cfg.mlflow_artifacts_dir.resolve().as_uri()
        mlflow.create_experiment(cfg.mlflow_experiment, **kwargs)
    mlflow.set_experiment(cfg.mlflow_experiment)


def _plot_test(test: pd.DataFrame, pred, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(test.index, test[TARGET], label="Réel", lw=1.2)
    ax.plot(test.index, pred, label="Prévision", lw=1.2, alpha=0.8)
    ax.set_title("Consommation Hauts-de-France – jeu de test (MW)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_importance(model: xgb.XGBRegressor, features: list[str], path: Path) -> None:
    imp = pd.Series(model.feature_importances_, index=features).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot.barh(ax=ax)
    ax.set_title("Importance des variables")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def champion_test_mape(cfg: Settings, conso: pd.DataFrame, weather: pd.DataFrame | None,
                       test_df: pd.DataFrame) -> float | None:
    """MAPE du champion actuel (modèle d'évaluation) sur le jeu de test du challenger.

    Le champion est évalué avec **ses propres** variables, même si la liste a changé depuis
    (ajout d'une variable, météo activée/désactivée...) : la comparaison n'est plus sautée.
    Renvoie None s'il n'y a pas de champion, ou s'il ne peut pas être évalué sur les données
    actuelles (variable indisponible) : il ne pourrait alors pas non plus servir de prévision.
    """
    if not champion_exists(cfg):
        return None
    champ, meta = load_champion(cfg, kind="eval")
    champ_features = meta["features"]
    full = build_features(conso, weather if meta.get("use_weather") else None)
    missing = [c for c in champ_features if c not in full.columns]
    if missing:
        log.warning("Champion non évaluable sur les données actuelles (variables manquantes : %s)", missing)
        return None
    x = full.loc[test_df.index, champ_features]
    return regression_metrics(test_df[TARGET], champ.predict(x))["mape"]


def train(cfg: Settings = settings, force_promote: bool = False) -> dict:
    cfg.ensure_dirs()
    conso, weather = load_data(cfg)
    use_weather = weather is not None
    features = feature_columns(use_weather)
    df = training_frame(conso, weather)
    train_df, val_df, test_df = split(df, cfg.test_days)
    log.info("Train %d h | Val %d h | Test %d h", len(train_df), len(val_df), len(test_df))

    setup_mlflow(cfg)

    with mlflow.start_run(run_name=f"xgb-{datetime.now():%Y%m%d-%H%M}") as run:
        # --- Baseline naïve saisonnière ---
        baseline = regression_metrics(test_df[TARGET], test_df["lag_168h"])
        mlflow.log_metrics({f"baseline_{k}": v for k, v in baseline.items()})

        # --- XGBoost : modèle d'évaluation (n'a jamais vu le jeu de test) ---
        eval_model = xgb.XGBRegressor(**XGB_PARAMS)
        eval_model.fit(
            train_df[features], train_df[TARGET],
            eval_set=[(val_df[features], val_df[TARGET])],
            verbose=False,
        )
        pred = eval_model.predict(test_df[features])
        metrics = regression_metrics(test_df[TARGET], pred)
        log.info("Baseline MAPE %.2f %% | XGBoost MAPE %.2f %%", baseline["mape"], metrics["mape"])

        best_iteration = eval_model.best_iteration
        n_final = int(best_iteration) + 1 if best_iteration is not None else XGB_PARAMS["n_estimators"]
        mlflow.log_params(dict(XGB_PARAMS))
        mlflow.log_params({
            "best_iteration": best_iteration,
            "refit_n_estimators": n_final,
            "use_weather": use_weather,
            "n_features": len(features),
            "train_start": str(train_df.index.min()),
            "train_end": str(val_df.index.max()),
            "test_start": str(test_df.index.min()),
            "test_end": str(test_df.index.max()),
            "data_end": str(df.index.max()),
            "n_rows": len(df),
        })
        mlflow.log_metrics(metrics)
        mlflow.set_tag("weather_in_test", "observed")  # température observée, pas prévue

        with tempfile.TemporaryDirectory() as tmp:
            _plot_test(test_df, pred, Path(tmp) / "test_predictions.png")
            _plot_importance(eval_model, features, Path(tmp) / "feature_importance.png")
            mlflow.log_artifacts(tmp, artifact_path="plots")

        signature = infer_signature(test_df[features].head(50).astype(float), pred[:50])
        input_example = test_df[features].head(3).astype(float)
        # Modèle d'évaluation : conservé dans le run (traçabilité), non enregistré dans le registre
        mlflow.xgboost.log_model(eval_model, name="model_eval", signature=signature, input_example=input_example)

        # --- Challenger vs champion (mêmes données de test, modèles d'évaluation) ---
        champion_mape = champion_test_mape(cfg, conso, weather, test_df)
        if champion_mape is not None:
            mlflow.log_metric("champion_mape_same_test", champion_mape)
        promoted = force_promote or champion_mape is None or metrics["mape"] < champion_mape
        mlflow.set_tag("promoted", str(promoted))

        version = None
        if promoted:
            # --- Modèle de production : réentraîné sur 100 % des données ---
            final_params = {k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"}
            final_params["n_estimators"] = n_final
            prod_model = xgb.XGBRegressor(**final_params)
            prod_model.fit(df[features], df[TARGET], verbose=False)

            model_info = mlflow.xgboost.log_model(
                prod_model,
                name="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=cfg.model_name,
            )
            version = int(model_info.registered_model_version)
            MlflowClient().set_registered_model_alias(cfg.model_name, "champion", str(version))
            save_champion(
                prod_model,
                {
                    "model_name": cfg.model_name,
                    "model_version": version,
                    "mlflow_run_id": run.info.run_id,
                    "trained_at": datetime.now(timezone.utc).isoformat(),
                    "features": features,
                    "use_weather": use_weather,
                    "metrics_test": metrics,
                    "baseline_test": baseline,
                    "train_end": str(val_df.index.max()),
                    "test_start": str(test_df.index.min()),
                    "test_end": str(test_df.index.max()),
                    "data_end": str(df.index.max()),
                    "refit_on_full_data": True,
                    "refit_n_estimators": n_final,
                },
                cfg,
                eval_model=eval_model,
            )
            log.info("Nouveau champion : version %s (MAPE test %.2f %%, réentraîné sur %d h)",
                     version, metrics["mape"], len(df))
        else:
            log.info("Challenger non promu (MAPE %.2f %% >= champion %.2f %%)", metrics["mape"], champion_mape)

    return {"version": version, "metrics": metrics, "baseline": baseline,
            "champion_mape": champion_mape, "promoted": promoted}
