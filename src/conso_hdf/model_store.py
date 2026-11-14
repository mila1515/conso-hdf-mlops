"""Stockage local du modèle « champion » (celui servi en production par l'API).

MLflow garde l'historique complet de tous les modèles (registre + alias `champion`) ;
ce dossier local contient une copie du champion pour que l'API n'ait pas besoin
d'un serveur MLflow pour démarrer.

Deux fichiers modèle :
- `model.json`      : modèle de **production**, réentraîné sur toutes les données disponibles ;
- `model_eval.json` : modèle d'**évaluation**, entraîné avant le jeu de test. C'est lui qui sert
  aux comparaisons champion / challenger et au backfill, car il n'a jamais vu les données récentes.
"""

from __future__ import annotations

import json
from typing import Literal

import xgboost as xgb

from conso_hdf.config import Settings, settings

ModelKind = Literal["production", "eval"]
_FILES = {"production": "model.json", "eval": "model_eval.json"}


def save_champion(
    model: xgb.XGBRegressor,
    metadata: dict,
    cfg: Settings = settings,
    eval_model: xgb.XGBRegressor | None = None,
) -> None:
    cfg.champion_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(cfg.champion_dir / _FILES["production"])
    eval_path = cfg.champion_dir / _FILES["eval"]
    if eval_model is not None:
        eval_model.save_model(eval_path)
    elif eval_path.exists():
        eval_path.unlink()  # éviter de garder le modèle d'évaluation d'un ancien champion
    (cfg.champion_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))


def champion_exists(cfg: Settings = settings) -> bool:
    return (cfg.champion_dir / _FILES["production"]).exists()


def load_champion(cfg: Settings = settings, kind: ModelKind = "production") -> tuple[xgb.XGBRegressor, dict]:
    """Charge le champion. `kind="eval"` renvoie le modèle d'évaluation s'il existe
    (sinon le modèle de production : cas d'un champion créé avant l'ajout du réentraînement complet,
    dont le modèle de production *était* le modèle d'évaluation)."""
    path = cfg.champion_dir / _FILES["production"]
    if not path.exists():
        raise FileNotFoundError("Aucun modèle champion : lancez `python -m conso_hdf.pipeline train`.")
    if kind == "eval" and (cfg.champion_dir / _FILES["eval"]).exists():
        path = cfg.champion_dir / _FILES["eval"]
    model = xgb.XGBRegressor()
    model.load_model(path)
    metadata = json.loads((cfg.champion_dir / "metadata.json").read_text())
    return model, metadata
