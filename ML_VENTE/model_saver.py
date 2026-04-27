# ============================================================
# model_saver.py — Sauvegarde et chargement des modèles ML
# ============================================================
#
# Sauvegarde : joblib (rapide, efficace pour sklearn/xgboost)
# Arborescence générée :
#   models/
#     stacking_model.joblib     ← modèle Stacking principal
#     metadata.json             ← métriques + date + features
#     arima_model.pkl           ← ARIMA (statsmodels)
#     prophet_model.pkl         ← Prophet
#     lstm_model.pt             ← LSTM (PyTorch state_dict)
#     lstm_scaler.joblib        ← MinMaxScaler associé au LSTM
# ============================================================

import os
import json
import pickle
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

MODELS_DIR = Path("models")


def _ensure_dir():
    """Crée le répertoire models/ s'il n'existe pas."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════
# SAUVEGARDE
# ════════════════════════════════════════════════════════════

def save_stacking_model(model, metriques: dict, features: list) -> str:
    """
    Sauvegarde le modèle Stacking + métadonnées.

    Paramètres
    ----------
    model      : StackingRegressor entraîné
    metriques  : dict {"R²": float, "MAE": float, "RMSE": float}
    features   : liste des colonnes utilisées

    Retourne
    --------
    str : chemin du fichier sauvegardé
    """
    _ensure_dir()
    path = MODELS_DIR / "stacking_model.joblib"
    joblib.dump(model, path, compress=3)

    # Métadonnées JSON
    meta = {
        "model_type"   : "StackingRegressor",
        "saved_at"     : datetime.now().isoformat(),
        "features"     : features,
        "metriques"    : {k: float(v) for k, v in metriques.items()},
        "joblib_path"  : str(path),
    }
    _save_metadata(meta)

    size_mb = path.stat().st_size / 1_048_576
    print(f"✅ Stacking sauvegardé : {path}  ({size_mb:.2f} MB)")
    return str(path)


def save_arima_model(arima_result: dict) -> str:
    """Sauvegarde le modèle ARIMA (statsmodels ResultsWrapper)."""
    _ensure_dir()
    path = MODELS_DIR / "arima_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(arima_result["modele"], f)
    print(f"✅ ARIMA sauvegardé : {path}")
    return str(path)


def save_prophet_model(prophet_result: dict) -> str:
    """Sauvegarde le modèle Prophet."""
    _ensure_dir()
    path = MODELS_DIR / "prophet_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(prophet_result["modele"], f)
    print(f"✅ Prophet sauvegardé : {path}")
    return str(path)


def save_lstm_model(lstm_result: dict) -> tuple:
    """
    Sauvegarde le modèle LSTM PyTorch (state_dict) + son scaler.

    Retourne
    --------
    (str, str) : (chemin modèle, chemin scaler)
    """
    try:
        import torch
    except ImportError:
        print("⚠️  PyTorch non installé — LSTM non sauvegardé.")
        return None, None

    _ensure_dir()
    model_path  = MODELS_DIR / "lstm_model.pt"
    scaler_path = MODELS_DIR / "lstm_scaler.joblib"

    torch.save(lstm_result["modele"].state_dict(), model_path)
    joblib.dump(lstm_result["scaler"], scaler_path)

    print(f"✅ LSTM sauvegardé   : {model_path}")
    print(f"✅ Scaler sauvegardé : {scaler_path}")
    return str(model_path), str(scaler_path)


def _save_metadata(meta: dict):
    """Écrit / met à jour models/metadata.json."""
    meta_path = MODELS_DIR / "metadata.json"
    existing  = {}
    if meta_path.exists():
        with open(meta_path) as f:
            existing = json.load(f)
    existing.update(meta)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


# ════════════════════════════════════════════════════════════
# CHARGEMENT
# ════════════════════════════════════════════════════════════

def load_stacking_model():
    """
    Charge le modèle Stacking depuis models/stacking_model.joblib.

    Retourne
    --------
    (model, metadata_dict)
    """
    path = MODELS_DIR / "stacking_model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {path}\nLancez d'abord le pipeline principal.")

    model = joblib.load(path)

    meta_path = MODELS_DIR / "metadata.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    print(f"✅ Stacking chargé depuis {path}")
    if "saved_at" in meta:
        print(f"   Sauvegardé le : {meta['saved_at']}")
    if "metriques" in meta:
        m = meta["metriques"]
        print(f"   R²={m.get('R²', m.get('R2', '?')):.4f} | MAE={m.get('MAE', '?'):.2f} | RMSE={m.get('RMSE', '?'):.2f}")

    return model, meta


def load_arima_model():
    """Charge le modèle ARIMA."""
    path = MODELS_DIR / "arima_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"ARIMA introuvable : {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"✅ ARIMA chargé depuis {path}")
    return model


def load_prophet_model():
    """Charge le modèle Prophet."""
    path = MODELS_DIR / "prophet_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Prophet introuvable : {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"✅ Prophet chargé depuis {path}")
    return model


def load_lstm_model(n_steps: int = 30, hidden_size: int = 64,
                    num_layers: int = 2, dropout: float = 0.2):
    """Charge le modèle LSTM PyTorch + son scaler."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError("PyTorch requis : pip install torch")

    from modeling import _build_lstm_pytorch   # import local

    model_path  = MODELS_DIR / "lstm_model.pt"
    scaler_path = MODELS_DIR / "lstm_scaler.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"LSTM introuvable : {model_path}")

    model = _build_lstm_pytorch(n_steps, hidden_size, num_layers, dropout)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    scaler = joblib.load(scaler_path)

    print(f"✅ LSTM chargé depuis {model_path}")
    return model, scaler


# ════════════════════════════════════════════════════════════
# RAPPORT DE SAUVEGARDE
# ════════════════════════════════════════════════════════════

def print_saved_models():
    """Affiche un résumé des modèles sauvegardés."""
    print("\n" + "=" * 55)
    print("📁 MODÈLES SAUVEGARDÉS")
    print("=" * 55)

    if not MODELS_DIR.exists():
        print("  Aucun modèle sauvegardé (dossier models/ absent)")
        return

    for f in sorted(MODELS_DIR.iterdir()):
        size = f.stat().st_size
        unit = "KB" if size < 1_048_576 else "MB"
        val  = size / 1024 if size < 1_048_576 else size / 1_048_576
        print(f"  {f.name:30s}  {val:7.2f} {unit}")

    meta_path = MODELS_DIR / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"\n  Dernière sauvegarde : {meta.get('saved_at', 'inconnue')}")
        if "metriques" in meta:
            m = meta["metriques"]
            r2 = m.get("R²", m.get("R2", None))
            print(f"  Metriques Stacking  : R²={r2} | MAE={m['MAE']} | RMSE={m['RMSE']}")
