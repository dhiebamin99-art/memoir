# ============================================================
# main_stock.py — Pipeline complet : ML Vente + Stocks + Achats
# ============================================================
#
# ORDRE D'EXÉCUTION :
#
#   ┌─────────────────────────────────────────────────┐
#   │  BLOC A — PIPELINE ML (issu de ML_VENTE)        │
#   │  1. Extraction SQL données de vente             │
#   │  2. Prétraitement & détection outliers          │
#   │  3. Feature Engineering                         │
#   │  4. Data Augmentation                           │
#   │  5. Entraînement modèles ML (RF, GBM, XGB,      │
#   │     Stacking) + ARIMA + Prophet + LSTM          │
#   │  6. Comparaison & sélection meilleur modèle     │
#   │  7. 💾 SAUVEGARDE des modèles                   │
#   │  8. Prévision CA mensuel futur (12 mois)        │
#   └─────────────────────────────────────────────────┘
#
#   ┌─────────────────────────────────────────────────┐
#   │  BLOC B — PLANIFICATION STOCKS & ACHATS         │
#   │  9.  Extraction SQL : BOM + Stock PF + Stock MP │
#   │  10. Calcul Besoins PF                          │
#   │  11. Calcul Achats PF = Prévision - Stock PF    │
#   │  12. Calcul Besoins MP via BOM                  │
#   │  13. Achats MP nets = Besoins MP - Stock MP     │
#   │  14. Simulation Stocks Finaux                   │
#   │  15. Visualisations complètes                   │
#   │  16. Export Excel                               │
#   └─────────────────────────────────────────────────┘
#
# Usage :
#   python main_stock.py
#
# Prérequis :
#   pip install pyodbc pandas numpy scikit-learn xgboost
#   pip install statsmodels prophet torch joblib openpyxl
# ============================================================

import os
import sys
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

# ── Imports ML_VENTE (bloc A) ────────────────────────────────
from data_extraction import load_data, get_connection
from preprocessing   import (preprocess, detect_outliers_iqr,
                              detect_outliers_zscore,
                              detect_outliers_isolation_forest,
                              winsorize)
from features        import build_features
from augmentation    import augment_data
from modeling        import (
    train_and_evaluate, feature_importance,
    train_arima, train_prophet, train_lstm,
    compare_all_models,
    predict_future_ca,
)
from visualization   import (
    plot_distribution, plot_outliers,
    plot_before_after_cleaning,
    plot_quantity_by_article,
    plot_feature_importance,
    plot_augmentation,
    plot_model_comparison,
    plot_ca_global,
)

# ── Imports Stocks & Achats (bloc B) ─────────────────────────
from model_saver        import (save_stacking_model, save_arima_model,
                                 save_prophet_model, save_lstm_model,
                                 print_saved_models)
from stock_planning     import run_stock_planning
from visualization_stock import plot_all_stock

# ── Répertoire graphiques ─────────────────────────────────────
os.makedirs("graphs", exist_ok=True)


# ════════════════════════════════════════════════════════════
# PARAMÈTRES GLOBAUX — modifiez ici
# ════════════════════════════════════════════════════════════

N_MOIS_PREVISION  = 12     # Horizon prévision (mois)
RUN_ARIMA         = True   # Activer ARIMA
RUN_PROPHET       = True   # Activer Prophet
RUN_LSTM          = True   # Activer LSTM (nécessite PyTorch)
LSTM_EPOCHS       = 50     # Époques LSTM
EXPORT_EXCEL      = True   # Exporter plan d'achats en .xlsx


# ════════════════════════════════════════════════════════════
# HELPER — Prévision détaillée par article (pour BOM)
# ════════════════════════════════════════════════════════════

def _predict_future_ca_detail(model, df, n_mois: int) -> pd.DataFrame:
    """
    Retourne le détail des prévisions par article × mois.
    Colonnes : article | annee | mois | quantite_prevue | ca_prevu
    """
    from config import FEATURES

    df          = df.copy()
    df["date_facture"] = pd.to_datetime(df["date_facture"])
    derniere_date = df["date_facture"].max()

    if "prix_dernier" not in df.columns:
        raise ValueError("Colonne 'prix_dernier' absente. Vérifiez data_extraction.py")

    prix_ref = (
        df.groupby("article")["prix_dernier"].last()
        .reset_index().rename(columns={"prix_dernier": "prix_ref"})
    )
    stats_article = (
        df.groupby("article")
        .agg(qte_moy_art=("quantite","mean"),
             qte_lag_1  =("quantite","last"),
             qte_lag_7  =("quantite","last"))
        .reset_index().merge(prix_ref, on="article", how="left")
    )

    futures = []
    for mois_offset in range(1, n_mois + 1):
        future_date = derniere_date + pd.DateOffset(months=mois_offset)
        for _, row in stats_article.iterrows():
            qte_moy_mois = df[
                (df["article"] == row["article"]) &
                (df["mois"]    == future_date.month)
            ]["quantite"].mean()
            if pd.isna(qte_moy_mois):
                qte_moy_mois = row["qte_moy_art"]
            futures.append({
                "date_facture"   : future_date,
                "article"        : row["article"],
                "mois"           : future_date.month,
                "trimestre"      : future_date.quarter,
                "annee"          : future_date.year,
                "jour"           : 15,
                "jour_semaine"   : future_date.dayofweek,
                "qte_moy_article": row["qte_moy_art"],
                "qte_moy_mois"   : qte_moy_mois,
                "qte_lag_1"      : row["qte_lag_1"],
                "qte_lag_7"      : row["qte_lag_7"],
                "prix_dernier"   : row["prix_ref"],
            })

    df_future = pd.DataFrame(futures)
    available = [f for f in FEATURES if f in df_future.columns]
    df_future["quantite_prevue"] = model.predict(df_future[available]).clip(min=0)
    df_future["ca_prevu"]        = df_future["quantite_prevue"] * df_future["prix_dernier"]

    return df_future[["article","annee","mois","quantite_prevue","ca_prevu"]]


# ════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ════════════════════════════════════════════════════════════

def main():
    print("\n" + "🚀 " * 18)
    print("   PIPELINE COMPLET — PRÉVISION VENTES + PLANIFICATION STOCKS")
    print("🚀 " * 18 + "\n")

    # ──────────────────────────────────────────────────────────
    # BLOC A — PIPELINE ML
    # ──────────────────────────────────────────────────────────

    # ── 1. EXTRACTION ────────────────────────────────────────
    print("\n" + "█" * 55)
    print("  ÉTAPE 1 — Extraction des données de vente (SQL)")
    print("█" * 55)
    df_raw = load_data()
    print(f"  → {len(df_raw)} lignes | {df_raw['article'].nunique()} articles")

    # ── 2. PRÉ-TRAITEMENT ────────────────────────────────────
    print("\n" + "█" * 55)
    print("  ÉTAPE 2 — Prétraitement & Détection des Outliers")
    print("█" * 55)
    plot_distribution(df_raw)

    outliers_iqr, lower, upper = detect_outliers_iqr(df_raw)
    outliers_z                 = detect_outliers_zscore(df_raw)
    df_with_iso, _             = detect_outliers_isolation_forest(df_raw)

    plot_outliers(df_with_iso, outliers_iqr, outliers_z, lower, upper)

    df_clean  = preprocess(df_raw, method="iqr")
    df_winsor = winsorize(df_raw, lower, upper)
    plot_before_after_cleaning(df_raw, df_clean, df_winsor)

    # ── 3. FEATURE ENGINEERING ───────────────────────────────
    print("\n" + "█" * 55)
    print("  ÉTAPE 3 — Feature Engineering")
    print("█" * 55)
    df_feat = build_features(df_clean)
    plot_quantity_by_article(df_feat)

    importances_pre = feature_importance(df_feat)
    plot_feature_importance(importances_pre, "Importance des variables (avant augmentation)")

    # ── 4. DATA AUGMENTATION ─────────────────────────────────
    print("\n" + "█" * 55)
    print("  ÉTAPE 4 — Data Augmentation")
    print("█" * 55)
    df_augmented = augment_data(df_feat)
    plot_augmentation(df_feat, df_augmented)

    importances_post = feature_importance(df_augmented)
    plot_feature_importance(importances_post, "Importance des variables (après augmentation)")

    # ── 5. MODÈLES ML CLASSIQUES ─────────────────────────────
    print("\n" + "█" * 55)
    print("  ÉTAPE 5 — Entraînement Modèles ML Classiques")
    print("█" * 55)
    best_model, resultats_ml = train_and_evaluate(df_augmented)
    plot_model_comparison(resultats_ml)

    # ── 6. MODÈLES SÉRIES TEMPORELLES ────────────────────────
    print("\n" + "█" * 55)
    print("  ÉTAPE 6 — Modèles Séries Temporelles")
    print("█" * 55)

    arima_res = prophet_res = lstm_res = {}

    if RUN_ARIMA:
        print("\n--- ARIMA ---")
        arima_res = train_arima(df_augmented, article=None, n_test=30)

    if RUN_PROPHET:
        print("\n--- PROPHET ---")
        prophet_res = train_prophet(df_augmented, article=None, n_test=30)

    if RUN_LSTM:
        print("\n--- LSTM (PyTorch) ---")
        lstm_res = train_lstm(df_augmented, article=None,
                              n_steps=30, epochs=LSTM_EPOCHS, n_test=30)

    # ── 7. COMPARAISON GLOBALE ────────────────────────────────
    print("\n" + "█" * 55)
    print("  ÉTAPE 7 — Comparaison Globale Tous Modèles")
    print("█" * 55)
    tous_resultats = compare_all_models(
        resultats_ml, arima_res, prophet_res, lstm_res
    )
    plot_model_comparison(tous_resultats)

    # ── 7b. SAUVEGARDE DES MODÈLES ────────────────────────────
    print("\n" + "█" * 55)
    print("  ÉTAPE 7b — 💾 Sauvegarde des Modèles")
    print("█" * 55)

    from config import FEATURES
    features_used = [f for f in FEATURES if f in df_augmented.columns]

    # Métriques du meilleur modèle ML
    metriques_stacking = resultats_ml.get("Stacking", list(resultats_ml.values())[-1])

    save_stacking_model(best_model, metriques_stacking, features_used)

    if arima_res and "modele" in arima_res:
        save_arima_model(arima_res)

    if prophet_res and "modele" in prophet_res:
        save_prophet_model(prophet_res)

    if lstm_res and "modele" in lstm_res:
        save_lstm_model(lstm_res)

    print_saved_models()

    # ── 8. PRÉVISION CA FUTUR ────────────────────────────────
    print("\n" + "█" * 55)
    print(f"  ÉTAPE 8 — Prévision CA {N_MOIS_PREVISION} mois futurs")
    print("█" * 55)

    # Prévision agrégée (affichage)
    ca_mensuel, ca_total = predict_future_ca(
        best_model, df_augmented, n_mois=N_MOIS_PREVISION
    )
    plot_ca_global(ca_mensuel)

    # Prévision détaillée par article (nécessaire pour BOM)
    print("\n  Calcul du détail par article pour la planification stocks...")
    ca_detail = _predict_future_ca_detail(
        best_model, df_augmented, n_mois=N_MOIS_PREVISION
    )
    print(f"  → {len(ca_detail)} lignes (article × mois)")

    # ──────────────────────────────────────────────────────────
    # BLOC B — PLANIFICATION STOCKS & ACHATS
    # ──────────────────────────────────────────────────────────

    print("\n" + "█" * 55)
    print("  ÉTAPE 9-16 — Planification Stocks & Achats")
    print("█" * 55)

    # Nouvelle connexion SQL pour les données stocks
    from data_extraction import get_connection
    conn = get_connection()

    try:
        planning = run_stock_planning(
            conn            = conn,
            ca_mensuel_detail = ca_detail,
            n_mois          = N_MOIS_PREVISION,
            export_excel    = EXPORT_EXCEL,
        )
    finally:
        conn.close()

    # Visualisations stocks
    plot_all_stock(planning, n_mois=N_MOIS_PREVISION)

    # ──────────────────────────────────────────────────────────
    # RÉSUMÉ FINAL
    # ──────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("🏆 RÉSUMÉ FINAL DU PIPELINE")
    print("=" * 60)

    # Meilleur modèle
    meilleur = max(
        tous_resultats,
        key=lambda k: tous_resultats[k].get("R²", tous_resultats[k].get("R2", 0))
    )
    m = tous_resultats[meilleur]
    print(f"\n  Meilleur modèle  : {meilleur}")
    print(f"  R²   : {m.get('R²', m.get('R2', 0)):.4f}")
    print(f"  MAE  : {m['MAE']:.2f}")
    print(f"  RMSE : {m['RMSE']:.2f}")

    # CA
    print(f"\n  💰 CA Prévu sur {N_MOIS_PREVISION} mois : {ca_total:,.2f} DT")

    # Stocks
    pf = planning["achats_pf"]
    mp = planning["achats_mp"]
    print(f"\n  🏭 Articles PF à commander        : {(pf['achats_pf'] > 0).sum()}")
    print(f"  🏭 Composants MP à commander      : {(mp['achats_mp_net'] > 0).sum()}")
    print(f"  🔴 PF en risque de rupture        : {(pf['statut_pf'] == '🔴 Rupture risque').sum()}")
    print(f"  🔴 MP en risque de rupture        : {(mp['statut_mp'] == '🔴 Rupture risque').sum()}")

    # Fichiers générés
    print(f"\n  📁 Fichiers générés :")
    print(f"     models/                → modèles sauvegardés")
    print(f"     graphs/                → {len(os.listdir('graphs'))} graphiques PNG")
    if EXPORT_EXCEL:
        print(f"     plan_achats.xlsx      → plan d'achats complet")

    print("\n✅ Pipeline terminé avec succès !")
    print("=" * 60)


# ════════════════════════════════════════════════════════════
# MODE RECHARGEMENT — Utilise le modèle sauvegardé
# ════════════════════════════════════════════════════════════

def run_with_saved_model(n_mois: int = 12):
    """
    Exécute uniquement le bloc B (planification stocks) en
    chargeant le modèle Stacking déjà sauvegardé.
    Utile pour relancer la planification sans ré-entraîner.
    """
    from model_saver    import load_stacking_model
    from data_extraction import load_data, get_connection

    print("\n" + "=" * 55)
    print("🔄 MODE RECHARGEMENT — Modèle sauvegardé")
    print("=" * 55)

    # Charger modèle sauvegardé
    model, meta = load_stacking_model()
    print(f"  Features : {meta.get('features', 'non renseignées')}")

    # Nouvelles données ventes
    print("\n  Extraction des données de vente...")
    df_raw  = load_data()
    df_clean = preprocess(df_raw, method="iqr")
    df_feat  = build_features(df_clean)

    # Prévision CA
    print(f"\n  Calcul prévision CA {n_mois} mois...")
    ca_detail = _predict_future_ca_detail(model, df_feat, n_mois=n_mois)

    # Planification
    conn = get_connection()
    try:
        planning = run_stock_planning(
            conn              = conn,
            ca_mensuel_detail = ca_detail,
            n_mois            = n_mois,
            export_excel      = True,
        )
    finally:
        conn.close()

    plot_all_stock(planning, n_mois=n_mois)
    print("\n✅ Planification terminée.")
    return planning


if __name__ == "__main__":
    # Par défaut : pipeline complet (entraînement + planification)
    # Pour utiliser le modèle sauvegardé : run_with_saved_model()
    main()
