# ============================================================
# main.py — Orchestrateur principal du projet
# ============================================================
#
# Ordre d'exécution :
#   1. Extraction des données (SQL)
#   2. Prétraitement & détection des outliers
#   3. Feature Engineering
#   4. Data Augmentation
#   5. Modèles ML classiques (RF, GBM, XGBoost, Stacking)
#   6. Modèles séries temporelles (ARIMA, Prophet, LSTM)
#   7. Comparaison globale tous modèles
#   8. Prévision CA mensuel futur
#
# Installation des librairies séries temporelles :
#   pip install statsmodels prophet tensorflow
#
# Usage :
#   python main.py
# ============================================================

from data_extraction import load_data
from preprocessing   import preprocess, detect_outliers_iqr, detect_outliers_zscore, detect_outliers_isolation_forest, winsorize
from features        import build_features
from augmentation    import augment_data
from modeling        import (
    train_and_evaluate, feature_importance,
    train_arima, train_prophet, train_lstm,
    compare_all_models,
    predict_ca_global, predict_future_ca,
)
from visualization   import (
    plot_distribution,
    plot_outliers,
    plot_before_after_cleaning,
    plot_quantity_by_article,
    plot_feature_importance,
    plot_augmentation,
    plot_model_comparison,
    plot_ca_global,
)


def main():
    print("\n" + "🚀 " * 18)
    print("   PIPELINE PRÉVISION DES VENTES")
    print("🚀 " * 18 + "\n")

    # ── 1. EXTRACTION ────────────────────────────────────────
    print("\n📦 ÉTAPE 1 — Extraction des données")
    df_raw = load_data()

    # ── 2. PRÉ-TRAITEMENT ─────────────────────────────────────
    print("\n🔧 ÉTAPE 2 — Prétraitement & Outliers")
    plot_distribution(df_raw)

    outliers_iqr, lower, upper = detect_outliers_iqr(df_raw)
    outliers_z                 = detect_outliers_zscore(df_raw)
    df_with_iso, _             = detect_outliers_isolation_forest(df_raw)

    plot_outliers(df_with_iso, outliers_iqr, outliers_z, lower, upper)

    df_clean  = preprocess(df_raw, method="iqr")
    df_winsor = winsorize(df_raw, lower, upper)
    plot_before_after_cleaning(df_raw, df_clean, df_winsor)

    # ── 3. FEATURE ENGINEERING ───────────────────────────────
    print("\n⚙️  ÉTAPE 3 — Feature Engineering")
    df_feat = build_features(df_clean)
    plot_quantity_by_article(df_feat)

    importances_pre = feature_importance(df_feat)
    plot_feature_importance(importances_pre, "Importance des variables (avant augmentation)")

    # ── 4. DATA AUGMENTATION ─────────────────────────────────
    print("\n📊 ÉTAPE 4 — Data Augmentation")
    df_augmented = augment_data(df_feat)
    plot_augmentation(df_feat, df_augmented)

    importances_post = feature_importance(df_augmented)
    plot_feature_importance(importances_post, "Importance des variables (après augmentation)")

    # ── 5. MODÈLES ML CLASSIQUES ─────────────────────────────
    print("\n🤖 ÉTAPE 5 — Modèles ML Classiques")
    best_model, resultats_ml = train_and_evaluate(df_augmented)
    plot_model_comparison(resultats_ml)

    # ── 6. MODÈLES SÉRIES TEMPORELLES ────────────────────────
    print("\n📈 ÉTAPE 6 — Modèles Séries Temporelles")

    # ARIMA — sur la série agrégée (tous articles)
    print("\n--- ARIMA ---")
    arima_res = train_arima(df_feat, article=None, n_test=30)

    # Prophet — sur la série agrégée
    print("\n--- PROPHET ---")
    prophet_res = train_prophet(df_feat, article=None, n_test=30)

    # LSTM — sur la série agrégée
    # Conseil : filtrer sur un article fréquent pour un LSTM plus précis
    # ex : train_lstm(df_feat, article="ART001", n_steps=30, epochs=50)
    print("\n--- LSTM ---")
    lstm_res = train_lstm(df_feat, article=None, n_steps=30, epochs=50, n_test=30)

    # ── 7. COMPARAISON GLOBALE ────────────────────────────────
    print("\n🏆 ÉTAPE 7 — Comparaison Globale Tous Modèles")
    tous_resultats = compare_all_models(resultats_ml, arima_res, prophet_res, lstm_res)
    plot_model_comparison(tous_resultats)

    # ── 8. PRÉVISION CA MENSUEL FUTUR ────────────────────────
    print("\n💰 ÉTAPE 8 — Prévision CA Mensuel (12 mois futurs)")
    # Utilise le meilleur modèle ML (Stacking) + dernier prix par article
    ca_mensuel, ca_total = predict_future_ca(best_model, df_feat, n_mois=12)
    plot_ca_global(ca_mensuel)

    # ── RÉSUMÉ FINAL ──────────────────────────────────────────
    print("\n" + "=" * 55)
    print("🏆 RÉSUMÉ FINAL")
    print("=" * 55)

    # Trouver le meilleur modèle global
    meilleur = max(
        tous_resultats,
        key=lambda k: tous_resultats[k].get("R²", tous_resultats[k].get("R2", 0))
    )
    m = tous_resultats[meilleur]
    print(f"Meilleur modèle global : {meilleur}")
    print(f"R²   : {m.get('R²', m.get('R2', 0)):.4f}")
    print(f"MAE  : {m['MAE']:.2f}")
    print(f"RMSE : {m['RMSE']:.2f}")
    print(f"\n💰 CA Prévu sur 12 mois : {ca_total:,.2f} DT")
    print("\n✅ Pipeline terminé avec succès !")


if __name__ == "__main__":
    main()
