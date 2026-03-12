# ============================================================
# validation.py — Validation du modèle Stacking
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model  import Ridge
from sklearn.ensemble      import (RandomForestRegressor,
                                   GradientBoostingRegressor,
                                   StackingRegressor)
from sklearn.metrics       import mean_absolute_error, mean_squared_error, r2_score
from xgboost               import XGBRegressor

from data_extraction import load_data
from preprocessing   import preprocess
from features        import build_features
from config          import FEATURES, TARGET, RANDOM_STATE


# ════════════════════════════════════════════════════════════
# PARAMÈTRE — Mois à valider
# ════════════════════════════════════════════════════════════
# None = dernier mois disponible automatiquement
# Exemple manuel : MOIS_VALIDATION = 9 / ANNEE_VALIDATION = 2023
MOIS_VALIDATION  = None
ANNEE_VALIDATION = None


# ════════════════════════════════════════════════════════════
# ÉTAPE 1 — Chargement
# ════════════════════════════════════════════════════════════

def charger_donnees() -> pd.DataFrame:
    print("=" * 55)
    print("ETAPE 1 — Chargement des données (SQL Sage X3)")
    print("=" * 55)
    df = load_data()
    df["date_facture"] = pd.to_datetime(df["date_facture"])
    print(f"  {len(df)} lignes | "
          f"{df['article'].nunique()} articles | "
          f"{df['date_facture'].min().strftime('%d-%m-%Y')} → "
          f"{df['date_facture'].max().strftime('%d-%m-%Y')}")
    return df


# ════════════════════════════════════════════════════════════
# ÉTAPE 2 — Nettoyage
# ════════════════════════════════════════════════════════════

def nettoyer(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 55)
    print("ETAPE 2 — Nettoyage")
    print("=" * 55)
    return preprocess(df)


# ════════════════════════════════════════════════════════════
# ÉTAPE 3 — Features
# ════════════════════════════════════════════════════════════

def construire_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 55)
    print("ETAPE 3 — Feature Engineering")
    print("=" * 55)
    return build_features(df)


# ════════════════════════════════════════════════════════════
# ÉTAPE 4 — Entraînement Stacking
# ════════════════════════════════════════════════════════════

def entrainer_stacking(df: pd.DataFrame, mois: int, annee: int) -> tuple:
    print("\n" + "=" * 55)
    print(f"ETAPE 4 — Entraînement Stacking")
    print(f"  Train : tout sauf {mois:02d}/{annee}")
    print(f"  Test  : {mois:02d}/{annee} uniquement")
    print("=" * 55)

    df = df.copy()
    df["date_facture"] = pd.to_datetime(df["date_facture"])

    masque_test = (
        (df["date_facture"].dt.month == mois) &
        (df["date_facture"].dt.year  == annee)
    )
    df_train = df[~masque_test].copy()
    df_test  = df[ masque_test].copy()

    print(f"  Train : {len(df_train)} lignes")
    print(f"  Test  : {len(df_test)} lignes | "
          f"{df_test['article'].nunique()} articles")

    # Garder seulement les features présentes dans les données
    features_dispo = [f for f in FEATURES if f in df_train.columns]
    manquantes     = [f for f in FEATURES if f not in df_train.columns]
    if manquantes:
        print(f"  Features absentes (ignorées) : {manquantes}")
    print(f"  Features utilisées : {features_dispo}")

    # Encoder les colonnes texte (ex: article = 'PRFMARSOL002')
    from sklearn.preprocessing import LabelEncoder
    df_train = df_train.copy()
    df_test  = df_test.copy()
    for col in features_dispo:
        if df_train[col].dtype == object:
            le = LabelEncoder()
            le.fit(df_train[col].astype(str))
            df_train[col] = le.transform(df_train[col].astype(str))
            df_test[col]  = df_test[col].astype(str).map(
                lambda x, le=le: le.transform([x])[0] if x in le.classes_ else 0
            )

    X_train = df_train[features_dispo].fillna(0)
    y_train = df_train[TARGET]
    X_test  = df_test[features_dispo].fillna(0)
    y_true  = df_test[TARGET].values

    modele = StackingRegressor(
        estimators=[
            ("rf",  RandomForestRegressor(
                        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
            ("gbm", GradientBoostingRegressor(
                        n_estimators=200, learning_rate=0.05,
                        random_state=RANDOM_STATE)),
            ("xgb", XGBRegressor(
                        n_estimators=300, learning_rate=0.05, max_depth=6,
                        random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)),
        ],
        final_estimator=Ridge(),
        cv=5,
    )

    print("  Entraînement en cours...")
    modele.fit(X_train, y_train)
    y_pred = modele.predict(X_test).clip(min=0)

    return modele, df_test, y_true, y_pred, features_dispo


# ════════════════════════════════════════════════════════════
# ÉTAPE 5 — Métriques
# ════════════════════════════════════════════════════════════

def afficher_metriques(y_true, y_pred, mois, annee) -> dict:
    print("\n" + "=" * 55)
    print(f"ETAPE 5 — Résultats validation {mois:02d}/{annee}")
    print("=" * 55)

    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    denom = np.where(y_true == 0, 1, y_true)
    mape  = np.mean(np.abs((y_true - y_pred) / denom)) * 100

    print(f"  R²   : {r2:.4f}  {'✅' if r2 >= 0.85 else '⚠️'}")
    print(f"  MAE  : {mae:.2f} unités")
    print(f"  RMSE : {rmse:.2f} unités")
    print(f"  MAPE : {mape:.2f}%  "
          f"({'✅ Bon' if mape < 15 else '⚠️ Acceptable' if mape < 25 else '❌ Elevé'})")

    return {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape}


# ════════════════════════════════════════════════════════════
# ÉTAPE 6 — Calcul CA
# ════════════════════════════════════════════════════════════

def calculer_ca(df_test: pd.DataFrame, y_pred: np.ndarray, mois, annee) -> pd.DataFrame:
    print("\n" + "=" * 55)
    print(f"ETAPE 6 — Calcul CA — {mois:02d}/{annee}")
    print("  CA = Somme( quantite_predite × prix_article )")
    print("=" * 55)

    # Vérifier que prix_dernier existe
    if "prix_dernier" not in df_test.columns:
        # Utiliser prix_unitaire si prix_dernier absent
        df_test = df_test.copy()
        df_test["prix_dernier"] = df_test["prix_unitaire"]
        print("  (prix_unitaire utilisé comme prix de référence)")

    df_r = df_test.copy()
    df_r["quantite_reelle"]  = df_test[TARGET].values
    df_r["quantite_predite"] = y_pred
    df_r["ca_reel"]          = df_r["quantite_reelle"]  * df_r["prix_dernier"]
    df_r["ca_predit"]        = df_r["quantite_predite"] * df_r["prix_dernier"]

    resume = (
        df_r.groupby("article")
        .agg(
            qte_reelle   = ("quantite_reelle",  "sum"),
            qte_predite  = ("quantite_predite", "sum"),
            prix_article = ("prix_dernier",     "mean"),
            ca_reel      = ("ca_reel",          "sum"),
            ca_predit    = ("ca_predit",         "sum"),
        )
        .reset_index()
    )

    resume["ecart_pct"] = (
        (resume["ca_predit"] - resume["ca_reel"])
        / resume["ca_reel"].replace(0, np.nan) * 100
    ).fillna(0)

    # Affichage
    print(f"\n  {'Article':12s} {'Qte Réelle':>12} {'Qte Prédite':>12} "
          f"{'Prix':>8} {'CA Réel':>14} {'CA Prédit':>14} {'Ecart%':>8}")
    print("  " + "-" * 86)
    for _, r in resume.iterrows():
        signe = "+" if r["ecart_pct"] >= 0 else ""
        print(f"  {r['article']:12s} "
              f"{r['qte_reelle']:>12.0f} "
              f"{r['qte_predite']:>12.0f} "
              f"{r['prix_article']:>8.2f} "
              f"{r['ca_reel']:>14,.2f} "
              f"{r['ca_predit']:>14,.2f} "
              f"{signe}{r['ecart_pct']:>7.1f}%")

    ca_reel_total   = resume["ca_reel"].sum()
    ca_predit_total = resume["ca_predit"].sum()
    ecart_total     = ca_predit_total - ca_reel_total
    ecart_pct_total = ecart_total / ca_reel_total * 100 if ca_reel_total > 0 else 0

    print("  " + "=" * 86)
    print(f"  {'TOTAL':12s} "
          f"{resume['qte_reelle'].sum():>12.0f} "
          f"{resume['qte_predite'].sum():>12.0f} "
          f"{'':>8} "
          f"{ca_reel_total:>14,.2f} "
          f"{ca_predit_total:>14,.2f} "
          f"{'+' if ecart_total >= 0 else ''}{ecart_pct_total:>7.1f}%")

    print(f"\n  CA Réel    : {ca_reel_total:>14,.2f} DT")
    print(f"  CA Prédit  : {ca_predit_total:>14,.2f} DT")
    print(f"  Écart      : {ecart_total:>+14,.2f} DT ({ecart_pct_total:+.1f}%)")

    return resume


# ════════════════════════════════════════════════════════════
# GRAPHIQUES
# ════════════════════════════════════════════════════════════

def plot_validation(y_true, y_pred, resume, metriques, mois, annee):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Validation Stacking — {mois:02d}/{annee}\n"
        f"R²={metriques['R2']:.4f} | MAE={metriques['MAE']:.1f} | "
        f"MAPE={metriques['MAPE']:.1f}%",
        fontsize=12, fontweight="bold"
    )

    # Scatter réel vs prédit
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.5, color="#3498db", s=20)
    lim = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Parfait")
    ax.set_xlabel("Quantité Réelle")
    ax.set_ylabel("Quantité Prédite")
    ax.set_title("Réel vs Prédit")
    ax.legend()

    # CA par article
    ax = axes[1]
    x  = np.arange(len(resume))
    w  = 0.35
    ax.bar(x - w/2, resume["ca_reel"],   w, label="CA Réel",   color="#2ecc71")
    ax.bar(x + w/2, resume["ca_predit"], w, label="CA Prédit", color="#e74c3c", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(resume["article"], rotation=45, fontsize=7)
    ax.set_ylabel("CA (DT)")
    ax.set_title("CA Réel vs Prédit par Article")
    ax.legend()

    # Écart %
    ax     = axes[2]
    ecarts = resume["ecart_pct"].values
    cols   = ["#e74c3c" if e < 0 else "#2ecc71" for e in ecarts]
    ax.bar(resume["article"], ecarts, color=cols)
    ax.axhline(0,   color="black",  linewidth=0.8)
    ax.axhline( 10, color="orange", linestyle="--", linewidth=1, label="±10%")
    ax.axhline(-10, color="orange", linestyle="--", linewidth=1)
    ax.set_ylabel("Écart CA (%)")
    ax.set_title("Écart CA Prédit vs Réel")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("validation_stacking.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Graphique sauvegardé : validation_stacking.png")


# ════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ════════════════════════════════════════════════════════════

def main():
    # Chargement
    df_raw = charger_donnees()

    # Détection automatique du mois si non spécifié
    mois  = MOIS_VALIDATION
    annee = ANNEE_VALIDATION
    if mois is None or annee is None:
        df_raw["date_facture"] = pd.to_datetime(df_raw["date_facture"])
        derniere = df_raw["date_facture"].max()
        mois, annee = derniere.month, derniere.year
        print(f"\n  Mois de validation détecté automatiquement : {mois:02d}/{annee}")

    print("\n" + "█" * 55)
    print(f"  VALIDATION STACKING — {mois:02d}/{annee}")
    print("  CA = Somme( quantite_predite × prix_article )")
    print("█" * 55)

    df_clean = nettoyer(df_raw)
    df_feat  = construire_features(df_clean)

    modele, df_test, y_true, y_pred, features = entrainer_stacking(df_feat, mois, annee)
    metriques = afficher_metriques(y_true, y_pred, mois, annee)
    resume    = calculer_ca(df_test, y_pred, mois, annee)
    plot_validation(y_true, y_pred, resume, metriques, mois, annee)

    print("\n" + "=" * 55)
    print("  VALIDATION TERMINÉE")
    print("=" * 55)


if __name__ == "__main__":
    main()