# ============================================================
# visualization.py — Graphiques (distribution, outliers, augmentation)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ── Distribution de la quantité ───────────────────────────────
def plot_distribution(df: pd.DataFrame, col: str = "quantite") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].hist(df[col], bins=50, color="steelblue", edgecolor="black")
    axes[0].set_title("Distribution Quantité")
    axes[0].set_xlabel("Quantité")
    axes[0].set_ylabel("Fréquence")

    axes[1].boxplot(df[col], vert=True, patch_artist=True,
                    boxprops=dict(facecolor="lightblue"))
    axes[1].set_title("Boxplot Quantité")
    axes[1].set_ylabel("Quantité")

    stats.probplot(df[col], dist="norm", plot=axes[2])
    axes[2].set_title("QQ-Plot (Normalité)")

    plt.suptitle("Analyse du Bruit — Quantité", fontsize=14)
    plt.tight_layout()
    plt.show()


# ── Visualisation des outliers (3 méthodes) ───────────────────
def plot_outliers(df: pd.DataFrame, outliers_iqr, outliers_z, lower_iqr: float, upper_iqr: float, col: str = "quantite") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # IQR
    axes[0].scatter(range(len(df)), df[col], color="steelblue", alpha=0.5, s=10, label="Normal")
    axes[0].scatter(outliers_iqr.index, outliers_iqr[col], color="red", s=30, label="Outlier", zorder=5)
    axes[0].axhline(upper_iqr, color="orange", linestyle="--", label=f"Borne sup {upper_iqr:.0f}")
    axes[0].axhline(lower_iqr, color="green",  linestyle="--", label=f"Borne inf {lower_iqr:.0f}")
    axes[0].set_title("Outliers — Méthode IQR")
    axes[0].legend(fontsize=8)

    # Z-Score
    axes[1].scatter(range(len(df)), df[col], color="steelblue", alpha=0.5, s=10, label="Normal")
    axes[1].scatter(outliers_z.index, outliers_z[col], color="red", s=30, label="Outlier", zorder=5)
    axes[1].set_title("Outliers — Méthode Z-Score")
    axes[1].legend(fontsize=8)

    # Isolation Forest (si colonne 'anomalie' présente)
    if "anomalie" in df.columns:
        colors_map = df["anomalie"].map({1: "steelblue", -1: "red"})
        axes[2].scatter(range(len(df)), df[col], c=colors_map, alpha=0.5, s=10)
        axes[2].set_title("Outliers — Isolation Forest")
        axes[2].legend(handles=[
            plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="steelblue", label="Normal"),
            plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="red",       label="Anomalie"),
        ])

    plt.suptitle("Détection des Outliers — 3 Méthodes", fontsize=14)
    plt.tight_layout()
    plt.show()


# ── Avant / après nettoyage ───────────────────────────────────
def plot_before_after_cleaning(df_original, df_clean, df_winsor, col: str = "quantite") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df_original[col], bins=50, color="red",       alpha=0.7, label="Avant")
    axes[0].hist(df_clean[col],    bins=50, color="steelblue", alpha=0.7, label="Après (IQR)")
    axes[0].set_title("Distribution Avant vs Après suppression")
    axes[0].legend()

    axes[1].hist(df_original[col], bins=50, color="red",   alpha=0.7, label="Avant")
    axes[1].hist(df_winsor[col],   bins=50, color="green", alpha=0.7, label="Après (Winsor)")
    axes[1].set_title("Distribution Avant vs Après Winsorisation")
    axes[1].legend()

    plt.suptitle("Impact du Traitement des Outliers", fontsize=14)
    plt.tight_layout()
    plt.show()


# ── Quantité par article ──────────────────────────────────────
def plot_quantity_by_article(df: pd.DataFrame) -> None:
    df.groupby("article")["quantite"].mean().sort_values(ascending=False).plot(
        kind="bar", figsize=(14, 6), color="steelblue"
    )
    plt.title("Quantité moyenne par article")
    plt.xlabel("Article")
    plt.ylabel("Quantité")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# ── Importance des variables ──────────────────────────────────
def plot_feature_importance(importances: pd.Series, title: str = "Importance des variables") -> None:
    plt.figure(figsize=(10, 5))
    importances.plot(kind="bar", color=["steelblue", "orange", "green"])
    plt.title(title)
    plt.ylabel("Score d'importance")
    plt.xlabel("Variables")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# ── Avant / après augmentation ────────────────────────────────
def plot_augmentation(df_original, df_augmented) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0,0].hist(df_original["quantite"],  bins=40, alpha=0.6, color="red",   label="Original")
    axes[0,0].hist(df_augmented["quantite"], bins=40, alpha=0.6, color="green", label="Augmenté")
    axes[0,0].set_title("Distribution Quantité")
    axes[0,0].legend()

    source_counts = df_augmented["source"].value_counts()
    axes[0,1].bar(source_counts.index, source_counts.values,
                  color=["steelblue","orange","green","red","purple"])
    axes[0,1].set_title("Lignes par méthode d'augmentation")
    axes[0,1].tick_params(axis="x", rotation=20)

    df_original["mois"].value_counts().sort_index().plot(kind="bar", ax=axes[1,0], color="red",   alpha=0.6, label="Original")
    df_augmented["mois"].value_counts().sort_index().plot(kind="bar", ax=axes[1,0], color="green", alpha=0.6, label="Augmenté")
    axes[1,0].set_title("Distribution par Mois")
    axes[1,0].legend()

    df_original["article"].value_counts().head(10).plot(kind="barh", ax=axes[1,1], color="red",   alpha=0.6, label="Original")
    df_augmented["article"].value_counts().head(10).plot(kind="barh", ax=axes[1,1], color="green", alpha=0.6, label="Augmenté")
    axes[1,1].set_title("Top 10 Articles")
    axes[1,1].legend()

    plt.suptitle("Data Augmentation — Avant vs Après", fontsize=14)
    plt.tight_layout()
    plt.show()


# ── CA Global prévu par mois ──────────────────────────────────
def plot_ca_global(ca_mensuel: pd.DataFrame) -> None:
    """Affiche le CA prévu mois par mois sous forme de courbe + barres."""
    fig, ax1 = plt.subplots(figsize=(14, 6))

    labels = [f"{int(r.annee)}-{int(r.mois):02d}" for _, r in ca_mensuel.iterrows()]

    ax1.bar(labels, ca_mensuel["ca_prevu_total"], color="steelblue", alpha=0.6, label="CA prévu (DT)")
    ax1.set_ylabel("CA Prévu (DT)", color="steelblue")
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(labels, ca_mensuel["quantite_prevue"], color="orange", marker="o", linewidth=2, label="Quantité prévue")
    ax2.set_ylabel("Quantité Prévue", color="orange")

    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.95))
    plt.title("Prévision CA Global & Quantités — par Mois")
    plt.tight_layout()
    plt.show()

def plot_model_comparison(resultats: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    noms     = list(resultats.keys())
    r2_vals  = [resultats[n].get("R²", resultats[n].get("R2", 0))  for n in noms]
    mae_vals = [resultats[n]["MAE"] for n in noms]
    colors   = ["lightblue"] * (len(noms) - 1) + ["steelblue"]

    ax1.bar(noms, r2_vals, color=colors)
    ax1.set_title("R² par modèle")
    ax1.set_ylabel("R²")
    ax1.set_ylim(0, 1)
    ax1.axhline(0.7, color="red", linestyle="--", label="Objectif 0.7")
    ax1.set_xticklabels(noms, rotation=20, ha="right")
    ax1.legend()

    ax2.bar(noms, mae_vals, color=colors)
    ax2.set_title("MAE par modèle")
    ax2.set_ylabel("MAE")
    ax2.set_xticklabels(noms, rotation=20, ha="right")

    plt.suptitle("Comparaison modèles — avec Preprocessing", fontsize=14)
    plt.tight_layout()
    plt.show()
