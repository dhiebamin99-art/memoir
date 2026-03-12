# ============================================================
# preprocessing.py — Nettoyage et détection/traitement des outliers
# ============================================================

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from config import IQR_LOWER_FACTOR, IQR_UPPER_FACTOR, RANDOM_STATE


# ── Statistiques descriptives ────────────────────────────────
def describe_data(df: pd.DataFrame, col: str = "quantite") -> None:
    print("=" * 55)
    print("🔍 STATISTIQUES DESCRIPTIVES")
    print("=" * 55)
    print(df[col].describe())


# ── Méthode IQR ──────────────────────────────────────────────
def detect_outliers_iqr(df: pd.DataFrame, col: str = "quantite") -> tuple:
    """Retourne (df_outliers, lower_bound, upper_bound)."""
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - IQR_LOWER_FACTOR * IQR
    upper = Q3 + IQR_UPPER_FACTOR * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"\n📌 IQR → borne inf: {lower:.2f} | borne sup: {upper:.2f} | outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    return outliers, lower, upper


# ── Méthode Z-Score ──────────────────────────────────────────
def detect_outliers_zscore(df: pd.DataFrame, col: str = "quantite", threshold: float = 3.0) -> pd.DataFrame:
    z_scores = np.abs(stats.zscore(df[col]))
    outliers = df[z_scores > threshold]
    print(f"📌 Z-Score (seuil={threshold}) → outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    return outliers


# ── Méthode Isolation Forest ─────────────────────────────────
def detect_outliers_isolation_forest(df: pd.DataFrame, col: str = "quantite", contamination: float = 0.05) -> pd.DataFrame:
    df = df.copy()
    iso = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
    df["anomalie"] = iso.fit_predict(df[[col]])
    outliers = df[df["anomalie"] == -1]
    print(f"📌 Isolation Forest (contam={contamination}) → outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    return df, outliers


# ── Suppression des outliers (IQR) ───────────────────────────
def remove_outliers_iqr(df: pd.DataFrame, lower: float, upper: float, col: str = "quantite") -> pd.DataFrame:
    df_clean = df[(df[col] >= lower) & (df[col] <= upper)].copy()
    print(f"\n🧹 Suppression IQR : {len(df)} → {len(df_clean)} lignes (supprimées : {len(df) - len(df_clean)})")
    return df_clean


# ── Winsorisation ─────────────────────────────────────────────
def winsorize(df: pd.DataFrame, lower: float, upper: float, col: str = "quantite") -> pd.DataFrame:
    df_w = df.copy()
    df_w[col] = df_w[col].clip(lower=lower, upper=upper)
    print(f"🧹 Winsorisation : valeurs cappées entre {lower:.0f} et {upper:.0f}")
    return df_w


# ── Pipeline complet ──────────────────────────────────────────
def preprocess(df: pd.DataFrame, method: str = "iqr") -> pd.DataFrame:
    """
    Lance l'analyse complète et retourne le DataFrame nettoyé.
    method : 'iqr' (suppression) | 'winsor' (winsorisation)
    """
    print("\n" + "=" * 55)
    print("🚨 DÉTECTION DES OUTLIERS")
    print("=" * 55)

    describe_data(df)

    _, lower, upper = detect_outliers_iqr(df)
    detect_outliers_zscore(df)
    detect_outliers_isolation_forest(df)

    if method == "winsor":
        return winsorize(df, lower, upper)
    else:
        return remove_outliers_iqr(df, lower, upper)


if __name__ == "__main__":
    # Test rapide avec données fictives
    import pandas as pd, numpy as np
    df_test = pd.DataFrame({"quantite": np.random.exponential(scale=50, size=1000)})
    df_clean = preprocess(df_test, method="iqr")
    print(df_clean["quantite"].describe())
