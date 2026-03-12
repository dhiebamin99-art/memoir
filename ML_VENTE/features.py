# ============================================================
# features.py — Feature engineering (variables temporelles, lags, encodages)
# ============================================================

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def add_date_features(df: pd.DataFrame, date_col: str = "date_facture") -> pd.DataFrame:
    """Ajoute les variables temporelles dérivées de la date."""
    df = df.copy()
    df[date_col]       = pd.to_datetime(df[date_col])
    df["mois"]         = df[date_col].dt.month
    df["trimestre"]    = df[date_col].dt.quarter
    df["annee"]        = df[date_col].dt.year
    df["jour"]         = df[date_col].dt.day
    df["jour_semaine"] = df[date_col].dt.dayofweek
    df["est_fin_mois"] = df[date_col].dt.is_month_end.astype(int)
    print("✅ Variables temporelles ajoutées : mois, trimestre, annee, jour, jour_semaine, est_fin_mois")
    return df


def add_statistical_features(df: pd.DataFrame, group_col: str = "article_enc", target_col: str = "quantite") -> pd.DataFrame:
    """Ajoute les moyennes agrégées par article et par mois."""
    df = df.copy()
    df["qte_moy_article"] = df.groupby(group_col)[target_col].transform("mean")
    df["qte_moy_mois"]    = df.groupby([group_col, "mois"])[target_col].transform("mean")
    print("✅ Features statistiques ajoutées : qte_moy_article, qte_moy_mois")
    return df


def add_lag_features(df: pd.DataFrame, group_col: str = "article_enc", target_col: str = "quantite") -> pd.DataFrame:
    """Ajoute les features de lag (J-1, J-7) triées par date."""
    df = df.sort_values("date_facture").copy()
    df["qte_lag_1"] = df.groupby(group_col)[target_col].shift(1)
    df["qte_lag_7"] = df.groupby(group_col)[target_col].shift(7)
    print("✅ Lag features ajoutées : qte_lag_1, qte_lag_7")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode les colonnes catégorielles (site, article, date)."""
    df = df.copy()
    le = LabelEncoder()
    df["site_enc"]    = le.fit_transform(df["site"].astype(str))
    df["article_enc"] = le.fit_transform(df["article"].astype(str))
    df["date_enc"]    = le.fit_transform(df["date_facture"].astype(str))
    print("✅ Encodage LabelEncoder : site_enc, article_enc, date_enc")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline complet de feature engineering."""
    print("\n" + "=" * 55)
    print("⚙️  FEATURE ENGINEERING")
    print("=" * 55)
    df = add_date_features(df)
    df = encode_categoricals(df)
    df = add_statistical_features(df)
    df = add_lag_features(df)
    df = df.dropna()
    print(f"✅ Dataset final après feature engineering : {df.shape}")
    return df


if __name__ == "__main__":
    import numpy as np
    dates = pd.date_range("2022-01-01", periods=500, freq="D")
    df_test = pd.DataFrame({
        "date_facture": np.random.choice(dates, 500),
        "site"        : np.random.choice(["SiteA", "SiteB"], 500),
        "article"     : np.random.choice(["ART001", "ART002", "ART003"], 500),
        "quantite"    : np.random.exponential(50, 500),
    })
    df_feat = build_features(df_test)
    print(df_feat.head())
