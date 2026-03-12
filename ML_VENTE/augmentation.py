                                                # ============================================================
# augmentation.py — Data Augmentation (4 méthodes)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from config import AUGMENTATION_CONFIG, RANDOM_STATE

COLS_NUMERIQUES = ["quantite", "qte_moy_article", "qte_moy_mois", "qte_lag_1", "qte_lag_7"]


# ── Méthode 1 : Gaussian Noise ───────────────────────────────
def add_gaussian_noise(df: pd.DataFrame, cols: list = COLS_NUMERIQUES, noise_factor: float = None) -> pd.DataFrame:
    if noise_factor is None:
        noise_factor = AUGMENTATION_CONFIG["gaussian_noise_factor"]
    df_noisy = df.copy()
    for col in cols:
        if col in df_noisy.columns:
            std   = df[col].std()
            noise = np.random.normal(0, noise_factor * std, size=len(df))
            df_noisy[col] = (df[col] + noise).clip(lower=0)
    df_noisy["source"] = "gaussian_noise"
    print(f"✅ Gaussian Noise : +{len(df_noisy)} lignes")
    return df_noisy


# ── Méthode 2 : Interpolation temporelle ─────────────────────
def interpolate_temporal(df: pd.DataFrame, n_samples: int = None) -> pd.DataFrame:
    if n_samples is None:
        n_samples = AUGMENTATION_CONFIG["interpolation_samples"]
    df_sorted = df.sort_values("date_facture").reset_index(drop=True)
    new_rows  = []

    for _ in range(n_samples):
        idx   = np.random.randint(0, len(df_sorted) - 1)
        row1  = df_sorted.iloc[idx]
        row2  = df_sorted.iloc[idx + 1]
        alpha = np.random.uniform(0.2, 0.8)

        new_row = row1.copy()
        for col in COLS_NUMERIQUES:
            if col in df_sorted.columns:
                new_row[col] = alpha * row1[col] + (1 - alpha) * row2[col]

        delta    = (row2["date_facture"] - row1["date_facture"]).days
        new_date = row1["date_facture"] + pd.Timedelta(days=int(alpha * delta))
        new_row["date_facture"] = new_date
        new_row["mois"]         = new_date.month
        new_row["trimestre"]    = (new_date.month - 1) // 3 + 1
        new_row["annee"]        = new_date.year
        new_row["jour_semaine"] = new_date.dayofweek
        new_row["est_fin_mois"] = int(new_date.is_month_end)
        new_rows.append(new_row)

    df_interp          = pd.DataFrame(new_rows)
    df_interp["source"] = "interpolation"
    print(f"✅ Interpolation temporelle : +{len(df_interp)} lignes")
    return df_interp


# ── Méthode 3 : Seasonal Jitter ──────────────────────────────
def seasonal_jitter(df: pd.DataFrame, n_samples: int = None) -> pd.DataFrame:
    if n_samples is None:
        n_samples = AUGMENTATION_CONFIG["seasonal_samples"]
    new_rows = []

    for _ in range(n_samples):
        row      = df.sample(1).iloc[0].copy()
        delta    = np.random.randint(-30, 30)
        new_date = row["date_facture"] + pd.Timedelta(days=delta)

        row["date_facture"] = new_date
        row["mois"]         = new_date.month
        row["trimestre"]    = (new_date.month - 1) // 3 + 1
        row["annee"]        = new_date.year
        row["jour_semaine"] = new_date.dayofweek
        row["est_fin_mois"] = int(new_date.is_month_end)
        row["date_enc"]     = int(new_date.timestamp())
        row["quantite"]     = max(0, row["quantite"] * np.random.uniform(0.95, 1.05))
        new_rows.append(row)

    df_seasonal          = pd.DataFrame(new_rows)
    df_seasonal["source"] = "seasonal_jitter"
    print(f"✅ Seasonal Jitter : +{len(df_seasonal)} lignes")
    return df_seasonal


# ── Méthode 4 : SMOGN-like (valeurs rares) ───────────────────
def smogn_like(df: pd.DataFrame, n_samples: int = None) -> pd.DataFrame:
    if n_samples is None:
        n_samples = AUGMENTATION_CONFIG["smogn_samples"]
    q10     = df["quantite"].quantile(0.10)
    q90     = df["quantite"].quantile(0.90)
    df_rare = df[(df["quantite"] <= q10) | (df["quantite"] >= q90)]
    new_rows = []

    for _ in range(n_samples):
        if len(df_rare) < 2:
            break
        row1, row2 = df_rare.sample(2).iloc[0], df_rare.sample(2).iloc[1]
        alpha = np.random.uniform(0.3, 0.7)
        new_row = row1.copy()
        for col in COLS_NUMERIQUES:
            if col in df_rare.columns:
                new_row[col] = max(0, alpha * row1[col] + (1 - alpha) * row2[col])
        new_rows.append(new_row)

    df_smogn          = pd.DataFrame(new_rows)
    df_smogn["source"] = "smogn_rare"
    print(f"✅ SMOGN Rare Values : +{len(df_smogn)} lignes")
    return df_smogn


# ── Pipeline complet ──────────────────────────────────────────
def augment_data(df: pd.DataFrame) -> pd.DataFrame:
    """Combine les 4 méthodes et retourne le dataset augmenté."""
    print("\n" + "=" * 55)
    print("📊 DATA AUGMENTATION")
    print("=" * 55)
    n_original = len(df)
    df["source"] = "original"

    df_augmented = pd.concat([
        df,
        add_gaussian_noise(df),
        interpolate_temporal(df),
        seasonal_jitter(df),
        smogn_like(df),
    ], ignore_index=True)

    # Re-encoder après augmentation
    le = LabelEncoder()
    df_augmented["site_enc"]    = le.fit_transform(df_augmented["site"].astype(str))
    df_augmented["article_enc"] = le.fit_transform(df_augmented["article"].astype(str))

    df_augmented = df_augmented.dropna(subset=["qte_lag_1", "qte_lag_7", "qte_moy_article"])
    df_augmented = df_augmented.drop_duplicates().reset_index(drop=True)

    print(f"\n{'='*50}")
    print(f"📊 Original  : {n_original} lignes")
    print(f"📊 Augmenté  : {len(df_augmented)} lignes")
    print(f"📈 +{len(df_augmented)-n_original} lignes ({(len(df_augmented)/n_original - 1)*100:.1f}%)")
    print(f"{'='*50}")
    return df_augmented


if __name__ == "__main__":
    print("Lancez ce module via main.py")
