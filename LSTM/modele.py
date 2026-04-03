# ============================================================
# lstm_model.py — LSTM autonome avec ses propres features
# ============================================================
#
# Ce fichier est INDÉPENDANT du reste du projet.
# Il charge les données, construit ses propres features
# (Ramadan, Aïd, saisonnalité) et entraîne le LSTM.
#
# Lancement : python lstm_model.py
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics       import mean_absolute_error, mean_squared_error, r2_score

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("PyTorch non installe : pip install torch")


# ════════════════════════════════════════════════════════════
# HYPERPARAMÈTRES — modifier ici pour tuning
# ════════════════════════════════════════════════════════════

HP = {
    # Séquence — réduit car 460 jours utiles seulement
    "n_steps"       : 14,      # 14 jours (était 30 — trop long vs données)
    "n_test"        : 45,      # 45 jours de test (était 60)

    # Architecture — simplifiée pour dataset court
    "hidden_size"   : 32,      # réduit (était 64 — overfitting sur 460 jours)
    "num_layers"    : 1,       # 1 seule couche (était 2 — trop complexe)
    "bidirectional" : False,   # désactivé (était True — trop de paramètres)
    "dropout"       : 0.1,     # réduit (était 0.2)

    # Entraînement — lr plus faible pour série volatile
    "epochs"        : 150,     # plus d epochs pour compenser la simplicité
    "batch_size"    : 16,      # batch plus petit (était 32)
    "learning_rate" : 0.0005,  # réduit (était 0.001 — oscillait)
    "lr_step"       : 50,      # réduire lr tous les 50 époques
    "lr_gamma"      : 0.5,
}


# ════════════════════════════════════════════════════════════
# ÉTAPE 1 — CHARGEMENT
# ════════════════════════════════════════════════════════════

def charger_donnees() -> pd.DataFrame:
    """Charge les données via data_extraction.py existant."""
    import sys
    sys.path.append(r"C:\Users\dhieb\Desktop\old_Ventes\ML_VENTE")
    from data_extraction import load_data
    print("=" * 55)
    print("ETAPE 1 — Chargement des données")
    print("=" * 55)
    df = load_data()
    df["date_facture"] = pd.to_datetime(df["date_facture"])
    print(f"  {len(df)} lignes | {df['article'].nunique()} articles | "
          f"{df['date_facture'].min().strftime('%d-%m-%Y')} → "
          f"{df['date_facture'].max().strftime('%d-%m-%Y')}")
    return df


# ════════════════════════════════════════════════════════════
# ÉTAPE 2 — FEATURES SPÉCIFIQUES LSTM
# ════════════════════════════════════════════════════════════

# Périodes Ramadan / Aïd Tunisie
PERIODES = {
    "ramadan" : [
        ("2021-04-13", "2021-05-12"),
        ("2022-04-02", "2022-05-01"),
        ("2023-03-23", "2023-04-20"),
        ("2024-03-11", "2024-04-09"),
    ],
    "aid_fitr" : [
        ("2021-05-13", "2021-05-15"),
        ("2022-05-02", "2022-05-04"),
        ("2023-04-21", "2023-04-23"),
        ("2024-04-10", "2024-04-12"),
    ],
    "aid_adha" : [
        ("2021-07-20", "2021-07-22"),
        ("2022-07-09", "2022-07-11"),
        ("2023-06-28", "2023-06-30"),
        ("2024-06-17", "2024-06-19"),
    ],
    "ete" : [
        ("2021-07-01", "2021-08-31"),
        ("2022-07-01", "2022-08-31"),
        ("2023-07-01", "2023-08-31"),
        ("2024-07-01", "2024-08-31"),
    ],
}


def est_dans_periode(date: pd.Timestamp, periodes: list) -> int:
    for debut, fin in periodes:
        if pd.Timestamp(debut) <= date <= pd.Timestamp(fin):
            return 1
    return 0


def construire_features_lstm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prétraitement spécifique LSTM :

      1. Agrégation par jour (quantité totale)
         LSTM travaille sur une seule série continue,
         pas par article comme XGBoost.

      2. Remplissage des jours manquants avec 0
         LSTM a besoin d'une série continue sans trous.

      3. Ajout des périodes spéciales (Ramadan, Aïd, Été)
         Ces événements créent des patterns forts et répétitifs
         que LSTM doit apprendre à anticiper.

      4. Winsorisation 99e percentile sur jours normaux
         Limite les pics extrêmes sans toucher Ramadan/Aïd.

      5. Features temporelles cycliques sin/cos
         LSTM comprend mieux le cycle mensuel et hebdomadaire
         sous forme de sin/cos que comme entier brut.
    """
    print("\n" + "=" * 55)
    print("ETAPE 2 — Features spécifiques LSTM")
    print("=" * 55)

    df = df.copy()
    df["date_facture"] = pd.to_datetime(df["date_facture"])

    # Agréger par jour
    agg_cols = {"quantite": "sum"}
    if "discount" in df.columns:
        agg_cols["discount"]   = "sum"
    if "montant_ht" in df.columns:
        agg_cols["montant_ht"] = "sum"

    df_agg = df.groupby("date_facture").agg(agg_cols).sort_index()

    # Remplir jours manquants
    idx    = pd.date_range(df_agg.index.min(), df_agg.index.max(), freq="D")
    df_agg = df_agg.reindex(idx, fill_value=0)
    df_agg.index.name = "date_facture"
    print(f"  Jours totaux après remplissage : {len(df_agg)}")

    # Périodes spéciales
    df_agg["est_ramadan"]  = [est_dans_periode(d, PERIODES["ramadan"])  for d in df_agg.index]
    df_agg["est_aid_fitr"] = [est_dans_periode(d, PERIODES["aid_fitr"]) for d in df_agg.index]
    df_agg["est_aid_adha"] = [est_dans_periode(d, PERIODES["aid_adha"]) for d in df_agg.index]
    df_agg["est_ete"]      = [est_dans_periode(d, PERIODES["ete"])      for d in df_agg.index]

    print(f"  Jours Ramadan  : {df_agg['est_ramadan'].sum()}")
    print(f"  Jours Aid Fitr : {df_agg['est_aid_fitr'].sum()}")
    print(f"  Jours Ete      : {df_agg['est_ete'].sum()}")

    # Features temporelles cycliques (sin/cos)
    # sin/cos permettent à LSTM de comprendre que mois 12 → mois 1
    df_agg["mois_sin"] = np.sin(2 * np.pi * df_agg.index.month      / 12)
    df_agg["mois_cos"] = np.cos(2 * np.pi * df_agg.index.month      / 12)
    df_agg["jour_sin"] = np.sin(2 * np.pi * df_agg.index.dayofweek  / 7)
    df_agg["jour_cos"] = np.cos(2 * np.pi * df_agg.index.dayofweek  / 7)

    # Remise en % si disponible
    if "discount" in df_agg.columns and "montant_ht" in df_agg.columns:
        prix_avant         = df_agg["montant_ht"] + df_agg["discount"].abs()
        df_agg["remise_pct"] = (df_agg["discount"].abs() / prix_avant.replace(0, np.nan) * 100).fillna(0)
    else:
        df_agg["remise_pct"] = 0

    # Winsorisation 99e percentile (protéger Ramadan/Aïd)
    masque_normal = (df_agg["est_ramadan"] == 0) & (df_agg["est_aid_fitr"] == 0)
    p99 = df_agg.loc[masque_normal, "quantite"].quantile(0.99)
    avant = (df_agg.loc[masque_normal, "quantite"] > p99).sum()
    df_agg.loc[masque_normal & (df_agg["quantite"] > p99), "quantite"] = p99
    print(f"  Winsorisation  : {avant} pics limités à {p99:.0f}")

    return df_agg


# ════════════════════════════════════════════════════════════
# ÉTAPE 3 — PRÉPARATION SÉQUENCES
# ════════════════════════════════════════════════════════════

FEAT_LSTM = [
    "quantite",      # ⭐⭐⭐ signal principal
    "est_ramadan",   # ⭐⭐⭐ pic fort annuel
    "est_aid_fitr",  # ⭐⭐⭐ pic brutal
    "est_aid_adha",  # ⭐⭐
    "est_ete",       # ⭐⭐  baisse estivale
    "mois_sin",      # ⭐⭐  saisonnalité cyclique
    "mois_cos",      # ⭐⭐
    "jour_sin",      # ⭐   saisonnalité hebdo
    "jour_cos",      # ⭐
    "remise_pct",    # ⭐   choc externe
]


def preparer_sequences(df_agg: pd.DataFrame) -> tuple:
    print("\n" + "=" * 55)
    print("ETAPE 3 — Préparation des séquences")
    print("=" * 55)

    feat_dispo = [f for f in FEAT_LSTM if f in df_agg.columns]
    manquantes = [f for f in FEAT_LSTM if f not in df_agg.columns]
    if manquantes:
        print(f"  Features absentes (mises à 0) : {manquantes}")
        for f in manquantes:
            df_agg[f] = 0

    data      = df_agg[feat_dispo].values.astype(float)
    dates     = df_agg.index
    n_features = len(feat_dispo)

    # Normalisation MinMaxScaler [0,1] sur toutes les features
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_n = scaler.fit_transform(data)

    # Scaler séparé pour quantite seulement (dénormalisation correcte)
    scaler_qte = MinMaxScaler(feature_range=(0, 1))
    scaler_qte.fit(data[:, 0].reshape(-1, 1))

    # Séquences glissantes
    X_list, y_list = [], []
    for i in range(HP["n_steps"], len(data_n)):
        X_list.append(data_n[i - HP["n_steps"]:i])
        y_list.append(data_n[i, 0])   # quantite normalisée

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # Split train/test
    n_test  = HP["n_test"]
    split   = len(X) - n_test
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dates_test      = dates[HP["n_steps"] + split:]

    print(f"  Features LSTM  : {feat_dispo}")
    print(f"  Séquences train: {len(X_train)} | test: {len(X_test)}")
    print(f"  Fenêtre entrée : {HP['n_steps']} jours")

    return (X_train, X_test, y_train, y_test,
            scaler, scaler_qte, n_features, feat_dispo, dates_test)


# ════════════════════════════════════════════════════════════
# ÉTAPE 4 — ARCHITECTURE LSTM
# ════════════════════════════════════════════════════════════

class LSTMModel(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        hidden = HP["hidden_size"]
        n_dir  = 2 if HP["bidirectional"] else 1

        self.lstm = nn.LSTM(
            input_size    = n_features,
            hidden_size   = hidden,
            num_layers    = HP["num_layers"],
            batch_first   = True,
            bidirectional = HP["bidirectional"],
            dropout       = HP["dropout"] if HP["num_layers"] > 1 else 0,
        )
        self.dropout = nn.Dropout(HP["dropout"])
        self.fc      = nn.Linear(hidden * n_dir, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.dropout(out[:, -1, :])
        return self.fc(out)


# ════════════════════════════════════════════════════════════
# ÉTAPE 5 — ENTRAÎNEMENT
# ════════════════════════════════════════════════════════════

def entrainer(X_train, y_train, n_features: int) -> tuple:
    print("\n" + "=" * 55)
    print("ETAPE 4 — Entraînement LSTM")
    print(f"  hidden={HP['hidden_size']} | layers={HP['num_layers']} | "
          f"bi={HP['bidirectional']} | dropout={HP['dropout']}")
    print(f"  epochs={HP['epochs']} | lr={HP['learning_rate']} | "
          f"batch={HP['batch_size']}")
    print("=" * 55)

    ds = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).unsqueeze(-1)
    )
    dl = DataLoader(ds, batch_size=HP["batch_size"], shuffle=False)

    modele    = LSTMModel(n_features)
    optimizer = torch.optim.Adam(modele.parameters(), lr=HP["learning_rate"])
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=HP["lr_step"], gamma=HP["lr_gamma"]
    )

    historique = []
    for epoch in range(HP["epochs"]):
        modele.train()
        loss_ep = 0
        for xb, yb in dl:
            optimizer.zero_grad()
            loss = criterion(modele(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(modele.parameters(), max_norm=1.0)
            optimizer.step()
            loss_ep += loss.item()
        scheduler.step()
        historique.append(loss_ep / len(dl))
        if (epoch + 1) % 20 == 0:
            print(f"  Epoque {epoch+1:3d}/{HP['epochs']} | "
                  f"Loss: {historique[-1]:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

    return modele, historique


# ════════════════════════════════════════════════════════════
# ÉTAPE 6 — ÉVALUATION
# ════════════════════════════════════════════════════════════

def evaluer(modele, X_test, y_test, scaler, n_features) -> dict:
    print("\n" + "=" * 55)
    print("ETAPE 5 — Évaluation")
    print("=" * 55)

    modele.eval()
    with torch.no_grad():
        preds_n = modele(torch.FloatTensor(X_test)).numpy().flatten()

    def denorm(vals):
        tmp = np.zeros((len(vals), n_features))
        tmp[:, 0] = vals
        return scaler.inverse_transform(tmp)[:, 0].clip(min=0)

    y_pred = denorm(preds_n)
    y_true = denorm(y_test)

    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # MAPE uniquement sur jours avec ventes réelles > 0
    masque = y_true > 0
    mape   = np.mean(np.abs((y_true[masque] - y_pred[masque]) / y_true[masque])) * 100

    print(f"  R²   : {r2:.4f}  {'✅' if r2 >= 0.80 else '⚠️'}")
    print(f"  MAE  : {mae:.1f}")
    print(f"  RMSE : {rmse:.1f}")
    print(f"  MAPE : {mape:.1f}%")

    return {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape,
            "y_true": y_true, "y_pred": y_pred}


# ════════════════════════════════════════════════════════════
# GRAPHIQUES
# ════════════════════════════════════════════════════════════

def plot_resultats(metriques: dict, historique: list, dates_test) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"LSTM — R²={metriques['R2']:.4f} | "
        f"MAE={metriques['MAE']:.1f} | MAPE={metriques['MAPE']:.1f}%",
        fontsize=12, fontweight="bold"
    )

    # Loss
    axes[0].plot(historique, color="#3498db")
    axes[0].set_title("Courbe de Loss")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("MSE Loss")

    # Réel vs Prédit
    axes[1].plot(metriques["y_true"], label="Réel",   color="#2c3e50", linewidth=1.2)
    axes[1].plot(metriques["y_pred"], label="Prédit", color="#e74c3c",
                 linewidth=1.2, linestyle="--")
    axes[1].set_title("Réel vs Prédit (période test)")
    axes[1].set_ylabel("Quantité totale / jour")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("lstm_resultat.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Graphique : lstm_resultat.png")


# ════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ════════════════════════════════════════════════════════════

def main():
    if not TORCH_OK:
        print("PyTorch requis : pip install torch")
        return

    print("\n" + "█" * 55)
    print("  LSTM — Pipeline autonome")
    print("█" * 55)

    df         = charger_donnees()
    df_agg     = construire_features_lstm(df)
    (X_train, X_test, y_train, y_test,
     scaler, scaler_qte, n_features, feat_dispo, dates_test) = preparer_sequences(df_agg)

    modele, historique = entrainer(X_train, y_train, n_features)
    metriques          = evaluer(modele, X_test, y_test, scaler_qte, n_features)
    plot_resultats(metriques, historique, dates_test)

    print("\n" + "=" * 55)
    print("  LSTM terminé.")
    print("=" * 55)

    return modele, scaler, metriques


if __name__ == "__main__":
    main()