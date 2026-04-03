# ============================================================
# modeling.py — Entraînement et évaluation des modèles
# Modèles : Linear Regression, Random Forest, Gradient Boosting,
#            XGBoost, Stacking, ARIMA, Prophet, LSTM
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from config import FEATURES, TARGET, RANDOM_STATE, TEST_SIZE, N_ESTIMATORS, MAX_DEPTH_TREE


# ════════════════════════════════════════════════════════════
# UTILITAIRES COMMUNS
# ════════════════════════════════════════════════════════════

def prepare_xy(df):
    available = [f for f in FEATURES if f in df.columns]
    return df[available], df[TARGET]


def build_preprocessor(X):
    num_cols = X.select_dtypes(include=["int64","float64","int32"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()
    print(f"Numeriques  : {num_cols}")
    print(f"Categorielles : {cat_cols}")
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(),                       num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])


def evaluate_model(y_test, y_pred, nom=""):
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"{nom:25s} -> R²: {r2:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")
    return {"R²": r2, "MAE": mae, "RMSE": rmse}


def feature_importance(df):
    X, y     = prepare_xy(df)
    num_cols = X.select_dtypes(include=["int64","float64","int32","bool"]).columns.tolist()
    X_num    = df[[c for c in num_cols if c in FEATURES]]
    dt       = DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=MAX_DEPTH_TREE)
    dt.fit(X_num, y)
    importances = pd.Series(dt.feature_importances_, index=X_num.columns).sort_values(ascending=False)
    print("\nImportance des variables (DecisionTree) :")
    print(importances)
    return importances


# ════════════════════════════════════════════════════════════
# MODELES ML CLASSIQUES
# ════════════════════════════════════════════════════════════

def build_stacking_model(preprocessor):
    estimators = [
        ("rf",  Pipeline([("pre", preprocessor), ("model", RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE))])),
        ("gbm", Pipeline([("pre", preprocessor), ("model", GradientBoostingRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE))])),
        ("xgb", Pipeline([("pre", preprocessor), ("model", XGBRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, verbosity=0))])),
    ]
    meta = Pipeline([("pre", StandardScaler()), ("model", Ridge())])
    return StackingRegressor(estimators=estimators, final_estimator=meta, cv=5, passthrough=False)


def train_and_evaluate(df):
    print("\n" + "="*55)
    print("MODELES ML CLASSIQUES")
    print("="*55)

    X, y         = prepare_xy(df)
    preprocessor = build_preprocessor(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    stacking_model = build_stacking_model(preprocessor)
    stacking_model.fit(X_train, y_train)
    y_pred_stack = stacking_model.predict(X_test)

    modeles = {
        "Linear Regression" : Pipeline([("pre", preprocessor), ("model", LinearRegression())]),
        "Random Forest"     : Pipeline([("pre", preprocessor), ("model", RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE))]),
        "Gradient Boosting" : Pipeline([("pre", preprocessor), ("model", GradientBoostingRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE))]),
        "XGBoost"           : Pipeline([("pre", preprocessor), ("model", XGBRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, verbosity=0))]),
        "Stacking"          : stacking_model,
    }

    print(f"\n{'Modele':25s} -> R2       | MAE    | RMSE")
    print("-"*58)
    resultats = {}
    for nom, modele in modeles.items():
        if nom != "Stacking":
            modele.fit(X_train, y_train)
            y_pred = modele.predict(X_test)
        else:
            y_pred = y_pred_stack
        resultats[nom] = evaluate_model(y_test, y_pred, nom)

    return stacking_model, resultats


# ════════════════════════════════════════════════════════════
# ARIMA
# ════════════════════════════════════════════════════════════

def train_arima(df, article=None, n_test=30):
    """
    ARIMA(p,d,q) sur la serie temporelle des quantites.
    - article : filtrer un article specifique (None = tous)
    - n_test  : nombre de jours du jeu de test
    Installation : pip install statsmodels
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        print("statsmodels non installe. Lancez : pip install statsmodels")
        return {}

    print("\n" + "="*55)
    print("ARIMA — Serie Temporelle")
    print("="*55)

    df_s = df[df["article"] == article].copy() if article else df.copy()
    serie = (
        df_s.groupby("date_facture")["quantite"]
        .sum().asfreq("D").fillna(0).sort_index()
    )

    # Test stationnarite ADF
    adf_p = adfuller(serie)[1]
    d = 0 if adf_p < 0.05 else 1
    print(f"ADF p-value : {adf_p:.4f} -> d = {d}")

    train_s, test_s = serie.iloc[:-n_test], serie.iloc[-n_test:]

    result = ARIMA(train_s, order=(5, d, 2)).fit()
    forecast = np.maximum(result.forecast(steps=n_test).values, 0)

    metriques = evaluate_model(test_s.values, forecast, "ARIMA")
    print(f"AIC : {result.aic:.2f}")

    return {"modele": result, "train": train_s, "test": test_s,
            "forecast": forecast, "metriques": metriques, "serie": serie}


def forecast_arima(arima_result, n_mois=12):
    """Prevision future avec ARIMA (n_mois * 30 jours)."""
    n_jours  = n_mois * 30
    forecast = np.maximum(arima_result["modele"].forecast(steps=n_jours).values, 0)
    print(f"ARIMA : prevision sur {n_jours} jours ({n_mois} mois)")
    return forecast


# ════════════════════════════════════════════════════════════
# PROPHET
# ════════════════════════════════════════════════════════════

def train_prophet(df, article=None, n_test=30):
    """
    Prophet (Meta) sur la serie temporelle des quantites.
    Gere automatiquement la saisonnalite annuelle et hebdomadaire.
    - article : filtrer un article specifique (None = tous)
    - n_test  : nombre de jours du jeu de test
    Installation : pip install prophet
    """
    try:
        from prophet import Prophet
    except ImportError:
        print("Prophet non installe. Lancez : pip install prophet")
        return {}

    print("\n" + "="*55)
    print("PROPHET — Series Temporelles (Meta)")
    print("="*55)

    df_s = df[df["article"] == article].copy() if article else df.copy()
    df_p = (
        df_s.groupby("date_facture")["quantite"].sum()
        .reset_index()
        .rename(columns={"date_facture": "ds", "quantite": "y"})
        .sort_values("ds")
    )
    df_p["y"] = df_p["y"].clip(lower=0)

    train_df, test_df = df_p.iloc[:-n_test], df_p.iloc[-n_test:]

    model_p = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
    )
    model_p.fit(train_df)

    future   = model_p.make_future_dataframe(periods=n_test)
    forecast = model_p.predict(future)
    y_pred   = np.maximum(forecast["yhat"].tail(n_test).values, 0)

    metriques = evaluate_model(test_df["y"].values, y_pred, "Prophet")

    return {"modele": model_p, "train": train_df, "test": test_df,
            "forecast": forecast, "metriques": metriques}


def forecast_prophet(prophet_result, n_mois=12):
    """Prevision future avec Prophet."""
    n_jours  = n_mois * 30
    model    = prophet_result["modele"]
    future   = model.make_future_dataframe(periods=n_jours)
    forecast = model.predict(future)
    forecast["yhat"] = forecast["yhat"].clip(lower=0)
    futurs   = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(n_jours)
    print(f"Prophet : prevision sur {n_jours} jours ({n_mois} mois)")
    return futurs


# ════════════════════════════════════════════════════════════
# LSTM
# ════════════════════════════════════════════════════════════

def create_sequences(data, n_steps=30):
    """Transforme une serie en sequences (X, y) pour LSTM."""
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i - n_steps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


# ── Classe LSTM PyTorch ───────────────────────────────────────
def _build_lstm_pytorch(n_steps, hidden_size=64, num_layers=2, dropout=0.2):
    """
    Construit un modèle LSTM avec PyTorch.
    Architecture : LSTM(64, 2 couches) -> Dropout(0.2) -> Linear(1)
    """
    import torch
    import torch.nn as nn

    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size  = 1,
                hidden_size = hidden_size,
                num_layers  = num_layers,
                dropout     = dropout,
                batch_first = True,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc      = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out     = self.dropout(out[:, -1, :])  # derniere sortie
            return self.fc(out)

    return LSTMModel()


def train_lstm(df, article=None, n_steps=30, epochs=50, n_test=30):
    """
    Reseau LSTM (PyTorch) pour la prevision de series temporelles.
    Compatible Python 3.12+ et 3.14.
    Architecture : LSTM(64, 2 couches) -> Dropout(0.2) -> Linear(1)

    - article : filtrer un article specifique (None = tous articles agrege)
    - n_steps : fenetre temporelle en jours (ex: 30 = utilise 30 jours passes)
    - epochs  : nombre d epoques d entrainement
    - n_test  : nombre de jours du jeu de test
    Installation : pip install torch
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("PyTorch non installe. Lancez : pip install torch")
        return {}

    print("\n" + "="*55)
    print("LSTM — Reseau de Neurones Recurrent (PyTorch)")
    print("="*55)

    # ── Préparer la série temporelle ─────────────────────────
    df_s = df[df["article"] == article].copy() if article else df.copy()
    serie = (
        df_s.groupby("date_facture")["quantite"]
        .sum().asfreq("D").fillna(0).sort_index()
        .values.reshape(-1, 1)
    )

    # Normalisation [0, 1]
    scaler       = MinMaxScaler(feature_range=(0, 1))
    serie_scaled = scaler.fit_transform(serie)

    # Créer les séquences
    X_seq, y_seq = create_sequences(serie_scaled, n_steps)

    # Split train / test
    split           = len(X_seq) - n_test
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    print(f"Fenetre : {n_steps} jours | Train : {len(X_train)} | Test : {len(X_test)}")

    # Convertir en tenseurs PyTorch
    X_train_t = torch.FloatTensor(X_train).unsqueeze(-1)  # (batch, steps, 1)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(-1)
    X_test_t  = torch.FloatTensor(X_test).unsqueeze(-1)

    # DataLoader pour l'entraînement par batch
    dataset    = TensorDataset(X_train_t, y_train_t)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # ── Construire et entraîner le modèle ────────────────────
    model_lstm = _build_lstm_pytorch(n_steps)
    optimizer  = torch.optim.Adam(model_lstm.parameters(), lr=0.001)
    criterion  = nn.MSELoss()

    best_loss    = float("inf")
    patience_cnt = 0
    patience     = 10         # early stopping

    print(f"\nEntrainement LSTM ({epochs} epoques max, early stopping patience={patience})")
    losses = []

    for epoch in range(epochs):
        model_lstm.train()
        epoch_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model_lstm(X_batch)
            loss   = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        # Early stopping
        if avg_loss < best_loss:
            best_loss    = avg_loss
            best_weights = {k: v.clone() for k, v in model_lstm.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early stopping a l epoque {epoch+1} (best loss: {best_loss:.6f})")
                break

        if (epoch + 1) % 10 == 0:
            print(f"  Epoque {epoch+1:3d}/{epochs} | Loss : {avg_loss:.6f}")

    # Restaurer les meilleurs poids
    model_lstm.load_state_dict(best_weights)

    # ── Prédiction sur le jeu de test ────────────────────────
    model_lstm.eval()
    with torch.no_grad():
        y_pred_scaled = model_lstm(X_test_t).numpy().flatten()

    y_pred = np.maximum(
        scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten(), 0
    )
    y_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    metriques = evaluate_model(y_real, y_pred, "LSTM (PyTorch)")

    return {
        "modele"      : model_lstm,
        "scaler"      : scaler,
        "losses"      : losses,
        "y_pred"      : y_pred,
        "y_real"      : y_real,
        "metriques"   : metriques,
        "serie_scaled": serie_scaled,
        "n_steps"     : n_steps,
    }


def forecast_lstm(lstm_result, n_mois=12):
    """Prevision recursive avec LSTM PyTorch (chaque prediction alimente la suivante)."""
    import torch

    model        = lstm_result["modele"]
    scaler       = lstm_result["scaler"]
    n_steps      = lstm_result["n_steps"]
    serie_scaled = lstm_result["serie_scaled"]

    last_seq    = serie_scaled[-n_steps:].reshape(1, n_steps, 1)
    predictions = []

    model.eval()
    with torch.no_grad():
        for _ in range(n_mois * 30):
            input_t = torch.FloatTensor(last_seq)
            pred    = model(input_t).item()
            predictions.append(pred)
            # Décaler la fenêtre
            last_seq = np.append(last_seq[:, 1:, :], [[[pred]]], axis=1)

    predictions = np.maximum(
        scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten(), 0
    )
    print(f"LSTM : prevision sur {n_mois * 30} jours ({n_mois} mois)")
    return predictions


# ════════════════════════════════════════════════════════════
# COMPARAISON GLOBALE
# ════════════════════════════════════════════════════════════

def compare_all_models(resultats_ml, arima_res=None, prophet_res=None, lstm_res=None):
    """Consolide les metriques de tous les modeles."""
    tous = dict(resultats_ml)
    if arima_res   and "metriques" in arima_res:   tous["ARIMA"]   = arima_res["metriques"]
    if prophet_res and "metriques" in prophet_res: tous["Prophet"] = prophet_res["metriques"]
    if lstm_res    and "metriques" in lstm_res:    tous["LSTM"]    = lstm_res["metriques"]

    print("\n" + "="*60)
    print("COMPARAISON FINALE — TOUS LES MODELES")
    print("="*60)
    print(f"{'Modele':25s} -> R²       | MAE    | RMSE")
    print("-"*60)
    for nom, m in tous.items():
        print(f"{nom:25s} -> R²: {m.get('R²', m.get('R2',0)):.4f} | MAE: {m['MAE']:.2f} | RMSE: {m['RMSE']:.2f}")

    meilleur = max(tous, key=lambda k: tous[k].get("R²", tous[k].get("R2", 0)))
    print(f"\nMeilleur modele : {meilleur}")
    return tous


# ════════════════════════════════════════════════════════════
# PREVISION CA GLOBAL
# ════════════════════════════════════════════════════════════

def predict_ca_global(model, df):
    """CA = Somme( quantite_prevue(i) x prix_dernier(i) ) — donnees historiques."""
    print("\n" + "="*55)
    print("CA GLOBAL — dernier prix par article")
    print("="*55)

    X, _ = prepare_xy(df)
    df   = df.copy()
    df["quantite_prevue"] = model.predict(X).clip(min=0)

    if "prix_dernier" not in df.columns:
        raise ValueError("Colonne 'prix_dernier' absente. Verifiez data_extraction.py")

    df["ca_prevu"] = df["quantite_prevue"] * df["prix_dernier"]

    ca_mensuel = (
        df.groupby(["annee", "mois"])
        .agg(ca_prevu_total=("ca_prevu","sum"), quantite_prevue=("quantite_prevue","sum"))
        .reset_index().sort_values(["annee","mois"])
    )
    ca_total = df["ca_prevu"].sum()
    print(f"CA Global Prevu total : {ca_total:,.2f} DT")
    print(ca_mensuel.to_string(index=False))
    return ca_mensuel, ca_total


def predict_future_ca(model, df, n_mois=12):
    """CA futur = Somme( quantite_prevue(i) x prix_dernier(i) ) sur n_mois mois."""
    print(f"\nPrevision CA futur — {n_mois} mois a venir...")
    print (f"ahhhhhhhhhhhhhhhhhhhhhi: {df}")

    derniere_date = df["date_facture"].max()

    if "prix_dernier" not in df.columns:
        raise ValueError("Colonne 'prix_dernier' absente. Verifiez data_extraction.py")

    prix_ref = (
        df.groupby("article")["prix_dernier"].last()
        .reset_index().rename(columns={"prix_dernier": "prix_ref"})
    )
    stats_article = (
        df.groupby("article")
        .agg(qte_moy_art=("quantite","mean"), qte_lag_1=("quantite","last"), qte_lag_7=("quantite","last"))
        .reset_index().merge(prix_ref, on="article", how="left")
    )

    futures = []
    for mois_offset in range(1, n_mois + 1):
        future_date = derniere_date + pd.DateOffset(months=mois_offset)
        for _, row in stats_article.iterrows():
            qte_moy_mois = df[
                (df["article"] == row["article"]) & (df["mois"] == future_date.month)
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

    ca_mensuel = (
        df_future.groupby(["annee","mois"])
        .agg(ca_prevu_total=("ca_prevu","sum"), quantite_prevue=("quantite_prevue","sum"))
        .reset_index().sort_values(["annee","mois"])
    )
    ca_total = df_future["ca_prevu"].sum()
    print(f"CA Prevu sur {n_mois} mois : {ca_total:,.2f} DT")
    print(ca_mensuel.to_string(index=False))
    return ca_mensuel, ca_total

# def predict_future_ca_2(model, df, n_mois=1):
#     """CA futur = Somme( quantite_prevue(i) x prix_dernier(i) ) sur n_mois mois."""
#     from sklearn.preprocessing import LabelEncoder
#     print(f"\nPrevision CA futur — {n_mois} mois a venir...")

#     df = df.copy()
#     df["date_facture"] = pd.to_datetime(df["date_facture"])
#     derniere_date = df["date_facture"].max()

#     if "prix_dernier" not in df.columns:
#         raise ValueError("Colonne 'prix_dernier' absente. Verifiez data_extraction.py")

#     prix_ref = (
#         df.groupby("article")["prix_dernier"].last()
#         .reset_index().rename(columns={"prix_dernier": "prix_ref"})
#     )
#     stats_article = (
#         df.groupby("article")
#         .agg(qte_moy_art=("quantite","mean"), qte_lag_1=("quantite","last"), qte_lag_7=("quantite","last"))
#         .reset_index().merge(prix_ref, on="article", how="left")
#     )

#     futures = []
#     for mois_offset in range(1, n_mois + 1):
#         future_date = derniere_date + pd.DateOffset(months=mois_offset)
#         for _, row in stats_article.iterrows():
#             qte_moy_mois = df[
#                 (df["article"] == row["article"]) & (df["mois"] == future_date.month)
#             ]["quantite"].mean()
#             if pd.isna(qte_moy_mois):
#                 qte_moy_mois = row["qte_moy_art"]
#             futures.append({
#                 "date_facture"   : future_date,
#                 "article"        : row["article"],   # gardé pour affichage
#                 "mois"           : future_date.month,
#                 "trimestre"      : future_date.quarter,
#                 "annee"          : future_date.year,
#                 "jour"           : 15,
#                 "jour_semaine"   : future_date.dayofweek,
#                 "qte_moy_article": row["qte_moy_art"],
#                 "qte_moy_mois"   : qte_moy_mois,
#                 "qte_lag_1"      : row["qte_lag_1"],
#                 "qte_lag_7"      : row["qte_lag_7"],
#                 "prix_dernier"   : row["prix_ref"],
#             })

#     df_future = pd.DataFrame(futures)

#     # ── Encoder article string → entier pour le modèle ───────
#     le = LabelEncoder()
#     le.fit(df["article"].astype(str))
#     df_future["article_nom"] = df_future["article"].astype(str)   # garder pour affichage
#     df_future["article"]     = df_future["article"].astype(str).map(
#         lambda x: le.transform([x])[0] if x in le.classes_ else 0
#     )

#     # ── Prédiction ────────────────────────────────────────────
#     available = [f for f in FEATURES if f in df_future.columns]
#     df_future["quantite_prevue"] = model.predict(df_future[available].fillna(0)).clip(min=0)
#     df_future["ca_prevu"]        = df_future["quantite_prevue"] * df_future["prix_dernier"]

#     # ── CA mensuel global ────────────────────────────────────
#     ca_mensuel = (
#         df_future.groupby(["annee","mois"])
#         .agg(ca_prevu_total=("ca_prevu","sum"), quantite_prevue=("quantite_prevue","sum"))
#         .reset_index().sort_values(["annee","mois"])
#     )
#     ca_total = df_future["ca_prevu"].sum()
#     print(f"\nCA Prevu sur {n_mois} mois : {ca_total:,.2f} DT")
#     print(ca_mensuel.to_string(index=False))

#     # ── Détail par article pour chaque mois ──────────────────
#     print(f"\n{'Annee':>6} {'Mois':>5}  {'Article':15s} {'Qte':>10} {'Prix':>10} {'CA (DT)':>14}")
#     print("-" * 65)
#     for (annee, mois), grp in df_future.groupby(["annee","mois"]):
#         detail = (
#             grp.groupby("article_nom")
#             .agg(qte=("quantite_prevue","sum"), prix=("prix_dernier","mean"), ca=("ca_prevu","sum"))
#             .reset_index()
#             .sort_values("ca", ascending=False)
#         )
#         for _, r in detail.iterrows():
#             print(f"{int(annee):>6} {int(mois):>5}  {r['article_nom']:15s} "
#                   f"{r['qte']:>10.0f} {r['prix']:>10.2f} {r['ca']:>14,.2f}")
#         sous_total = detail["ca"].sum()
#         print(f"{'':>6} {'':>5}  {'TOTAL MOIS':15s} {'':>10} {'':>10} {sous_total:>14,.2f}")
#         print("-" * 65)

#     return ca_mensuel, ca_total


if __name__ == "__main__":
    print("Lancez ce module via main.py")
