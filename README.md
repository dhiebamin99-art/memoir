# 🏭 Pipeline ML — Prévision Ventes & Planification Stocks

## Vue d'ensemble

Pipeline complet en deux blocs :

```
BLOC A — ML Vente          BLOC B — Planification
─────────────────          ──────────────────────
1. Extraction SQL    →     9.  BOM + Stock PF/MP
2. Prétraitement     →     10. Besoins PF
3. Feature Eng.      →     11. Achats PF
4. Augmentation      →     12. Besoins MP (BOM)
5. Modèles ML        →     13. Achats MP nets
6. ARIMA/Prophet/LSTM →    14. Stocks finaux
7. 💾 Sauvegarde     →     15. Visualisations
8. Prévision CA      →     16. Export Excel
```

---

## Structure des fichiers

```
ML_VENTE_STOCK/
│
├── ── BLOC A (ML_VENTE existant) ──────────────
│   ├── config.py              Paramètres globaux
│   ├── data_extraction.py     Extraction SQL Sage X3
│   ├── preprocessing.py       Nettoyage & outliers
│   ├── features.py            Feature engineering
│   ├── augmentation.py        Data augmentation (4 méthodes)
│   ├── modeling.py            RF, GBM, XGB, Stacking, ARIMA, Prophet, LSTM
│   ├── visualization.py       Graphiques ML
│   └── validation.py          Validation walk-forward
│
├── ── NOUVEAUX MODULES ───────────────────────
│   ├── model_saver.py         💾 Sauvegarde/chargement modèles
│   ├── stock_planning.py      🏭 Calculs stocks & achats
│   ├── visualization_stock.py 📊 Graphiques planification
│   └── main_stock.py          🚀 Orchestrateur principal
│
├── models/                    Modèles sauvegardés (auto-créé)
│   ├── stacking_model.joblib
│   ├── metadata.json
│   ├── arima_model.pkl
│   ├── prophet_model.pkl
│   ├── lstm_model.pt
│   └── lstm_scaler.joblib
│
├── graphs/                    Graphiques PNG (auto-créé)
└── plan_achats.xlsx           Export Excel (auto-créé)
```

---

## Installation

```bash
pip install pyodbc pandas numpy scikit-learn xgboost
pip install statsmodels prophet torch joblib openpyxl matplotlib scipy
```

---
## SQL Query

ML_VENTE/Config.py
SQL_QUERY, BOMSQL, STOCKPFSQL et STOCKMPSQL

## Utilisation

### Pipeline complet (entraînement + planification)
```bash
python ML_VENTE/main_stock.py
```

### Rechargement rapide (modèle sauvegardé + nouveaux stocks)
```python
from main_stock import run_with_saved_model
planning = run_with_saved_model(n_mois=12)
```

---

## Formules appliquées

| Étape | Formule |
|-------|---------|
| Achats PF | `max(0, Prévision - Stock PF)` |
| Besoins MP | `Σ(Achats PF × BOM_qty)` |
| Achats MP nets | `max(0, Besoins MP - Stock MP)` |
| Stock final PF | `Stock PF + Achats PF - Prévision` |
| Stock final MP | `Stock MP + Achats MP - Besoins MP` |

---

## Paramètres à configurer dans `main_stock.py`

```python
N_MOIS_PREVISION = 12    # Horizon de prévision
RUN_ARIMA        = True  # Activer ARIMA
RUN_PROPHET      = True  # Activer Prophet
RUN_LSTM         = True  # Activer LSTM
LSTM_EPOCHS      = 50    # Époques LSTM
EXPORT_EXCEL     = True  # Export plan d'achats
```

---

## Graphiques générés

| Fichier | Description |
|---------|-------------|
| `ca_prevision.png` | CA et quantités prévus par mois |
| `achats_pf.png` | Plan achats produits finis |
| `couverture_pf.png` | Taux de couverture stock PF |
| `achats_mp.png` | Plan achats matières premières |
| `couverture_mp.png` | Taux de couverture stock MP |
| `stock_final.png` | Stocks avant/après achats |
| `dashboard_stock.png` | Dashboard récapitulatif |

---

## Codes couleur (couverture stock)

- 🟢 **Suffisant** : couverture ≥ 75%  
- 🟡 **Stock faible** : couverture entre 25% et 75%  
- 🔴 **Rupture risque** : couverture < 25%
