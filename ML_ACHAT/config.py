# ============================================================
# config.py — Paramètres globaux du projet
# ============================================================

# --- Connexion SQL Server ---
DB_CONFIG = {
    "server"   : "172.17.1.205\\V11X3",
    "database" : "mateg",
    "username" : "sa",
    "password" : "P@ssw0rd",
}

# --- Requête SQL ---
SQL_QUERY = """
SELECT
    H.CPY_0                    AS CPY,
    H.FCY_0                    AS FCY,
    H.ACCDAT_0                 AS datefac,
    H.BPRNAM_0                 AS fournisseur,
    H.CUR_0                    AS devise,
    H.STA_0                    AS statu,
    D.ITMREF_0                 AS ITMREF,
    D.QTYPUU_0                 AS QTY,
    D.NETPRI_0                 AS PRI,
    D.AMTNOTLIN_0              AS AMTNOT
FROM [mateg].[GRMATEGS].[PINVOICE] H
INNER JOIN [mateg].[GRMATEGS].[PINVOICED] D
       ON H.INVNUM_0 = D.INVNUM_0
      AND H.CPY_0    = D.CPY_0
      --where D.ITMREF_0 = 'MTPPANBOI008'
ORDER BY H.ACCDAT_0 DESC;
"""

# --- Noms des colonnes après renommage ---
COLUMN_NAMES = [
    "societe", "num_facture", "date_facture", "fournisseur", "devise",
    "statu", "article", "quantite", "prix_unitaire",
    "montant_global"
]

# --- Features utilisées pour la modélisation ---
FEATURES = [
    "jour", "annee", "jour_semaine", "article",
    "qte_moy_article", "qte_moy_mois",
    "qte_lag_1", "mois", "qte_lag_7", "trimestre"
]

TARGET = "quantite"

# --- Paramètres outliers ---
IQR_LOWER_FACTOR = 1.5
IQR_UPPER_FACTOR = 1.5

# --- Paramètres augmentation ---
AUGMENTATION_CONFIG = {
    "gaussian_noise_factor" : 0.02,
    "interpolation_samples" : 300,
    "seasonal_samples"      : 300,
    "smogn_samples"         : 200,
}

# --- Paramètres modèles ---
RANDOM_STATE   = 42
TEST_SIZE      = 0.2
N_ESTIMATORS   = 100
MAX_DEPTH_TREE = 5
