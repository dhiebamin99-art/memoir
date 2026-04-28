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
    CPY_0       AS [Société],
    NUM_0       AS [N° facture],
    INVDAT_0    AS [Date facture],
    BPCINV_0    AS [Client facturé],
    REP1_0      AS [Représentant],
    ITMREF_0    AS [article],
    QTY_0       AS [Quantité facturée],
    NETPRI_0    AS [Prix unitaire],
    (DISCRGVAL1_0 + DISCRGVAL2_0 + DISCRGVAL3_0 +
     DISCRGVAL4_0 + DISCRGVAL5_0 + DISCRGVAL6_0 +
     DISCRGVAL7_0 + DISCRGVAL8_0 + DISCRGVAL9_0) AS DISCRG,
    AMTNOTLIN_0  AS [Montant HT],
    AMTTAXLIN_0  AS [Montant TVA],
    AMTATILIN_0  AS [Montant TTC],
    SALFCY_0     AS [Site]
FROM [mateg].[GRMATEGS].[SINVOICED]
ORDER BY INVDAT_0 DESC;
"""

BOMSQL = """
SELECT
    B.ITMREF_0   AS Produit_fini,
    D.CPNITMREF_0 AS Composant,
    D.BOMQTY_0   AS Quantite
FROM GRMATEGS.BOM B
INNER JOIN GRMATEGS.BOMD D
    ON B.ITMREF_0 = D.ITMREF_0
"""

STOCKPFSQL = """
SELECT
    S.ITMREF_0   AS Article,
    I.ITMDES1_0  AS Designation,
    SUM(S.QTYPCU_0) AS Quantite_Stock
FROM [mateg].[GRMATEGS].STOCK S
JOIN [mateg].[GRMATEGS].ITMMASTER I ON S.ITMREF_0 = I.ITMREF_0
WHERE (I.TCLCOD_0 NOT IN ('ING','EMB','MTP','SPR'))
GROUP BY S.ITMREF_0, I.ITMDES1_0
ORDER BY S.ITMREF_0
"""

STOCKMPSQL = """
SELECT
    S.ITMREF_0   AS Article,
    I.ITMDES1_0  AS Designation,
    SUM(S.QTYPCU_0) AS Quantite_Stock
FROM [mateg].[GRMATEGS].STOCK S
JOIN [mateg].[GRMATEGS].ITMMASTER I ON S.ITMREF_0 = I.ITMREF_0
WHERE (I.TCLCOD_0 IN ('ING','EMB','MTP','SPR'))
GROUP BY S.ITMREF_0, I.ITMDES1_0
ORDER BY S.ITMREF_0
"""

# --- Noms des colonnes après renommage ---
COLUMN_NAMES = [
    "societe", "num_facture", "date_facture", "client", "representant",
    "article", "quantite", "prix_unitaire", "discount", "montant_ht",
    "montant_tva", "montant_ttc", "site"
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
