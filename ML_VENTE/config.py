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
   AND B.BOMALT_0 = D.BOMALT_0
   AND B.BOMTYPS_0 = D.BOMTYPS_0
WHERE 
    B.BOMSTA_0 = 1                    -- active BOM
    AND (B.BOMDAT_0 <= GETDATE() OR B.BOMDAT_0 IS NULL)
    AND (B.ENDDAT_0 >= GETDATE() OR B.ENDDAT_0 IS NULL)
    AND D.CPNSTA_0 = 1                -- active component in BOM
    AND D.BOMQTY_0 > 0;
"""

STOCKPFSQL = """
    SELECT
    S.ITMREF_0      AS Article,
    I.ITMDES1_0     AS Designation,
    SUM(S.QTYPCU_0) AS Quantite_Stock
FROM [mateg].[GRMATEGS].STOCK S
JOIN [mateg].[GRMATEGS].ITMMASTER I ON S.ITMREF_0 = I.ITMREF_0
WHERE 
    I.ITMSTA_0 = 1                     -- actif
    AND S.STOFCY_0 = @Site
    AND S.PHYSTA_0 = 1                 -- mouvement physique
    AND S.PHYF_0 = 0                   -- non figé
    AND NOT EXISTS (                   -- n'est PAS un composant dans aucune BOM active
        SELECT 1 FROM GRMATEGS.BOMD BD
        JOIN GRMATEGS.BOM B ON B.ITMREF_0 = BD.ITMREF_0 AND B.BOMALT_0 = BD.BOMALT_0
        WHERE BD.CPNITMREF_0 = I.ITMREF_0 AND B.BOMSTA_0 = 1
    )
    AND (
        EXISTS (                       -- soit il a une BOM active (fabriqué)
            SELECT 1 FROM GRMATEGS.BOM B2
            WHERE B2.ITMREF_0 = I.ITMREF_0 AND B2.BOMSTA_0 = 1
        )
        OR EXISTS (                    -- soit il a des commandes clients
            SELECT 1 FROM GRMATEGS.SORDERQ SQ
            WHERE SQ.ITMREF_0 = I.ITMREF_0 AND SQ.SOHNUM_0 IS NOT NULL
        )
    )
GROUP BY S.ITMREF_0, I.ITMDES1_0
ORDER BY S.ITMREF_0;
"""

STOCKMPSQL = """
SELECT
    S.ITMREF_0      AS Article,
    I.ITMDES1_0     AS Designation,
    SUM(S.QTYPCU_0) AS Quantite_Stock
FROM [mateg].[GRMATEGS].STOCK S
JOIN [mateg].[GRMATEGS].ITMMASTER I ON S.ITMREF_0 = I.ITMREF_0
WHERE 
    I.ITMSTA_0 = 1
    AND S.STOFCY_0 = @Site
    AND S.PHYSTA_0 = 1
    AND S.PHYF_0 = 0
GROUP BY S.ITMREF_0, I.ITMDES1_0
ORDER BY S.ITMREF_0;
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
