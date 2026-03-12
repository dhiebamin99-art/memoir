# ============================================================
# data_extraction.py — Connexion SQL et extraction des données
# ============================================================

import pyodbc
import pandas as pd
from config import DB_CONFIG, SQL_QUERY, COLUMN_NAMES


def get_connection():
    """Crée et retourne une connexion pyodbc vers SQL Server."""
    cfg = DB_CONFIG
    conn_str = (
        f"DRIVER={{SQL Server}};"
        f"SERVER={cfg['server']};"
        f"DATABASE={cfg['database']};"
        f"UID={cfg['username']};"
        f"PWD={cfg['password']};"
    )
    conn = pyodbc.connect(conn_str)
    print("✅ Connexion SQL Server établie.")
    return conn


def extract_data(conn) -> pd.DataFrame:
    """Exécute la requête SQL et retourne un DataFrame brut."""
    df = pd.read_sql(SQL_QUERY, conn)
    print(f"✅ Extraction OK : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme les colonnes avec des noms normalisés."""
    df.columns = COLUMN_NAMES
    print(f"✅ Colonnes renommées : {COLUMN_NAMES}")
    return df


def get_last_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait le dernier prix unitaire connu pour chaque article
    (basé sur la date de facture la plus récente).
    Retourne un DataFrame : article → prix_dernier
    """
    df_sorted = df.sort_values("date_facture")
    prix_dernier = (
        df_sorted.groupby("article")["prix_unitaire"]
        .last()
        .reset_index()
        .rename(columns={"prix_unitaire": "prix_dernier"})
    )
    print(f"✅ Dernier prix extrait pour {len(prix_dernier)} articles")
    return prix_dernier


def aggregate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les quantités par date, site et article,
    puis joint le dernier prix connu de chaque article.
    """
    # Agrégation des quantités
    df_agg = (
        df.groupby(["date_facture", "site", "article"])
        .agg(quantite=("quantite", "sum"))
        .reset_index()
    )

    # Dernier prix par article
    prix_dernier = get_last_price(df)

    # Jointure : chaque ligne reçoit le dernier prix de son article
    df_agg = df_agg.merge(prix_dernier, on="article", how="left")

    print(f"✅ Agrégation : {df_agg.shape[0]} lignes | prix_dernier joint par article")
    return df_agg


def load_data() -> pd.DataFrame:
    """Pipeline complet : connexion → extraction → renommage → agrégation."""
    conn = get_connection()
    df   = extract_data(conn)
    df   = rename_columns(df)
    df   = aggregate_data(df)
    conn.close()
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.head())
