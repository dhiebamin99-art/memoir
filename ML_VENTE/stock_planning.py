# ============================================================
# stock_planning.py — Planification des stocks et des achats
# ============================================================
#
# Formules appliquées :
#   1) Besoins PF    = Prévision mensuelle par article
#   2) Achats PF     = max(0, Besoins PF - Stock actuel PF)
#   3) Besoins MP    = Σ( Achats PF(i) × BOM_qty(i, composant) )
#   4) Achats MP brut= Besoins MP par composant
#   5) Achats MP net = max(0, Achats MP brut - Stock actuel MP)
#   6) Stock final PF= Stock actuel PF + Achats PF - Prévision
#   7) Stock final MP= Stock actuel MP + Achats MP net - Besoins MP
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path


# ════════════════════════════════════════════════════════════
# EXTRACTION SQL — DONNÉES MÉTIER
# ════════════════════════════════════════════════════════════

def load_bom(conn) -> pd.DataFrame:
    """
    Charge la nomenclature (Bill of Materials).
    Colonnes : Produit_fini | Composant | Quantite
    """
    query = """
    SELECT
        B.ITMREF_0   AS Produit_fini,
        D.CPNITMREF_0 AS Composant,
        D.BOMQTY_0   AS Quantite
    FROM GRMATEGS.BOM B
    INNER JOIN GRMATEGS.BOMD D
        ON B.ITMREF_0 = D.ITMREF_0
    """
    df = pd.read_sql(query, conn)
    print(f"✅ BOM chargée : {len(df)} lignes | "
          f"{df['Produit_fini'].nunique()} PF | "
          f"{df['Composant'].nunique()} composants")
    return df


def load_stock_pf(conn) -> pd.DataFrame:
    """
    Charge le stock des produits finis.
    Colonnes : Article | Designation | Quantite_Stock
    """
    query = """
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
    df = pd.read_sql(query, conn)
    df["Quantite_Stock"] = pd.to_numeric(df["Quantite_Stock"], errors="coerce").fillna(0)
    print(f"✅ Stock PF chargé : {len(df)} articles | total={df['Quantite_Stock'].sum():,.0f}")
    return df


def load_stock_mp(conn) -> pd.DataFrame:
    """
    Charge le stock des matières premières.
    Colonnes : Article | Designation | Quantite_Stock
    """
    query = """
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
    df = pd.read_sql(query, conn)
    df["Quantite_Stock"] = pd.to_numeric(df["Quantite_Stock"], errors="coerce").fillna(0)
    print(f"✅ Stock MP chargé : {len(df)} articles | total={df['Quantite_Stock'].sum():,.0f}")
    return df


# ════════════════════════════════════════════════════════════
# ÉTAPE 1 — BESOINS PF
# ════════════════════════════════════════════════════════════

def compute_besoins_pf(ca_mensuel_detail: pd.DataFrame) -> pd.DataFrame:
    """
    Résume les prévisions de quantité par article (tous mois confondus
    ou par mois selon le détail fourni).

    Paramètre
    ---------
    ca_mensuel_detail : DataFrame avec colonnes
        article | annee | mois | quantite_prevue | ca_prevu

    Retourne
    --------
    DataFrame : article | qte_totale_prevue | ca_total_prevu
    """
    besoins = (
        ca_mensuel_detail
        .groupby("article")
        .agg(
            qte_totale_prevue=("quantite_prevue", "sum"),
            ca_total_prevu   =("ca_prevu",        "sum"),
        )
        .reset_index()
    )
    print(f"\n✅ Besoins PF calculés : {len(besoins)} articles")
    return besoins


# ════════════════════════════════════════════════════════════
# ÉTAPE 2 — ACHATS PF
# ════════════════════════════════════════════════════════════

def compute_achats_pf(besoins_pf: pd.DataFrame,
                      stock_pf: pd.DataFrame) -> pd.DataFrame:
    """
    Achats PF = max(0, Prévision - Stock actuel PF)

    Paramètres
    ----------
    besoins_pf : article | qte_totale_prevue | ca_total_prevu
    stock_pf   : Article | Designation | Quantite_Stock

    Retourne
    --------
    DataFrame : article | prevision | stock_actuel | achats_pf | statut
    """
    df = besoins_pf.merge(
        stock_pf.rename(columns={
            "Article"        : "article",
            "Quantite_Stock" : "stock_actuel_pf",
            "Designation"    : "designation_pf",
        }),
        on="article", how="left"
    )
    df["stock_actuel_pf"] = df["stock_actuel_pf"].fillna(0)

    df["achats_pf"] = (df["qte_totale_prevue"] - df["stock_actuel_pf"]).clip(lower=0)
    # df["couverture_pf"] = np.where(
    #     df["qte_totale_prevue"] > 0,
    #     df["stock_actuel_pf"] / df["qte_totale_prevue"] * 100,
    #     100
    # ).clip(upper=100)
    df["couverture_pf"] = np.clip(
        np.where(
            df["qte_totale_prevue"] > 0,
            df["stock_actuel_pf"] / df["qte_totale_prevue"] * 100,
            100
        ),
        None,
        100
    )
    df["statut_pf"] = pd.cut(
        df["couverture_pf"],
        bins=[-1, 25, 75, 100],
        labels=["🔴 Rupture risque", "🟡 Stock faible", "🟢 Suffisant"]
    )

    print(f"\n✅ Achats PF calculés")
    print(f"   Articles nécessitant un achat : {(df['achats_pf'] > 0).sum()}")
    print(f"   Articles avec stock suffisant : {(df['achats_pf'] == 0).sum()}")
    return df


# ════════════════════════════════════════════════════════════
# ÉTAPE 3 — BESOINS MP (via BOM)
# ════════════════════════════════════════════════════════════

def compute_besoins_mp(achats_pf: pd.DataFrame,
                       bom: pd.DataFrame) -> pd.DataFrame:
    """
    Besoins MP = Σ( Achats PF(produit) × BOM_qty(produit, composant) )

    Paramètres
    ----------
    achats_pf : article | achats_pf | ...
    bom       : Produit_fini | Composant | Quantite

    Retourne
    --------
    DataFrame : Composant | besoins_mp_total
    """
    # Jointure BOM × achats PF
    df = bom.merge(
        achats_pf[["article", "achats_pf"]].rename(columns={"article": "Produit_fini"}),
        on="Produit_fini", how="inner"
    )
    df["achats_pf"] = df["achats_pf"].fillna(0)

    # Besoins = achats PF × quantité BOM
    df["besoin_mp"] = df["achats_pf"] * df["Quantite"]

    besoins = (
        df.groupby("Composant")
        .agg(
            besoins_mp_total =("besoin_mp",    "sum"),
            nb_produits_finis=("Produit_fini", "nunique"),
        )
        .reset_index()
    )
    print(f"\n✅ Besoins MP calculés : {len(besoins)} composants")
    print(f"   Besoins MP total : {besoins['besoins_mp_total'].sum():,.0f} unités")
    return besoins


# ════════════════════════════════════════════════════════════
# ÉTAPE 4 & 5 — ACHATS MP NETS
# ════════════════════════════════════════════════════════════

def compute_achats_mp(besoins_mp: pd.DataFrame,
                      stock_mp: pd.DataFrame) -> pd.DataFrame:
    """
    Achats MP nets = max(0, Besoins MP - Stock actuel MP)

    Paramètres
    ----------
    besoins_mp : Composant | besoins_mp_total | nb_produits_finis
    stock_mp   : Article | Designation | Quantite_Stock

    Retourne
    --------
    DataFrame enrichi avec colonnes stock, achats nets, statut
    """
    df = besoins_mp.merge(
        stock_mp.rename(columns={
            "Article"        : "Composant",
            "Quantite_Stock" : "stock_actuel_mp",
            "Designation"    : "designation_mp",
        }),
        on="Composant", how="left"
    )
    df["stock_actuel_mp"] = df["stock_actuel_mp"].fillna(0)

    df["achats_mp_net"] = (df["besoins_mp_total"] - df["stock_actuel_mp"]).clip(lower=0)
    # df["couverture_mp"] = np.where(
    #     df["besoins_mp_total"] > 0,
    #     df["stock_actuel_mp"] / df["besoins_mp_total"] * 100,
    #     100
    # ).clip(upper=100)
    df["couverture_mp"] = np.clip(
    np.where(
            df["besoins_mp_total"] > 0,
            df["stock_actuel_mp"] / df["besoins_mp_total"] * 100,
            100
        ),
        None,
        100
    )
    df["statut_mp"] = pd.cut(
        df["couverture_mp"],
        bins=[-1, 25, 75, 100],
        labels=["🔴 Rupture risque", "🟡 Stock faible", "🟢 Suffisant"]
    )

    print(f"\n✅ Achats MP nets calculés")
    print(f"   Composants à acheter     : {(df['achats_mp_net'] > 0).sum()}")
    print(f"   Composants avec couverture OK : {(df['achats_mp_net'] == 0).sum()}")
    return df


# ════════════════════════════════════════════════════════════
# ÉTAPE 6 & 7 — STOCKS FINAUX SIMULÉS
# ════════════════════════════════════════════════════════════

def compute_stock_final(achats_pf: pd.DataFrame,
                        achats_mp: pd.DataFrame) -> tuple:
    """
    Simule les stocks finaux après achats et consommation.

    Stock final PF = Stock actuel PF + Achats PF - Prévision
    Stock final MP = Stock actuel MP + Achats MP net - Besoins MP

    Retourne
    --------
    (df_stock_final_pf, df_stock_final_mp)
    """
    # PF
    pf = achats_pf.copy()
    pf["stock_final_pf"] = (
        pf["stock_actuel_pf"] + pf["achats_pf"] - pf["qte_totale_prevue"]
    ).clip(lower=0)

    # MP
    mp = achats_mp.copy()
    mp["stock_final_mp"] = (
        mp["stock_actuel_mp"] + mp["achats_mp_net"] - mp["besoins_mp_total"]
    ).clip(lower=0)

    print(f"\n✅ Stocks finaux simulés")
    print(f"   Stock final PF total : {pf['stock_final_pf'].sum():,.0f}")
    print(f"   Stock final MP total : {mp['stock_final_mp'].sum():,.0f}")
    return pf, mp


# ════════════════════════════════════════════════════════════
# RÉSUMÉ TEXTUEL
# ════════════════════════════════════════════════════════════

def print_planning_summary(achats_pf: pd.DataFrame,
                           achats_mp: pd.DataFrame,
                           n_mois: int = 12):
    """Affiche un résumé lisible du plan d'achats."""
    print("\n" + "█" * 60)
    print(f"  PLAN D'ACHATS — Horizon {n_mois} mois")
    print("█" * 60)

    # ── PF ────────────────────────────────────────────────────
    print(f"\n{'':2}{'PRODUITS FINIS':40s} {'Prévision':>12} {'Stock':>10} {'Achats':>10} {'Couv.':>7}")
    print("  " + "─" * 83)
    for _, r in achats_pf.sort_values("achats_pf", ascending=False).iterrows():
        print(f"  {str(r['article'])[:38]:40s} "
              f"{r['qte_totale_prevue']:>12,.0f} "
              f"{r['stock_actuel_pf']:>10,.0f} "
              f"{r['achats_pf']:>10,.0f} "
              f"{r['couverture_pf']:>6.0f}%")
    print("  " + "─" * 83)
    print(f"  {'TOTAL':40s} "
          f"{achats_pf['qte_totale_prevue'].sum():>12,.0f} "
          f"{achats_pf['stock_actuel_pf'].sum():>10,.0f} "
          f"{achats_pf['achats_pf'].sum():>10,.0f}")

    # ── MP ────────────────────────────────────────────────────
    to_buy = achats_mp[achats_mp["achats_mp_net"] > 0].sort_values("achats_mp_net", ascending=False)
    print(f"\n\n{'':2}{'MATIÈRES PREMIÈRES (à acheter)':40s} {'Besoins':>12} {'Stock':>10} {'Achats':>10} {'Couv.':>7}")
    print("  " + "─" * 83)
    for _, r in to_buy.iterrows():
        print(f"  {str(r['Composant'])[:38]:40s} "
              f"{r['besoins_mp_total']:>12,.0f} "
              f"{r['stock_actuel_mp']:>10,.0f} "
              f"{r['achats_mp_net']:>10,.0f} "
              f"{r['couverture_mp']:>6.0f}%")
    if len(to_buy) == 0:
        print("  Aucune matière première à commander.")
    print("  " + "─" * 83)
    print(f"  {'TOTAL':40s} "
          f"{achats_mp['besoins_mp_total'].sum():>12,.0f} "
          f"{achats_mp['stock_actuel_mp'].sum():>10,.0f} "
          f"{achats_mp['achats_mp_net'].sum():>10,.0f}")


# ════════════════════════════════════════════════════════════
# EXPORT EXCEL
# ════════════════════════════════════════════════════════════

def export_planning_excel(achats_pf: pd.DataFrame,
                          achats_mp: pd.DataFrame,
                          ca_mensuel: pd.DataFrame,
                          output_path: str = "plan_achats.xlsx"):
    """
    Exporte le plan d'achats complet dans un fichier Excel multi-onglets.

    Onglets :
      - Résumé CA mensuel
      - Plan Achats PF
      - Plan Achats MP
    """
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        ca_mensuel.to_excel(writer, sheet_name="CA_Mensuel_Prevision", index=False)
        achats_pf.to_excel(writer, sheet_name="Plan_Achats_PF", index=False)
        achats_mp.to_excel(writer, sheet_name="Plan_Achats_MP", index=False)

    print(f"\n✅ Plan d'achats exporté : {output_path}")
    return output_path


# ════════════════════════════════════════════════════════════
# PIPELINE COMPLET
# ════════════════════════════════════════════════════════════

def run_stock_planning(conn,
                       ca_mensuel_detail: pd.DataFrame,
                       n_mois: int = 12,
                       export_excel: bool = True) -> dict:
    """
    Pipeline complet de planification stocks / achats.

    Paramètres
    ----------
    conn               : connexion SQL Server (pyodbc)
    ca_mensuel_detail  : DataFrame issu de predict_future_ca_detail()
                         colonnes : article | annee | mois | quantite_prevue | ca_prevu
    n_mois             : horizon de prévision
    export_excel       : exporter le résultat en .xlsx

    Retourne
    --------
    dict avec clés :
        bom, stock_pf, stock_mp,
        besoins_pf, achats_pf,
        besoins_mp, achats_mp,
        stock_final_pf, stock_final_mp
    """
    print("\n" + "=" * 60)
    print("🏭 PLANIFICATION STOCKS & ACHATS")
    print("=" * 60)

    # Chargement données SQL
    bom      = load_bom(conn)
    stock_pf = load_stock_pf(conn)
    stock_mp = load_stock_mp(conn)

    # Calculs
    besoins_pf = compute_besoins_pf(ca_mensuel_detail)
    achats_pf  = compute_achats_pf(besoins_pf, stock_pf)
    besoins_mp = compute_besoins_mp(achats_pf, bom)
    achats_mp  = compute_achats_mp(besoins_mp, stock_mp)
    stock_final_pf, stock_final_mp = compute_stock_final(achats_pf, achats_mp)

    # Résumé
    print_planning_summary(achats_pf, achats_mp, n_mois)

    # CA mensuel agrégé
    ca_mensuel = (
        ca_mensuel_detail
        .groupby(["annee", "mois"])
        .agg(ca_prevu_total=("ca_prevu", "sum"),
             quantite_prevue=("quantite_prevue", "sum"))
        .reset_index()
        .sort_values(["annee", "mois"])
    )

    # Export
    if export_excel:
        export_planning_excel(stock_final_pf, stock_final_mp, ca_mensuel)

    return {
        "bom"            : bom,
        "stock_pf"       : stock_pf,
        "stock_mp"       : stock_mp,
        "besoins_pf"     : besoins_pf,
        "achats_pf"      : achats_pf,
        "besoins_mp"     : besoins_mp,
        "achats_mp"      : achats_mp,
        "stock_final_pf" : stock_final_pf,
        "stock_final_mp" : stock_final_mp,
        "ca_mensuel"     : ca_mensuel,
    }
