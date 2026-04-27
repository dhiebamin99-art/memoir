# ============================================================
# visualization_stock.py — Graphiques planification stocks & achats
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# Palette cohérente
C_GREEN  = "#2ecc71"
C_ORANGE = "#e67e22"
C_RED    = "#e74c3c"
C_BLUE   = "#3498db"
C_DARK   = "#2c3e50"
C_LIGHT  = "#ecf0f1"


# ════════════════════════════════════════════════════════════
# 1. CA MENSUEL PRÉVU
# ════════════════════════════════════════════════════════════

def plot_ca_prevision(ca_mensuel: pd.DataFrame, title: str = "Prévision CA Global") -> None:
    """Barres CA + courbe quantités sur 12 mois."""
    fig, ax1 = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#f8f9fa")
    ax1.set_facecolor("#f8f9fa")

    labels = [f"{int(r.annee)}-{int(r.mois):02d}" for _, r in ca_mensuel.iterrows()]
    x      = np.arange(len(labels))

    bars = ax1.bar(x, ca_mensuel["ca_prevu_total"], color=C_BLUE, alpha=0.75,
                   width=0.6, label="CA Prévu (DT)", zorder=3)
    ax1.set_ylabel("CA Prévu (DT)", color=C_BLUE, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=C_BLUE)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax1.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    # Valeurs sur barres
    for bar, val in zip(bars, ca_mensuel["ca_prevu_total"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                 f"{val:,.0f}", ha="center", va="bottom", fontsize=7, color=C_DARK)

    ax2 = ax1.twinx()
    ax2.plot(x, ca_mensuel["quantite_prevue"], color=C_ORANGE, marker="o",
             linewidth=2.5, markersize=6, label="Qté Prévue", zorder=4)
    ax2.set_ylabel("Quantité Prévue", color=C_ORANGE, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=C_ORANGE)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    plt.title(title, fontsize=14, fontweight="bold", color=C_DARK, pad=15)
    plt.tight_layout()
    plt.savefig("graphs/ca_prevision.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  📊 Graphique sauvegardé : graphs/ca_prevision.png")


# ════════════════════════════════════════════════════════════
# 2. PLAN ACHATS PF — COMPARAISON PRÉVISION VS STOCK
# ════════════════════════════════════════════════════════════

def plot_achats_pf(achats_pf: pd.DataFrame, top_n: int = 20) -> None:
    """Barres groupées : Prévision / Stock actuel / Achats nécessaires."""
    df = achats_pf.sort_values("qte_totale_prevue", ascending=False).head(top_n)
    articles = [str(a)[:15] for a in df["article"]]
    x = np.arange(len(articles))
    w = 0.28

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    ax.bar(x - w,   df["qte_totale_prevue"], w, label="Prévision",     color=C_BLUE,   alpha=0.85)
    ax.bar(x,       df["stock_actuel_pf"],   w, label="Stock Actuel",  color=C_GREEN,  alpha=0.85)
    ax.bar(x + w,   df["achats_pf"],         w, label="Achats Requis", color=C_RED,    alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(articles, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Quantité", fontsize=11)
    ax.set_title(f"Plan Achats Produits Finis — Top {top_n} articles",
                 fontsize=13, fontweight="bold", color=C_DARK)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    plt.tight_layout()
    plt.savefig("graphs/achats_pf.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  📊 Graphique sauvegardé : graphs/achats_pf.png")


# ════════════════════════════════════════════════════════════
# 3. COUVERTURE STOCK PF — JAUGE PAR ARTICLE
# ════════════════════════════════════════════════════════════

def plot_couverture_pf(achats_pf: pd.DataFrame, top_n: int = 20) -> None:
    """Barres horizontales colorées selon le taux de couverture."""
    df = achats_pf.sort_values("couverture_pf").head(top_n)
    colors = [C_RED if v < 25 else (C_ORANGE if v < 75 else C_GREEN)
              for v in df["couverture_pf"]]

    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.45)))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    bars = ax.barh([str(a)[:20] for a in df["article"]], df["couverture_pf"],
                   color=colors, alpha=0.85, edgecolor="white")
    ax.axvline(25, color=C_RED,    linestyle="--", linewidth=1.2, label="Seuil critique 25%")
    ax.axvline(75, color=C_ORANGE, linestyle="--", linewidth=1.2, label="Seuil acceptable 75%")
    ax.axvline(100, color=C_GREEN, linestyle=":",  linewidth=1.2, label="Couverture 100%")

    for bar, val in zip(bars, df["couverture_pf"]):
        ax.text(min(val + 1, 98), bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", va="center", fontsize=8, color=C_DARK)

    ax.set_xlabel("Taux de couverture (%)", fontsize=11)
    ax.set_title("Couverture Stock PF par Article (% besoins couverts)",
                 fontsize=13, fontweight="bold", color=C_DARK)
    ax.set_xlim(0, 115)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("graphs/couverture_pf.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  📊 Graphique sauvegardé : graphs/couverture_pf.png")


# ════════════════════════════════════════════════════════════
# 4. PLAN ACHATS MP
# ════════════════════════════════════════════════════════════

def plot_achats_mp(achats_mp: pd.DataFrame, top_n: int = 20) -> None:
    """Barres groupées : Besoins MP / Stock actuel / Achats nets."""
    df = (achats_mp[achats_mp["achats_mp_net"] > 0]
          .sort_values("achats_mp_net", ascending=False)
          .head(top_n))

    if df.empty:
        print("  Aucun achat MP nécessaire — graphique non généré.")
        return

    composants = [str(c)[:15] for c in df["Composant"]]
    x = np.arange(len(composants))
    w = 0.28

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    ax.bar(x - w, df["besoins_mp_total"], w, label="Besoins MP",    color=C_BLUE,   alpha=0.85)
    ax.bar(x,     df["stock_actuel_mp"],  w, label="Stock Actuel",  color=C_GREEN,  alpha=0.85)
    ax.bar(x + w, df["achats_mp_net"],    w, label="Achats Nets",   color=C_RED,    alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(composants, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Quantité", fontsize=11)
    ax.set_title(f"Plan Achats Matières Premières — Top {top_n} composants",
                 fontsize=13, fontweight="bold", color=C_DARK)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    plt.tight_layout()
    plt.savefig("graphs/achats_mp.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  📊 Graphique sauvegardé : graphs/achats_mp.png")


# ════════════════════════════════════════════════════════════
# 5. COUVERTURE STOCK MP
# ════════════════════════════════════════════════════════════

def plot_couverture_mp(achats_mp: pd.DataFrame, top_n: int = 20) -> None:
    """Barres horizontales couverture MP."""
    df = achats_mp.sort_values("couverture_mp").head(top_n)
    colors = [C_RED if v < 25 else (C_ORANGE if v < 75 else C_GREEN)
              for v in df["couverture_mp"]]

    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.45)))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    bars = ax.barh([str(c)[:20] for c in df["Composant"]], df["couverture_mp"],
                   color=colors, alpha=0.85, edgecolor="white")
    ax.axvline(25,  color=C_RED,    linestyle="--", linewidth=1.2, label="Seuil critique 25%")
    ax.axvline(75,  color=C_ORANGE, linestyle="--", linewidth=1.2, label="Seuil acceptable 75%")
    ax.axvline(100, color=C_GREEN,  linestyle=":",  linewidth=1.2, label="Couverture 100%")

    for bar, val in zip(bars, df["couverture_mp"]):
        ax.text(min(val + 1, 98), bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", va="center", fontsize=8, color=C_DARK)

    ax.set_xlabel("Taux de couverture (%)", fontsize=11)
    ax.set_title("Couverture Stock MP par Composant",
                 fontsize=13, fontweight="bold", color=C_DARK)
    ax.set_xlim(0, 115)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("graphs/couverture_mp.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  📊 Graphique sauvegardé : graphs/couverture_mp.png")


# ════════════════════════════════════════════════════════════
# 6. DASHBOARD RÉCAPITULATIF (4 graphiques en 1)
# ════════════════════════════════════════════════════════════

def plot_stock_dashboard(ca_mensuel: pd.DataFrame,
                         achats_pf: pd.DataFrame,
                         achats_mp: pd.DataFrame,
                         n_mois: int = 12) -> None:
    """
    Dashboard 2×2 :
      [CA mensuel prévu]  [Répartition statuts PF]
      [Top achats PF]     [Top achats MP]
    """
    import matplotlib.patches as mpatches

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#1a1a2e")
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel 1 : CA mensuel ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#16213e")
    labels = [f"{int(r.annee)}-{int(r.mois):02d}" for _, r in ca_mensuel.iterrows()]
    ax1.bar(range(len(labels)), ca_mensuel["ca_prevu_total"],
            color="#4fc3f7", alpha=0.85, width=0.7)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=7, color="white")
    ax1.tick_params(colors="white")
    ax1.set_title(f"CA Prévu sur {n_mois} mois (DT)", color="white", fontsize=11, fontweight="bold")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    for spine in ax1.spines.values():
        spine.set_edgecolor("#4fc3f7")
    ax1.grid(axis="y", color="#ffffff20", linestyle="--")

    # ── Panel 2 : Camembert statuts PF ───────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#16213e")
    statut_counts = achats_pf["statut_pf"].value_counts()
    pie_colors = {
        "🔴 Rupture risque": C_RED,
        "🟡 Stock faible"  : C_ORANGE,
        "🟢 Suffisant"     : C_GREEN,
    }
    colors_pie = [pie_colors.get(s, C_BLUE) for s in statut_counts.index]
    wedges, texts, autotexts = ax2.pie(
        statut_counts.values, labels=None,
        colors=colors_pie, autopct="%1.0f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(edgecolor="white", linewidth=1.5)
    )
    for at in autotexts:
        at.set_color("white"); at.set_fontsize(10); at.set_fontweight("bold")
    legend_patches = [mpatches.Patch(color=c, label=l)
                      for l, c in zip(statut_counts.index, colors_pie)]
    ax2.legend(handles=legend_patches, loc="lower center", fontsize=8,
               facecolor="#1a1a2e", labelcolor="white",
               bbox_to_anchor=(0.5, -0.15), ncol=1)
    ax2.set_title("Statut Couverture PF", color="white", fontsize=11, fontweight="bold")

    # ── Panel 3 : Top achats PF ───────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#16213e")
    top_pf = achats_pf[achats_pf["achats_pf"] > 0].sort_values("achats_pf", ascending=False).head(10)
    if not top_pf.empty:
        arts   = [str(a)[:12] for a in top_pf["article"]]
        bars3  = ax3.barh(arts, top_pf["achats_pf"], color="#ff6b6b", alpha=0.85)
        ax3.set_xlabel("Quantité à acheter", color="white", fontsize=9)
        ax3.tick_params(colors="white", labelsize=8)
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        for spine in ax3.spines.values():
            spine.set_edgecolor("#ff6b6b")
        ax3.grid(axis="x", color="#ffffff20", linestyle="--")
    ax3.set_title("Top 10 Achats PF Requis", color="white", fontsize=11, fontweight="bold")

    # ── Panel 4 : Top achats MP ───────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#16213e")
    top_mp = achats_mp[achats_mp["achats_mp_net"] > 0].sort_values("achats_mp_net", ascending=False).head(10)
    if not top_mp.empty:
        comps  = [str(c)[:12] for c in top_mp["Composant"]]
        ax4.barh(comps, top_mp["achats_mp_net"], color="#ffd93d", alpha=0.85)
        ax4.set_xlabel("Quantité à acheter", color="white", fontsize=9)
        ax4.tick_params(colors="white", labelsize=8)
        ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        for spine in ax4.spines.values():
            spine.set_edgecolor("#ffd93d")
        ax4.grid(axis="x", color="#ffffff20", linestyle="--")
    ax4.set_title("Top 10 Achats MP Requis", color="white", fontsize=11, fontweight="bold")

    fig.suptitle("🏭 DASHBOARD — PLANIFICATION STOCKS & ACHATS",
                 fontsize=16, fontweight="bold", color="white", y=1.01)

    plt.savefig("graphs/dashboard_stock.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print("  📊 Dashboard sauvegardé : graphs/dashboard_stock.png")


# ════════════════════════════════════════════════════════════
# 7. ÉVOLUTION STOCK FINAL PF SIMULÉ
# ════════════════════════════════════════════════════════════

def plot_stock_final(stock_final_pf: pd.DataFrame,
                     stock_final_mp: pd.DataFrame,
                     top_n: int = 15) -> None:
    """Barres avant/après pour PF et MP."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor("#f8f9fa")

    # PF
    df_pf = stock_final_pf.sort_values("stock_final_pf").head(top_n)
    arts  = [str(a)[:15] for a in df_pf["article"]]
    x     = np.arange(len(arts))
    ax1.set_facecolor("#f8f9fa")
    ax1.bar(x - 0.2, df_pf["stock_actuel_pf"], 0.4, label="Stock Actuel", color=C_BLUE,  alpha=0.8)
    ax1.bar(x + 0.2, df_pf["stock_final_pf"],  0.4, label="Stock Final",  color=C_GREEN, alpha=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(arts, rotation=45, ha="right", fontsize=8)
    ax1.set_title("Evolution Stock PF (Actuel → Après Achats)", fontweight="bold")
    ax1.legend(); ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # MP
    df_mp = stock_final_mp.sort_values("stock_final_mp").head(top_n)
    comps = [str(c)[:15] for c in df_mp["Composant"]]
    y     = np.arange(len(comps))
    ax2.set_facecolor("#f8f9fa")
    ax2.bar(y - 0.2, df_mp["stock_actuel_mp"], 0.4, label="Stock Actuel", color=C_BLUE,  alpha=0.8)
    ax2.bar(y + 0.2, df_mp["stock_final_mp"],  0.4, label="Stock Final",  color=C_GREEN, alpha=0.8)
    ax2.set_xticks(y); ax2.set_xticklabels(comps, rotation=45, ha="right", fontsize=8)
    ax2.set_title("Evolution Stock MP (Actuel → Après Achats)", fontweight="bold")
    ax2.legend(); ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("Simulation Stock Final — PF & MP", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("graphs/stock_final.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  📊 Graphique sauvegardé : graphs/stock_final.png")


# ════════════════════════════════════════════════════════════
# RUNNER GLOBAL
# ════════════════════════════════════════════════════════════

def plot_all_stock(planning_results: dict, n_mois: int = 12) -> None:
    """Lance tous les graphiques de planification."""
    import os
    os.makedirs("graphs", exist_ok=True)

    print("\n" + "=" * 55)
    print("📊 VISUALISATIONS — STOCKS & ACHATS")
    print("=" * 55)

    plot_ca_prevision(planning_results["ca_mensuel"])
    plot_achats_pf(planning_results["achats_pf"])
    plot_couverture_pf(planning_results["achats_pf"])
    plot_achats_mp(planning_results["achats_mp"])
    plot_couverture_mp(planning_results["achats_mp"])
    plot_stock_final(planning_results["stock_final_pf"], planning_results["stock_final_mp"])
    plot_stock_dashboard(
        planning_results["ca_mensuel"],
        planning_results["achats_pf"],
        planning_results["achats_mp"],
        n_mois=n_mois,
    )
