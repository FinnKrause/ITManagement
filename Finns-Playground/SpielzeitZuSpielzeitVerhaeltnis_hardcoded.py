#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spielzeit-zu-Spielzeit (robust) – Hardcoded
===========================================
Analytisch sinnvolle Visualisierungen der "Tail-Heaviness" der Spielzeit-Verteilungen.
- Robuste Vorverarbeitung: pro (user_id, game) nur ein Wert (max der Stunden).
- Metriken pro Spiel: P25, P50, P75, P90, P95, Gini-Koeffizient.
- Plots:
  (1) Top-20 Boxplots der Spielzeiten (log10-Skala) für die größten Spiele nach Spielerzahl.
  (2) Scatter: P90/P50 vs. Spielerzahl (zeigt Stabilität vs. Tail-Heaviness).
  (3) Lorenzkurven der Top-5 nach Spielerzahl (Ungleichverteilung der Spielzeit).
Pfad und Parameter sind hardcodiert. Die letzte CSV-Spalte wird ignoriert (nur 0en).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== Hardcoded Parameter =====
CSV_PATH = "../steam-200k.csv"
MIN_PLAYERS = 100
TOP_N = 20
OUT_DIR = "../images/Finn/"
# ===============================


def load_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path, header=None, names=["user_id", "game", "behavior", "value", "drop_me"]
    )
    # Nur 'play', letzte Spalte ignorieren; Stunden numerisch
    df = df.loc[df["behavior"] == "play", ["user_id", "game", "value"]].copy()
    df["hours"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["hours"])
    # Pro (user, game) nur ein Stundenwert (max, da 'value' typ. kumulierte Stunden sind)
    df = df.groupby(["user_id", "game"], as_index=False)["hours"].max()
    return df


def gini_coefficient(x: np.ndarray) -> float:
    """Gini-Koeffizient (0 = gleich verteilt, 1 = extrem ungleich)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    gini = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    return float(gini)


def compute_game_stats(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("game")["hours"]
    agg = grouped.agg(
        players="count",
        p25=lambda s: np.percentile(s, 25),
        p50=lambda s: np.percentile(s, 50),
        p75=lambda s: np.percentile(s, 75),
        p90=lambda s: np.percentile(s, 90),
        p95=lambda s: np.percentile(s, 95),
        mean="mean",
    ).reset_index()
    # Gini pro Spiel
    agg["gini"] = grouped.apply(lambda s: gini_coefficient(s.values)).values
    # Tail-Ratios (robust gegen Division durch 0)
    agg["ratio_p90_p50"] = agg["p90"] / agg["p50"].replace(0, np.nan)
    agg["ratio_p75_p25"] = agg["p75"] / agg["p25"].replace(0, np.nan)
    # Filter auf ausreichend Stichprobe
    agg = agg[agg["players"] >= MIN_PLAYERS].copy()
    return agg


def plot_top20_boxplots_log(df: pd.DataFrame, out_path: str) -> None:
    # Top-20 Spiele nach Spielerzahl
    top_games = (
        df.groupby("game")["user_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(TOP_N)
        .index.tolist()
    )
    data = [np.log10(df.loc[df["game"] == g, "hours"].values + 1.0) for g in top_games]
    plt.figure(figsize=(14, 8))
    plt.boxplot(data, vert=False, showfliers=False)
    plt.yticks(range(1, len(top_games) + 1), top_games)
    plt.xlabel("log10(Spielzeit in Stunden + 1)")
    plt.title(
        f"Top {len(top_games)} Spiele: Verteilung der Spielzeiten (Boxplots, log-Skala)"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_ratio_vs_players(agg: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(9, 7))
    plt.scatter(agg["players"], agg["ratio_p90_p50"], alpha=0.6)
    plt.xscale("log")
    plt.xlabel("Spielerzahl (log-Skala)")
    plt.ylabel("P90/P50 der Spielzeit")
    plt.title("Tail-Heaviness vs. Stichprobengröße (pro Spiel)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_lorenz_top5(df: pd.DataFrame, out_path: str) -> None:
    # Top-5 nach Spielerzahl
    top5 = (
        df.groupby("game")["user_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )
    plt.figure(figsize=(8, 8))
    # Gleichheitslinie
    plt.plot([0, 1], [0, 1], linestyle="--")
    for g in top5:
        v = df.loc[df["game"] == g, "hours"].values
        v = np.sort(v)
        cum = np.cumsum(v)
        cum = cum / cum[-1] if cum[-1] > 0 else cum
        xs = np.linspace(0, 1, len(cum), endpoint=True)
        plt.plot(xs, cum, label=g)
    plt.xlabel("Kumulierter Anteil der Spieler")
    plt.ylabel("Kumulierter Anteil der Spielzeit")
    plt.title("Lorenzkurven – Ungleichverteilung der Spielzeit (Top 5 Spiele)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    df = load_clean(CSV_PATH)
    agg = compute_game_stats(df)
    # Export Tabelle
    out_csv = OUT_DIR + "SpielzeitVerhaeltnis_stats_hardcoded.csv"
    agg.to_csv(out_csv, index=False)
    # Plots
    plot_top20_boxplots_log(df, OUT_DIR + "Top20_Boxplots_LogHours.png")
    plot_ratio_vs_players(agg, OUT_DIR + "TailRatios_vs_Players.png")
    plot_lorenz_top5(df, OUT_DIR + "Lorenz_Gini_Top5.png")
    print("[OK] geschrieben:", out_csv)
    print(
        "[OK] Plots: Top20_Boxplots_LogHours.png, TailRatios_vs_Players.png, Lorenz_Gini_Top5.png"
    )


if __name__ == "__main__":
    main()
