#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Median vs. Mittelwert – Hardcoded (analytisch robust)
====================================================
Ziel: Verzerrungen durch Ausreißer sauber sichtbar machen.
- Robuste Vorverarbeitung: pro (user_id, game) nur ein Stundenwert (max).
- Metriken pro Spiel: players, mean, median, diff (mean - median), ratio (mean/median),
  IQR, Std, CoV.
- Plots:
  (1) Scatter Mean vs. Median (y=x Referenz), Markergröße ~ Spielerzahl (log).
  (2) Top-20 nach (mean - median): stärkster Ausreißer-Einfluss in absoluten Stunden.
  (3) Ratio (mean/median) vs. Spielerzahl: zeigt Stabilität (log x-Achse).
Pfad und Parameter sind hardcodiert. Letzte CSV-Spalte wird ignoriert (nur 0en).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== Hardcoded Parameter =====
CSV_PATH = "../steam-200k.csv"
MIN_PLAYERS = 50
TOP_N = 20
OUT_DIR = "../images/Finn/"
# ===============================


def load_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path, header=None, names=["user_id", "game", "behavior", "value", "drop_me"]
    )
    df = df.loc[df["behavior"] == "play", ["user_id", "game", "value"]].copy()
    df["hours"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["hours"])
    # pro (user, game) nur ein Wert
    df = df.groupby(["user_id", "game"], as_index=False)["hours"].max()
    return df


def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    by_game = df.groupby("game")["hours"]
    agg = by_game.agg(
        players="count",
        mean_hours="mean",
        median_hours="median",
        q25=lambda s: np.percentile(s, 25),
        q75=lambda s: np.percentile(s, 75),
        std_hours="std",
    ).reset_index()
    agg = agg[agg["players"] >= MIN_PLAYERS].copy()
    agg["iqr"] = agg["q75"] - agg["q25"]
    agg["diff_mean_median"] = agg["mean_hours"] - agg["median_hours"]
    agg["ratio_mean_median"] = agg["mean_hours"] / agg["median_hours"].replace(
        0, np.nan
    )
    agg["cov"] = agg["std_hours"] / agg["mean_hours"].replace(0, np.nan)
    return agg


def plot_scatter_mean_vs_median(agg: pd.DataFrame, out_path: str) -> None:
    sizes = np.log10(agg["players"].values + 1.0) * 30.0  # Markergröße ~ log(players)
    plt.figure(figsize=(8, 8))
    plt.scatter(agg["median_hours"], agg["mean_hours"], s=sizes, alpha=0.6)
    lim = max(agg["median_hours"].max(), agg["mean_hours"].max())
    plt.plot([0, lim], [0, lim], linestyle="--")  # y=x
    plt.xlabel("Median der Spielzeit (Stunden)")
    plt.ylabel("Mittelwert der Spielzeit (Stunden)")
    plt.title("Mean vs. Median der Spielzeit (Markergröße ~ Spielerzahl)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_top_abs_diff(agg: pd.DataFrame, out_path: str) -> None:
    top = agg.nlargest(TOP_N, "diff_mean_median")
    plt.figure(figsize=(12, 8))
    plt.barh(top["game"], top["diff_mean_median"])
    plt.gca().invert_yaxis()
    plt.xlabel("Mean - Median (Stunden)")
    plt.ylabel("Spiel")
    plt.title(f"Top {TOP_N}: Absoluter Ausreißer-Einfluss (Mean - Median)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_ratio_vs_players(agg: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(9, 7))
    plt.scatter(agg["players"], agg["ratio_mean_median"], alpha=0.6)
    plt.xscale("log")
    plt.xlabel("Spielerzahl (log-Skala)")
    plt.ylabel("Mean / Median")
    plt.title("Verhältnis Mean/Median vs. Stichprobengröße (pro Spiel)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    df = load_clean(CSV_PATH)
    agg = compute_stats(df)
    out_csv = OUT_DIR + "MedianVsMittelwert_stats_hardcoded.csv"
    agg.to_csv(out_csv, index=False)
    plot_scatter_mean_vs_median(
        agg, OUT_DIR + "MedianVsMittelwert_scatter_hardcoded.png"
    )
    plot_top_abs_diff(agg, OUT_DIR + "MedianVsMittelwert_topAbsDiff_hardcoded.png")
    plot_ratio_vs_players(
        agg, OUT_DIR + "MedianVsMittelwert_ratioVsPlayers_hardcoded.png"
    )
    print("[OK] geschrieben:", out_csv)
    print(
        "[OK] Plots: MedianVsMittelwert_scatter_hardcoded.png, MedianVsMittelwert_topAbsDiff_hardcoded.png, MedianVsMittelwert_ratioVsPlayers_hardcoded.png"
    )


if __name__ == "__main__":
    main()
