import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSV laden
df = pd.read_csv("steam-200k.csv", header=None, names=["user_id", "game", "behavior", "hours", "zero"])

# Grundsicht
purchases = df[df["behavior"] == "purchase"]
plays_raw = df[(df["behavior"] == "play") & (df["hours"] > 0)]

# Owners je Spiel (unique)
owners_per_game = purchases.groupby("game")["user_id"].nunique().rename("owners")

# Spielstunden je User–Spiel aggregieren
plays = plays_raw.groupby(["user_id","game"])["hours"].sum().reset_index()

# Players je Spiel (unique)
players_per_game = plays.groupby("game")["user_id"].nunique().rename("players")

# Metrik-Tabelle je Spiel
game_stats = (
    owners_per_game.to_frame()
    .join(players_per_game, how="left").fillna(0)
)
# Hours je Spiel (Summe/Median pro Owner)
hours_per_game = plays.groupby("game")["hours"].agg(total_hours="sum", median_hours="median")
game_stats = game_stats.join(hours_per_game, how="left").fillna({"total_hours":0, "median_hours":0})

# Raten
game_stats["play_rate"] = (game_stats["players"] / game_stats["owners"]).replace([np.inf, np.nan], 0)
game_stats["backlog_rate"] = 1 - game_stats["play_rate"]

# --- Beispiel-Chart 1: Funnel/Play-Rate Top 20 nach Owners ---
top = game_stats[game_stats["owners"]>=5].sort_values("owners", ascending=False).head(20)
ax = top.sort_values("play_rate").plot(kind="barh", y="play_rate", figsize=(7,6))
ax.set_xlabel("Play-Rate (Players / Owners)")
ax.set_ylabel("Game")
plt.tight_layout(); plt.show()

# --- Beispiel-Chart 2: Popularity vs Engagement (Bubble) ---
subset = game_stats[game_stats["owners"]>=20].copy()
subset["bubble"] = subset["players"] / subset["owners"] # als Proxy für Größe
plt.figure(figsize=(7,6))
plt.scatter(subset["owners"], subset["median_hours"], s=300*subset["bubble"], alpha=0.6)
plt.xscale("log")
plt.xlabel("# Owners (log)"); plt.ylabel("Median Hours per Owner")
plt.title("Popularity vs Engagement")
plt.tight_layout(); plt.show()

# --- Beispiel-Chart 3: Collector vs Grinder (Scatter) ---
user_hours = plays.groupby("user_id")["hours"].sum()

owned_games = purchases.groupby("user_id")["game"].nunique().rename("owned_games")
user_stats = (
    owned_games.to_frame()
    .join(user_hours.rename("total_hours"), how="outer")
    .fillna(0)
)
plt.figure(figsize=(7,6))
plt.scatter(
    user_stats["owned_games"],
    user_stats["total_hours"],
    alpha=0.3,
    s=20
)
plt.xlabel("# Gekaufte Spiele je User"); plt.ylabel("Gesamte Spielstunden je User")
plt.title("Collector vs. Grinder")
plt.tight_layout(); plt.show()

# --- Beispiel-Chart 4: Stunden-Verteilung pro User (Histogram, log) ---
plt.figure(figsize=(7,6))
plt.hist(user_hours, bins=50)
plt.xscale("log")
plt.xlabel("Total Hours per User (log)"); plt.ylabel("# Users")
plt.title("Verteilung Spielstunden pro User")
plt.tight_layout(); plt.show()

# --- Beispiel-Chart 5: Lorenz + Gini ---
vals = user_hours.values
vals_sorted = np.sort(vals)
cum_hours = np.cumsum(vals_sorted) / vals_sorted.sum() if vals_sorted.sum()>0 else np.zeros_like(vals_sorted)
cum_users = np.arange(1, len(vals_sorted)+1) / len(vals_sorted) if len(vals_sorted)>0 else np.array([])
plt.figure(figsize=(7,6))
plt.plot(cum_users, cum_hours, label="Lorenz")
plt.plot([0,1],[0,1], linestyle="--", label="Gleichverteilung")
plt.xlabel("kumulierte User"); plt.ylabel("kumulierte Stunden")
plt.title("Lorenz-Kurve")
plt.legend(); plt.tight_layout(); plt.show()

# Gini (1 - 2 * Fläche unter Lorenz)
gini = 1 - 2*np.trapz(cum_hours, cum_users) if len(cum_users)>0 else np.nan
print("Gini:", round(float(gini), 3))
