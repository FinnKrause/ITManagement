import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSV laden
df = pd.read_csv("steam-200k.csv", header=None, names=["user_id", "game", "behavior", "hours", "zero"])
total_users = df["user_id"].nunique()

# Basisdaten vorbereiten
purchases = df[df["behavior"] == "purchase"]
plays_raw = df[(df["behavior"] == "play") & (df["hours"] > 0)]

# Besitzer je Spiel (unique)
ownership_pairs = purchases[["user_id", "game"]].drop_duplicates()
owners_per_game = ownership_pairs.groupby("game")["user_id"].nunique().rename("owners")

# Spielstunden je User–Spiel aggregieren
plays = plays_raw.groupby(["user_id", "game"])["hours"].sum().reset_index()

# Alle Besitzer mit ihren Spielstunden (0, falls nie gespielt)
owner_hours = ownership_pairs.merge(plays, on=["user_id", "game"], how="left").fillna({"hours": 0})
owner_hours["played"] = owner_hours["hours"] > 0
owner_hours["played_ge_2h"] = owner_hours["hours"] >= 2
owner_hours["played_ge_3h"] = owner_hours["hours"] >= 3

# Kennzahlen je Spiel
engagement = owner_hours.groupby("game").agg(
    owners=("user_id", "nunique"),
    players=("played", "sum"),
    share_played=("played", "mean"),
    share_played_ge_2h=("played_ge_2h", "mean"),
    share_played_ge_3h=("played_ge_3h", "mean"),
    avg_hours_per_owner=("hours", "mean"),
    median_hours_per_owner=("hours", "median"),
    total_hours=("hours", "sum")
)

# Besitzer vs. Gesamtuser
engagement["ownership_rate"] = engagement["owners"] / total_users
engagement["player_rate"] = engagement["players"] / total_users
engagement["play_rate"] = engagement["players"] / engagement["owners"]
engagement = engagement.replace([np.inf, np.nan], 0)

# Erfolgsindex zur schnellen Sortierung (gewichtete Mischung der Kriterien)
engagement["success_score"] = (
    0.4 * engagement["ownership_rate"].rank(pct=True)
    + 0.3 * engagement["avg_hours_per_owner"].rank(pct=True)
    + 0.3 * engagement["share_played_ge_2h"].rank(pct=True)
)

# Top-Spiele nach Erfolgskennzahlen anzeigen
min_owner_filter = 20
top_success = (
    engagement[engagement["owners"] >= min_owner_filter]
    .sort_values("success_score", ascending=False)
    .head(15)
)
print("Top Spiele nach Erfolgskennzahlen (mind. 20 Besitzer):")
print(
    top_success[
        [
            "owners",
            "ownership_rate",
            "players",
            "player_rate",
            "avg_hours_per_owner",
            "share_played_ge_2h",
            "share_played_ge_3h",
        ]
    ].round(
        {
            "ownership_rate": 4,
            "player_rate": 4,
            "avg_hours_per_owner": 2,
            "share_played_ge_2h": 2,
            "share_played_ge_3h": 2,
        }
    )
)

# --- Visualisierung 1: Anteil aktiver Spieler ---
top = engagement[engagement["owners"] >= min_owner_filter].sort_values("share_played", ascending=False).head(20)
ax = top.sort_values("share_played").plot(kind="barh", y="share_played", figsize=(7, 6))
ax.set_xlabel("Anteil aktiver Spieler (Played / Owners)")
ax.set_ylabel("Game")
ax.set_title("Top 20 Spiele nach Aktivierungsrate (mind. 20 Besitzer)")
plt.tight_layout()
plt.show()

# --- Visualisierung 2: Ownership vs. Engagement (Bubble) ---
subset = engagement[engagement["owners"] >= min_owner_filter].copy()
plt.figure(figsize=(7, 6))
plt.scatter(
    subset["owners"],
    subset["avg_hours_per_owner"],
    s=400 * subset["share_played_ge_2h"],
    alpha=0.6,
)
plt.xscale("log")
plt.xlabel("# Besitzer (log)")
plt.ylabel("Durchschnittliche Stunden je Besitzer")
plt.title("Ownership vs. Engagement (Bubble ~ Anteil >= 2h)")
plt.tight_layout()
plt.show()

"""

# --- Visualisierung 3: Collector vs Grinder ---
user_hours = plays.groupby("user_id")["hours"].sum()
owned_games = purchases.groupby("user_id")["game"].nunique().rename("owned_games")
user_stats = (
    owned_games.to_frame()
    .join(user_hours.rename("total_hours"), how="outer")
    .fillna(0)
)
plt.figure(figsize=(7, 6))
plt.scatter(
    user_stats["owned_games"],
    user_stats["total_hours"],
    alpha=0.3,
    s=20,
)
plt.xlabel("# Gekaufte Spiele je User")
plt.ylabel("Gesamte Spielstunden je User")
plt.title("Collector vs. Grinder")
plt.tight_layout()
plt.show()

# --- Visualisierung 4: Stunden-Verteilung pro User (Histogram, log) ---
plt.figure(figsize=(7, 6))
plt.hist(user_hours, bins=50)
plt.xscale("log")
plt.xlabel("Total Hours per User (log)")
plt.ylabel("# Users")
plt.title("Verteilung Spielstunden pro User")
plt.tight_layout()
plt.show()

# --- Visualisierung 5: Lorenz + Gini ---
vals = user_hours.values
vals_sorted = np.sort(vals)
cum_hours = np.cumsum(vals_sorted) / vals_sorted.sum() if vals_sorted.sum() > 0 else np.zeros_like(vals_sorted)
cum_users = np.arange(1, len(vals_sorted) + 1) / len(vals_sorted) if len(vals_sorted) > 0 else np.array([])
plt.figure(figsize=(7, 6))
plt.plot(cum_users, cum_hours, label="Lorenz")
plt.plot([0, 1], [0, 1], linestyle="--", label="Gleichverteilung")
plt.xlabel("kumulierte User")
plt.ylabel("kumulierte Stunden")
plt.title("Lorenz-Kurve")
plt.legend()
plt.tight_layout()
plt.show()

# Gini (1 - 2 * Fläche unter Lorenz)
gini = 1 - 2 * np.trapezoid(cum_hours, cum_users) if len(cum_users) > 0 else np.nan
print("Gini:", round(float(gini), 3))
"""
