import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) steam laden
steam = pd.read_csv("steam-200k.csv", header=None)
steam.columns = ["user_id", "game", "action", "value", "other"]
steam = steam[steam["action"].isin(["play", "purchase"])]

# spielzeit pro (user, game)
user_game = (
    steam[steam["action"] == "play"]
    .groupby(["user_id", "game"], as_index=False)["value"]
    .sum()
    .rename(columns={"value": "hours_user_game"})
)

# pro spiel: spieler, gesamtstunden, durchschnitt
game_use = user_game.groupby("game", as_index=False).agg(
    players=("user_id", "nunique"), total_hours=("hours_user_game", "sum")
)
game_use["avg_hours_per_player"] = game_use["total_hours"] / game_use["players"]

# käufe pro spiel
purchases = (
    steam[steam["action"] == "purchase"]
    .groupby("game", as_index=False)["user_id"]
    .nunique()
    .rename(columns={"user_id": "purchasers"})
)
game_use = game_use.merge(purchases, on="game", how="left")
game_use["purchasers"] = game_use["purchasers"].fillna(0).astype(int)

# 2) metacritic laden (nur erste spalte als titel + genre)
meta = pd.read_csv("metacritic_games.csv")

# <<<<<<<<<<<<<< HIER WICHTIG >>>>>>>>>>>>>>
# nimm die ERSTE spalte als titel, und 'genre' als genre.
# wenn deine genre-spalte anders heißt, ersetze unten 'genre' durch den richtigen namen.
meta_simple = meta[[meta.columns[0], "genre"]].copy()
meta_simple.columns = ["title", "genre"]

# 3) join: titel == game, alles klein/trimmen
game_use["key"] = game_use["game"].str.strip().str.lower()
meta_simple["key"] = meta_simple["title"].astype(str).str.strip().str.lower()

merged = game_use.merge(meta_simple[["key", "genre"]], on="key", how="left")
print(merged[merged[""]].count())
print(merged["genre"].isna().sum())

# # 4) gruppen nach genre
# genre_agg = merged.groupby("genre", as_index=False).agg(
#     total_players=("players", "sum"),
#     total_hours=("total_hours", "sum"),
#     games=("game", "nunique"),
#     total_purchasers=("purchasers", "sum"),
# )
# genre_agg["avg_hours_per_player"] = (
#     genre_agg["total_hours"] / genre_agg["total_players"]
# )
# genre_agg["purchase_ratio"] = genre_agg["total_purchasers"] / genre_agg["total_players"]

# # 5) plots

# # A) top 15 genres nach spielern
# g1 = genre_agg.sort_values("total_players", ascending=False).head(15)
# plt.figure(figsize=(10, 6))
# plt.bar(g1["genre"].astype(str), g1["total_players"])
# plt.title("Top Genres nach Spielern")
# plt.xlabel("Genre")
# plt.ylabel("Spieler (Summe)")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.show()

# # B) top 15 genres nach Ø stunden pro spieler
# g2 = genre_agg.sort_values("avg_hours_per_player", ascending=False).head(15)
# plt.figure(figsize=(10, 6))
# plt.bar(g2["genre"].astype(str), g2["avg_hours_per_player"])
# plt.title("Top Genres nach Ø Stunden pro Spieler")
# plt.xlabel("Genre")
# plt.ylabel("Ø Stunden pro Spieler")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.show()

# # C) kaufquote pro genre
# g3 = genre_agg.sort_values("purchase_ratio", ascending=False).head(15)
# plt.figure(figsize=(10, 6))
# plt.bar(g3["genre"].astype(str), g3["purchase_ratio"])
# plt.title("Kaufquote pro Genre (Purchasers / Players)")
# plt.xlabel("Genre")
# plt.ylabel("Kaufquote")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.show()

# # D) scatter: spieler vs. Ø stunden pro spieler (spielebene)
# plt.figure(figsize=(10, 6))
# plt.scatter(merged["players"], merged["avg_hours_per_player"], alpha=0.5)
# plt.title("Spiel: Spieler vs. Ø Stunden pro Spieler")
# plt.xlabel("Spieler (unique)")
# plt.ylabel("Ø Stunden pro Spieler")
# plt.tight_layout()
# plt.show()
