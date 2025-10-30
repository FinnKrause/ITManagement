import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("./steam-200k.csv")

# Spaltennamen vergeben
df.columns = ["user_id", "game", "behavior", "hours", "other"]

print(df.head())
print(df.info())
print(df.describe())

# Fehlende Werte anzeigen
print(df.isnull().sum())  # -> wir haben keine Fehlende Werte

# Spalte "other" entfernen
df = df.drop("other", axis=1)


"""

# Welche Spiele wurden am meisten gespielt (nach Stunden)?
plays = df[df["behavior"] == "play"]
hours_per_game = plays.groupby("game")["hours"].sum().reset_index()
hours_per_game = hours_per_game.sort_values("hours", ascending=False)
print(hours_per_game)

# Spiele Anzahl
games_count = df["game"].nunique()
print("Anzahl verschiedener Spiele:", games_count)

# Spiele Namen
games_name = df["game"].unique()
print(games_name)


# Welche Spiele haben die meisten Spieler?
top_by_players = (
    plays.groupby("game")["user_id"].nunique().sort_values(ascending=False).head(10)
)
print(top_by_players)


plays = df[df["behavior"] == "play"]
hours_per_user = plays.groupby("user_id")["hours"].sum()

hours_per_game = plays.groupby("game")["hours"].mean()


# Auf Genre mergen
mg = pd.read_csv("./metacritic_games.csv")
mgs = mg[[mg.columns[0], "genre", "developer"]].copy()
mgs.columns = ["game", "genre", "developer"]

merged = pd.merge(df, mgs, on="game", how="left")
merged_inner = pd.merge(df, mgs, on="game", how="inner")

print(merged.head())
merged_count = df["game"].nunique()
print(merged_count)

print(merged.count())

anzahl_nan = merged["genre"].isna().sum()
print(anzahl_nan)


print(merged_inner.head())
merged_inner_count = df["game"].nunique()
print(merged_inner_count)

print(merged_inner.count())

anzahl_nan = merged_inner["genre"].isna().sum()
print(anzahl_nan)

# Verschiedene Spiele

"""
