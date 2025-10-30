import math
import pandas as pd
import numpy as np

df = pd.read_csv("./steam-200k.csv")

df.info()

df.describe(include="all")

df.columns = ["user_id", "game", "behavior", "hours", "other"]

plays = df[df["behavior"] == "play"]

hours_per_game = plays.groupby("game")["hours"].sum().reset_index()

hours_per_game = hours_per_game.sort_values("hours", ascending=False)

print(hours_per_game)

games_count = df["game"].nunique()

print("Anzahl verschiedener Spiele:", games_count)
