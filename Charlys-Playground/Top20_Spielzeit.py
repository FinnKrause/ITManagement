import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 1) Daten laden
# -----------------------------
csv_path = "../steam-200k.csv" 
cols = ["user_id", "game", "action", "hours", "extra"]
df = pd.read_csv(csv_path, header=None, names=cols)

# -----------------------------
# 2) Nur "play"-Einträge
# -----------------------------
df_play = df[df["action"] == "play"].copy()
df_play["hours"] = pd.to_numeric(df_play["hours"], errors="coerce")
df_play = df_play.dropna(subset=["hours"])

# -----------------------------
# 3) Top 20 Spiele nach Spieleranzahl
# -----------------------------
player_counts = df_play.groupby("game")["user_id"].nunique()
top20_games = player_counts.sort_values(ascending=False).head(20).index

df_top20 = df_play[df_play["game"].isin(top20_games)]

# -----------------------------
# 4) Histogramme der Spielzeiten pro Spiel
# -----------------------------
plt.figure(figsize=(15, 10))
for i, game in enumerate(top20_games, 1):
    game_hours = df_top20[df_top20["game"] == game]["hours"]
    
    # x-Achse auf 0 bis 95. Perzentil der Spielzeiten begrenzen
    x_max = np.percentile(game_hours, 95)
    
    plt.subplot(5, 4, i)
    plt.hist(game_hours, bins=20, range=(0, x_max), color="skyblue", edgecolor="black")
    plt.title(game, fontsize=8)
    plt.xlabel("Stunden")
    plt.ylabel("Anzahl Spieler")
plt.tight_layout()
plt.suptitle("Histogramme der Spielzeiten (Top 20 Spiele, 0-95% Perzentil)", fontsize=16, y=1.02)
plt.show()

# -----------------------------
# 5) Boxplots der Spielzeiten pro Spiel
# -----------------------------
plt.figure(figsize=(12, 6))
box_data = [df_top20[df_top20["game"] == game]["hours"] for game in top20_games]
plt.boxplot(box_data, labels=top20_games, vert=False, patch_artist=True, 
            boxprops=dict(facecolor="skyblue", color="black"),
            medianprops=dict(color="red"))
plt.xlabel("Spielzeit in Stunden")
plt.title("Boxplots der Spielzeiten (Top 20 Spiele)")
plt.tight_layout()
plt.show()

"""
Ergebnis: Im Boxplot sieht man das Counter-Strike O... 
insgesamt von 75% der Spieler am längsten gespielt wird
"""