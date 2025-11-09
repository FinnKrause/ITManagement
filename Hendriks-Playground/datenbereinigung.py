import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSV laden
df = pd.read_csv("steam-200k.csv", header=None, names=["user_id", "game", "behavior", "hours", "zero"])

#zero spalte Eliminieren, durch ganzen Datensatz durch nur 0 Werte --> Redundant
df = df.drop(columns=["zero"])

#löschen der zero spalte anzeigen lassen
#print(df.columns)
#check hat funktioniert


def count_purchase_play(df):
    anzahl_purchase = (df["behavior"] == "purchase").sum()
    anzahl_play = (df["behavior"] == "play").sum()
    return anzahl_purchase, anzahl_play

# Anwendung:
anz_purchase, anz_play = count_purchase_play(df)
print("Anzahl 'purchase'-Zeilen:", anz_purchase)
print("Anzahl 'play'-Zeilen:", anz_play)
print (anz_play + anz_purchase)

def filter_by_game(df, game_name):
    return df[df["game"] == game_name]

spiel_df = filter_by_game(df, "Hyperdimension Neptunia Re;Birth1")

unique_users = df["user_id"].nunique()
print("\nAnzahl verschiedener User:", unique_users)

unique_games = df["game"].nunique()
print("Anzahl verschiedener Spiele:", unique_games)

# Vorbereitung der Daten für Engagement vs. Retention
# Schritt 1: Kauf- und Play-Daten kombinieren
purchases = df[df["behavior"] == "purchase"][["user_id", "game"]]
play_data = df[df["behavior"] == "play"][["user_id", "game", "hours"]]

# Schritt 2: Für jedes Spiel berechnen:
# - Durchschnittsspielzeit (Engagement)
# - Prozentualer Anteil der Käufer, die das Spiel tatsächlich gespielt haben (Retention)
games_analysis = []

for game in purchases["game"].unique():
    # Anzahl Käufer
    buyers = purchases[purchases["game"] == game]["user_id"].nunique()
    
    # Anzahl Spieler (Käufer, die das Spiel gespielt haben)
    players = play_data[play_data["game"] == game]["user_id"].nunique()
    
    # Retention Rate
    retention_rate = (players / buyers * 100) if buyers > 0 else 0
    
    # Durchschnittsspielzeit
    avg_playtime = play_data[play_data["game"] == game]["hours"].mean() if players > 0 else 0
    
    games_analysis.append({
        "game": game,
        "retention_rate": retention_rate,
        "avg_playtime": avg_playtime,
        "buyers": buyers,
        "players": players
    })

# DataFrame erstellen
games_df = pd.DataFrame(games_analysis)

# Nur Spiele mit mindestens 100 Käufern für aussagekräftige Daten
filtered_games = games_df[games_df["buyers"] >= 100]

# Scatter Plot
plt.figure(figsize=(12, 8))
plt.scatter(filtered_games["avg_playtime"], filtered_games["retention_rate"], 
           s=filtered_games["buyers"]/50, alpha=0.6, c=filtered_games["retention_rate"], cmap="viridis")

plt.xlabel("Durchschnittliche Spielzeit (Stunden)")
plt.ylabel("Retention Rate (%)")
plt.title("Spieler-Engagement vs. Retention\n(Größe = Anzahl Käufer)")
plt.colorbar(label="Retention Rate (%)")
plt.grid(True, alpha=0.3)

# Trendlinie hinzufügen
z = np.polyfit(filtered_games["avg_playtime"], filtered_games["retention_rate"], 1)
p = np.poly1d(z)
plt.plot(filtered_games["avg_playtime"], p(filtered_games["avg_playtime"]), "r--", alpha=0.8)

plt.tight_layout()
plt.show()