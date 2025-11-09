import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSV laden
df = pd.read_csv("steam-200k.csv", header=None, names=["user_id", "game", "behavior", "hours", "zero"])

# Zero-Spalte eliminieren
df = df.drop(columns=["zero"])

# Gesamtzahl der User berechnen
total_users = df["user_id"].nunique()

# Nur Kaufwerte berücksichtigen
purchases = df[df["behavior"] == "purchase"]

# Ownership-Rate berechnen (pro Spiel)
ownership_rates = purchases["game"].value_counts() / total_users * 100

# Top 20 Spiele mit höchster Ownership-Rate
top_20_rates = ownership_rates.head(20)

# Diagramm
plt.figure(figsize=(10, 6))
top_20_rates.sort_values().plot(kind="barh")
plt.title("Top 20 Spiele mit der höchsten Ownership-Rate")
plt.xlabel("Ownership-Rate (%)")
plt.ylabel("Spielname")
plt.xlim(0, 40)  # X-Achse bis 50% begrenzen
plt.tight_layout()
plt.show()

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

# Kategorien für Retention definieren
def categorize_retention(rate):
    if rate < 40:
        return "Niedrig (<40%)"
    elif rate <= 80:
        return "Mittel (40-80%)"
    else:
        return "Hoch (>80%)"

# Kategorien hinzufügen
filtered_games['retention_category'] = filtered_games['retention_rate'].apply(categorize_retention)

# Farben für die Kategorien
colors = {'Niedrig (<40%)': 'red', 'Mittel (40-80%)': 'orange', 'Hoch (>80%)': 'green'}

# Scatter Plot mit farbigen Kategorien
plt.figure(figsize=(12, 8))

for category, color in colors.items():
    category_data = filtered_games[filtered_games['retention_category'] == category]
    plt.scatter(category_data['avg_playtime'], category_data['retention_rate'],
               s=category_data['buyers']/50, alpha=0.6, color=color,
               label=category, edgecolors='black', linewidth=0.5)

plt.xlabel("Durchschnittliche Spielzeit (Stunden)")
plt.ylabel("Retention Rate (%)")
plt.title("Spieler-Engagement vs. Retention\n(Größe = Anzahl Käufer, Farbe = Retention-Kategorie)")
plt.grid(True, alpha=0.3)
plt.ylim(0, 100)

# Trendlinie
z = np.polyfit(filtered_games['avg_playtime'], filtered_games['retention_rate'], 1)
p = np.poly1d(z)
plt.plot(filtered_games['avg_playtime'], p(filtered_games['avg_playtime']), "k--", alpha=0.5, label="Trendlinie")

plt.legend()
plt.tight_layout()
plt.show()

# Zusätzlich: Übersicht der Kategorien anzeigen
print("\n=== Übersicht der Retention-Kategorien ===")
category_summary = filtered_games.groupby('retention_category').agg({
    'game': 'count',
    'retention_rate': 'mean',
    'avg_playtime': 'mean',
    'buyers': 'sum'
}).round(2)

category_summary.columns = ['Anzahl Spiele', 'Durchschn. Retention (%)', 'Durchschn. Spielzeit (h)', 'Gesamt Käufer']
print(category_summary)