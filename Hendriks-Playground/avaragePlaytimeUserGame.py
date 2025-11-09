import pandas as pd
import matplotlib.pyplot as plt

# Daten einlesen
df = pd.read_csv('steam-200k.csv', header=None, names=['UserID', 'Game', 'Action', 'Value', 'Other'])

# Nur Spielzeit-Datensätze filtern
play_data = df[df['Action'] == 'play'].copy()

# Gesamtspielzeit pro Spiel berechnen
game_playtime = play_data.groupby('Game')['Value'].sum().sort_values(ascending=False)

# Top 20 Spiele mit höchster Spielzeit
top_20_games = game_playtime.head(20)

# Durchschnittliche Spielzeit pro User berechnen
user_playtime = play_data.groupby('UserID')['Value'].sum()
avg_playtime_per_user = user_playtime.mean()


# Diagramm erstellen
plt.figure(figsize=(10, 6))
bars = plt.barh(range(len(top_20_games)), top_20_games.values)
plt.yticks(range(len(top_20_games)), top_20_games.index)
plt.xlabel("Gesamtspielzeit (Stunden)")
plt.title("Top 20 Spiele mit den höchsten Spielzeiten")
plt.gca().invert_yaxis()  # Höchste Spielzeit oben
plt.ticklabel_format(style="plain", axis="x")
plt.tight_layout()
plt.show()
