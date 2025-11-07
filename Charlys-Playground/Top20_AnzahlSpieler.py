import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1) Datei laden
# -----------------------------
csv_path = "../steam-200k.csv" 
cols = ["user_id", "game", "action", "hours", "extra"]
df = pd.read_csv(csv_path, header=None, names=cols)

# -----------------------------
# 2) Nur Spielzeit-Daten verwenden
# -----------------------------
df_play = df[df["action"] == "play"].copy()

# -----------------------------
# 3) Anzahl Spieler pro Spiel berechnen
# -----------------------------
# Hier zählt jeder Spieler einmal pro Spiel (einzigartige user_id)
players_per_game = df_play.groupby("game")["user_id"].nunique().sort_values(ascending=False)

# -----------------------------
# 4) Top 20 meistgespielte Spiele auswählen
# -----------------------------
top20 = players_per_game.head(20)

# -----------------------------
# 5) Ausgabe
# -----------------------------
print("Top 20 Spiele nach Anzahl eindeutiger Spieler:")
print(top20)

# -----------------------------
# 6) Plot
# -----------------------------
plt.figure(figsize=(12, 8))
top20.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Top 30 Spiele nach Anzahl Spieler")
plt.ylabel("Anzahl eindeutiger Spieler")
plt.xlabel("Spiel")
plt.xticks(rotation=75, ha="right")
plt.tight_layout()
plt.show()
