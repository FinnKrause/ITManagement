import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../steam-200k.csv")

# Spaltennamen vergeben
df.columns = ["user_id", "game", "behavior", "hours", "other"]


# 1️⃣ Spieler kategorisieren (Tester vs. Bleiber)
df["player_type"] = df["hours"].apply(lambda x: "Tester" if x < 3 else "Bleiber")

# 2️⃣ Anzahl Spieler pro Spiel & Kategorie zählen
summary = df.groupby(["game", "player_type"])["user_id"].count().reset_index()

# 1️⃣ Pivot-Tabelle erstellen: jede Spiel hat Spalten "Tester" und "Bleiber"
summary_pivot = summary.pivot(index="game", columns="player_type", values="user_id").fillna(0)

# 2️⃣ Total Spieler pro Spiel berechnen
summary_pivot["total_players"] = summary_pivot["Tester"] + summary_pivot["Bleiber"]

# 3️⃣ Bleiber-Anteil berechnen
summary_pivot["bleiber_percent"] = (summary_pivot["Bleiber"] / summary_pivot["total_players"]) * 100

# 4️⃣ Nach Bleiber-Anteil sortieren
summary_pivot = summary_pivot.sort_values(by="bleiber_percent", ascending=False)

# 5️⃣ Optional: nur die Top 20 Spiele anzeigen
top20 = summary_pivot.head(20)

# 6️⃣ Balkendiagramm plotten
plt.figure(figsize=(12,6))
top20["bleiber_percent"].plot(kind="bar", color="skyblue")
plt.title("Top 20 Spiele nach Bleiber-Anteil (>3 Stunden Spielzeit)")
plt.ylabel("Bleiber-Anteil (%)")
plt.xlabel("Spiel")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
