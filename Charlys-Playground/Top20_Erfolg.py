
"""
Annahme: Erfolg wird berechnet durch: Anzahl Spieler * Anzahl Bleiber
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 1) Datei laden
# -----------------------------
csv_path = "../steam-200k.csv"  # Pfad zu deiner Datei
cols = ["user_id", "game", "action", "hours", "extra"]
df = pd.read_csv(csv_path, header=None, names=cols)

# -----------------------------
# 2) Nur "play"-Einträge
# -----------------------------
df_play = df[df["action"] == "play"].copy()
df_play["hours"] = pd.to_numeric(df_play["hours"], errors="coerce")
df_play = df_play.dropna(subset=["hours"])

# -----------------------------
# 3) Spieler in Tester/Bleiber einteilen
# -----------------------------
df_play["time_bucket"] = np.where(df_play["hours"] > 3, ">3h", "<=3h")

# -----------------------------
# 4) Absolute Spielerzahlen pro Spiel und Kategorie
# -----------------------------
abs_counts = df_play.groupby(["game", "time_bucket"]).size().unstack(fill_value=0)

# Sicherstellen, dass beide Spalten existieren
for col in [">3h", "<=3h"]:
    if col not in abs_counts.columns:
        abs_counts[col] = 0
abs_counts = abs_counts[[">3h", "<=3h"]]

# Gesamtzahl Einträge pro Spiel
abs_counts["total"] = abs_counts.sum(axis=1)

# -----------------------------
# 5) Anteil der Bleiber berechnen
# -----------------------------
abs_counts["bleiber_ratio"] = abs_counts[">3h"] / abs_counts["total"]

# -----------------------------
# 6) Einfacher Erfolgs-Score
# -----------------------------
# Score = Anzahl Spieler * Anteil Bleiber
abs_counts["success_score"] = abs_counts["total"] * abs_counts["bleiber_ratio"]

# -----------------------------
# 7) Top 20 Spiele nach Erfolgs-Score
# -----------------------------
top20 = abs_counts.sort_values(by="success_score", ascending=False).head(20)

# -----------------------------
# 8) Ausgabe
# -----------------------------
print("Top 20 Spiele nach Erfolgs-Score (Spielerzahl × Anteil Bleiber):")
print(top20[["total", ">3h", "<=3h", "bleiber_ratio", "success_score"]])

# -----------------------------
# 9) Optional: Plot
# -----------------------------
plt.figure(figsize=(12, 8))
top20["success_score"].plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Top 20 Spiele nach Erfolgs-Score (Spielerzahl × Anteil Bleiber)")
plt.ylabel("Erfolgs-Score")
plt.xlabel("Spiel")
plt.xticks(rotation=75, ha="right")
plt.tight_layout()
plt.show()


"""
Ergebnis: nur leichte Abweichung zur AnzahlSpieler Tabelle -> macht nicht so viel Sinn
"""
