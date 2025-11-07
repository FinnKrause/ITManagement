
"""
Ziel:
- Spiele in zwei Gruppen einteilen: <=3h ("Tester") und >3h ("Bleiber")
- Absolute und relative Häufigkeiten berechnen
- Top 30 Spiele mit den meisten Spieler-Einträgen auswählen
- Diese Top 30 nach Bleiber-Anteil sortieren und anzeigen

Was zeigt das Diagramm?
- Für jedes der Top 20 meistgespielten Spiele (Meistgesammelten Daten) 
wird der Anteil der Spieler dargestellt, die länger als 3 Stunden spielen („Bleiber“)
und die das Spiel nur "Testen" (<=3h)
- Die Balken zeigen relativen Anteil der Bleiber und Tester pro Spiel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1) Daten laden
# ------------------------------------------------------------
csv_path = "../steam-200k.csv"
cols = ["user_id", "game", "action", "hours", "extra"]
df = pd.read_csv(csv_path, header=None, names=cols)

# ------------------------------------------------------------
# 2) Vorbereiten / Filtern (nur "play"-Einträge, Stunden als float)
# ------------------------------------------------------------
df = df[df["action"] == "play"].copy()
df["hours"] = pd.to_numeric(df["hours"], errors="coerce")
df = df.dropna(subset=["hours"])

# Bucket bilden: <=3h vs. >3h
df["time_bucket"] = np.where(df["hours"] > 3, ">3h", "<=3h")

# ------------------------------------------------------------
# 3) Absolute Häufigkeiten je Spiel und Bucket aggregieren
# ------------------------------------------------------------
abs_counts = (
    df.groupby(["game", "time_bucket"])
    .size()
    .unstack(fill_value=0)
    # .sort_values(by=[">3h", "<=3h"], ascending=False)
)

# Sicherstellen, dass beide Spalten existieren
for col in [">3h", "<=3h"]:
    if col not in abs_counts.columns:
        abs_counts[col] = 0
abs_counts = abs_counts[[">3h", "<=3h"]]

# Summe pro Spiel (wie viele Einträge insgesamt)
abs_counts["total"] = abs_counts.sum(axis=1)


# ------------------------------------------------------------
# 4) Relative Häufigkeiten je Spiel
# ------------------------------------------------------------
rel_freq = abs_counts.copy()
rel_freq[">3h"] = rel_freq[">3h"] / rel_freq["total"]
rel_freq["<=3h"] = rel_freq["<=3h"] / rel_freq["total"]
# Nach Bleiber-Anteil sortieren
# rel_freq = rel_freq.sort_values(by=">3h", ascending=False).head(30)

# ------------------------------------------------------------
# 5) Top 30 / 20 Spiele mit den meisten Spieler-Daten auswählen
# ------------------------------------------------------------
top30_games = rel_freq.sort_values(by="total", ascending=False).head(30)
top20_games = rel_freq.sort_values(by="total", ascending=False).head(20)

print(top30_games)
print(top20_games)

# Nach Bleiber-Anteil sortieren
top30_sorted= top30_games.sort_values(by=">3h", ascending=False)
top20_sorted= top20_games.sort_values(by=">3h", ascending=False)
"""
# ------------------------------------------------------------
# 6) Plot (Top 30)
# ------------------------------------------------------------
num_games = rel_freq.shape[0]
fig_height = max(6, min(0.12 * num_games, 60))  # Limit auf sinnvolle Höhe

plt.figure(figsize=(12, fig_height))
top30_sorted.sort_values(by=[">3h", "<=3h"], ascending=False)[[">3h", "<=3h"]].plot(
    kind="barh", stacked=True, width=0.9, legend=True
)

plt.xlabel("Relative Häufigkeit")
plt.ylabel("Spiel")
plt.title(
    "Relative Häufigkeitsverteilung der Spielzeiten pro Spiel\n(Buckets: <3h vs. >3h)"
)
plt.gca().invert_yaxis()  # Größte oben
plt.legend(title="Zeit-Bucket", loc="lower right")
plt.tight_layout()
plt.show()
"""

# ------------------------------------------------------------
# 7) Plot (Top 20)
# ------------------------------------------------------------
num_games = rel_freq.shape[0]
fig_height = max(6, min(0.12 * num_games, 60))  # Limit auf sinnvolle Höhe

plt.figure(figsize=(12, fig_height))
top20_sorted.sort_values(by=[">3h", "<=3h"], ascending=False)[[">3h", "<=3h"]].plot(
    kind="barh", stacked=True, width=0.9, legend=True
)

plt.xlabel("Relative Häufigkeit")
plt.ylabel("Spiel")
plt.title(
    "Relative Häufigkeitsverteilung der Spielzeiten pro Spiel\n(Buckets: <3h vs. >3h)"
)
plt.gca().invert_yaxis()  # Größte oben
plt.legend(title="Zeit-Bucket", loc="lower right")
plt.tight_layout()
plt.show()

"""
# ------------------------------------------------------------
# 7) Plot: relative Häufigkeitsverteilung (alle unique Spielnamen)
# ------------------------------------------------------------
# Für viele Spiele ist ein horizontaler, gestapelter Balkenplot oft lesbarer.
num_games = rel_freq.shape[0]
fig_height = max(6, min(0.12 * num_games, 60))  # Limit auf sinnvolle Höhe

plt.figure(figsize=(12, fig_height))
rel_freq.sort_values(by=[">3h", "<=3h"], ascending=False)[[">3h", "<=3h"]].plot(
    kind="barh", stacked=True, width=0.9, legend=True
)

plt.xlabel("Relative Häufigkeit")
plt.ylabel("Spiel")
plt.title(
    "Relative Häufigkeitsverteilung der Spielzeiten pro Spiel\n(Buckets: <3h vs. >3h)"
)
plt.gca().invert_yaxis()  # Größte oben
plt.legend(title="Zeit-Bucket", loc="lower right")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 6) (Optional) Die beiden DataFrames zur weiteren Verwendung bereitstellen:
# - abs_counts_df: absolute Häufigkeiten
# - rel_freq:      relative Häufigkeiten
# ------------------------------------------------------------
"""