# -*- coding: utf-8 -*-
"""
Aufgabe:
- Aggregiere die absoluten Häufigkeiten pro Spiel getrennt nach Spielzeit >3h und <3h
- Erzeuge daraus die relative Häufigkeitsverteilung und visualisiere sie in einem Plot
Hinweise:
- Genau 3 Stunden werden hier NICHT berücksichtigt (streng '<3h' vs. '>3h').
- Pfad zur Datei: /mnt/data/steam-200k.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1) Daten laden
# ------------------------------------------------------------
csv_path = "steam-200k.csv"

# Das Steam-200k-Dataset liegt häufig ohne Header vor. Wir vergeben daher Spaltennamen.
# Falls Ihre Datei Header enthält, können Sie 'header=0' setzen und die 'names'-Zeile entfernen.
cols = ["user_id", "game", "action", "hours", "extra"]
df = pd.read_csv(csv_path, header=None, names=cols)

# ------------------------------------------------------------
# 2) Vorbereiten / Filtern (nur "play"-Einträge, Stunden als float)
# ------------------------------------------------------------
df = df[df["action"] == "play"].copy()
df["hours"] = pd.to_numeric(df["hours"], errors="coerce")
df = df.dropna(subset=["hours"])

# Genau 3 Stunden ausschließen, weil "länger als 3h" und "kürzer als 3h" streng gemeint ist
df = df[df["hours"] != 3]

# Bucket bilden: <3h vs. >3h
df["time_bucket"] = np.where(df["hours"] > 3, ">3h", "<3h")

# ------------------------------------------------------------
# 3) Absolute Häufigkeiten je Spiel und Bucket aggregieren
# ------------------------------------------------------------
# "Einträge" = Zeilenanzahl (keine deduplizierten User). Falls einzigartige Spieler gewünscht sind,
# könnte man z.B. per .nunique() auf user_id aggregieren.
abs_counts = (
    df.groupby(["game", "time_bucket"])
    .size()
    .unstack(fill_value=0)
    .sort_values(by=[">3h", "<3h"], ascending=False)
)

# Optional: sicherstellen, dass beide Spalten existieren
for col in [">3h", "<3h"]:
    if col not in abs_counts.columns:
        abs_counts[col] = 0
abs_counts = abs_counts[[">3h", "<3h"]]

# Dies ist der geforderte DataFrame mit absoluten Häufigkeiten:
abs_counts_df = abs_counts.copy()

# ------------------------------------------------------------
# 4) Relative Häufigkeiten je Spiel
# ------------------------------------------------------------
row_sums = abs_counts_df.sum(axis=1).replace(0, np.nan)
rel_freq = abs_counts_df.div(row_sums, axis=0).fillna(0).head(20)

# ------------------------------------------------------------
# 5) Plot: relative Häufigkeitsverteilung (alle unique Spielnamen)
# ------------------------------------------------------------
# Für viele Spiele ist ein horizontaler, gestapelter Balkenplot oft lesbarer.
num_games = rel_freq.shape[0]
fig_height = max(6, min(0.12 * num_games, 60))  # Limit auf sinnvolle Höhe

plt.figure(figsize=(12, fig_height))
rel_freq.sort_values(by=[">3h", "<3h"], ascending=False)[[">3h", "<3h"]].plot(
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
