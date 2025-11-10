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


grenze_hours_played = 10
min_entries_per_game = 150  # Mindestanzahl an Einträgen pro Spiel für die Analyse

# ------------------------------------------------------------
# 1) Daten laden
# ------------------------------------------------------------
csv_path = "steam-200k.csv"

cols = ["user_id", "game", "action", "hours", "extra"]
df = pd.read_csv(csv_path, header=None, names=cols)

# ------------------------------------------------------------
# 2) Vorbereiten / Filtern (nur "play"-Einträge, Stunden als float)
# ------------------------------------------------------------
df = df[df["action"] == "play"].copy()
df["hours"] = pd.to_numeric(df["hours"], errors="coerce")
df = df.dropna(subset=["hours"])

# ------------------------------------------------------------
# 2b) Nur Spiele mit ausreichender Mindestanzahl an Einträgen berücksichtigen
# ------------------------------------------------------------
game_counts = df["game"].value_counts()
valid_games = game_counts[game_counts >= min_entries_per_game].index
df = df[df["game"].isin(valid_games)]

# Bucket bilden: <grenze vs. >=grenze
df["time_bucket"] = np.where(
    df["hours"] >= grenze_hours_played,
    f">={grenze_hours_played}h",
    f"<{grenze_hours_played}h",
)

# ------------------------------------------------------------
# 3) Absolute Häufigkeiten je Spiel und Bucket aggregieren
# ------------------------------------------------------------
abs_counts = (
    df.groupby(["game", "time_bucket"])
    .size()
    .unstack(fill_value=0)
    .sort_values(
        by=[f">={grenze_hours_played}h", f"<{grenze_hours_played}h"],
        ascending=False,
    )
)

for col in [f">={grenze_hours_played}h", f"<{grenze_hours_played}h"]:
    if col not in abs_counts.columns:
        abs_counts[col] = 0
abs_counts = abs_counts[[f">={grenze_hours_played}h", f"<{grenze_hours_played}h"]]

abs_counts_df = abs_counts.copy()

# ------------------------------------------------------------
# 4) Relative Häufigkeiten je Spiel
# ------------------------------------------------------------
row_sums = abs_counts_df.sum(axis=1).replace(0, np.nan)
rel_freq = abs_counts_df.div(row_sums, axis=0).fillna(0).head(20)

# ------------------------------------------------------------
# 5) Plot: relative Häufigkeitsverteilung
# ------------------------------------------------------------
num_games = rel_freq.shape[0]
fig_height = max(6, min(0.12 * num_games, 60))  # sinnvolle Höhe

cols = [f">={grenze_hours_played}h", f"<{grenze_hours_played}h"]
rel_sorted = rel_freq.sort_values(by=cols, ascending=False)
totals = abs_counts_df.reindex(rel_sorted.index).sum(axis=1).astype(int)

fig, ax = plt.subplots(figsize=(12, fig_height))

rel_sorted[cols].plot(
    kind="barh",
    stacked=True,
    width=0.9,
    legend=True,
    color=["#004A9F", "#C50F3C"],
    ax=ax,
)

ax.set_xlabel("Relative Häufigkeit")
ax.set_ylabel("Spiel")
ax.invert_yaxis()
ax.legend(title="Zeit-Bucket", loc="lower left")

# --- Beschriftung: rechts außerhalb der Balken ---
# Wir erweitern die x-Achse leicht (z. B. auf 1.15 statt 1.0)
ax.set_xlim(0, 1.15)

y_positions = np.arange(len(rel_sorted.index))
for y, n in zip(y_positions, totals):
    ax.text(1.02, y, f"n={n}", va="center", ha="left")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 6) (Optional) DataFrames für weitere Verwendung
# ------------------------------------------------------------
# - abs_counts_df: absolute Häufigkeiten
# - rel_freq:      relative Häufigkeiten
