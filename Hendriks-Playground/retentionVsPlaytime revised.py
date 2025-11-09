import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Daten einlesen
# =========================
df = pd.read_csv(
    "steam-200k.csv", header=None, names=["UserID", "Game", "Action", "Value", "Other"]
)

# =========================
# Daten vorbereiten
# =========================
purchases = df[df["Action"] == "purchase"][["UserID", "Game"]]
play_data = df[df["Action"] == "play"][["UserID", "Game", "Value"]]

games_analysis = []
for game in purchases["Game"].unique():
    buyers = purchases[purchases["Game"] == game]["UserID"].nunique()
    players = play_data[play_data["Game"] == game]["UserID"].nunique()
    retention_rate = (players / buyers * 100) if buyers > 0 else 0
    avg_playtime = (
        play_data[play_data["Game"] == game]["Value"].mean() if players > 0 else 0
    )

    games_analysis.append(
        {
            "game": game,
            "retention_rate": retention_rate,
            "avg_playtime": avg_playtime,  # Minuten
            "buyers": buyers,
            "players": players,
        }
    )

games_df = pd.DataFrame(games_analysis)
filtered_games = games_df[games_df["buyers"] >= 100].copy()

# =========================
# Spiele mit >150h Ø-Spielzeit (9000 Min.) anzeigen
# =========================
high_playtime = filtered_games[filtered_games["avg_playtime"] > 9000].sort_values(
    "avg_playtime", ascending=False
)

print("\n=== Spiele mit >150 Stunden Ø-Spielzeit (nur Spiele mit ≥100 Käufern) ===")
if high_playtime.empty:
    print("Keine Spiele über 150h gefunden.")
else:
    for _, row in high_playtime.iterrows():
        print(f"{row['game']} — {row['avg_playtime'] / 60:.1f}h")


# =========================
# Kategorien für Retention definieren
# =========================
def categorize_retention(rate):
    if rate < 40:
        return "Niedrig (<40%)"
    elif rate <= 90:
        return "Mittel (40-90%)"
    else:
        return "Hoch (>90%)"


filtered_games["retention_category"] = filtered_games["retention_rate"].apply(
    categorize_retention
)
colors = {"Niedrig (<40%)": "red", "Mittel (40-90%)": "orange", "Hoch (>90%)": "green"}

# =========================
# Diagramm erstellen
# =========================
plt.figure(figsize=(10, 6))

median_playtime = filtered_games["avg_playtime"].median()
median_retention = filtered_games["retention_rate"].median()

plt.axhline(y=median_retention, color="gray", linestyle="--", alpha=0.7)
plt.axvline(x=median_playtime, color="gray", linestyle="--", alpha=0.7)

for category, color in colors.items():
    category_data = filtered_games[filtered_games["retention_category"] == category]
    plt.scatter(
        category_data["avg_playtime"],
        category_data["retention_rate"],
        s=category_data["buyers"] / 50,
        alpha=0.7,
        color=color,
        label=category,
        edgecolors="black",
        linewidth=0.5,
    )

# Beschriftung der Punkte mit >150h
for _, row in high_playtime.iterrows():
    plt.text(
        row["avg_playtime"],
        row["retention_rate"] + 1,  # leicht über dem Punkt
        row["game"],
        fontsize=8,
        ha="left",
        va="bottom",
        color="black",
    )

plt.xlabel("Durchschnittliche Spielzeit (Minuten)")
plt.ylabel("Retention Rate (%)")
plt.title(
    "Spiele-Performance: Die 4 Erfolgs-Quadranten", fontsize=14, fontweight="bold"
)
plt.grid(True, alpha=0.3)
plt.ylim(0, 100)

# Trendlinie
z = np.polyfit(filtered_games["avg_playtime"], filtered_games["retention_rate"], 1)
p = np.poly1d(z)
plt.plot(
    filtered_games["avg_playtime"],
    p(filtered_games["avg_playtime"]),
    "k--",
    alpha=0.5,
    label="Trendlinie",
)

plt.legend()
plt.tight_layout()
plt.show()

# =========================
# Quadrantenanalyse
# =========================
filtered_games["quadrant"] = np.where(
    (filtered_games["retention_rate"] >= median_retention)
    & (filtered_games["avg_playtime"] >= median_playtime),
    "Oben Rechts",
    np.where(
        (filtered_games["retention_rate"] >= median_retention)
        & (filtered_games["avg_playtime"] < median_playtime),
        "Oben Links",
        np.where(
            (filtered_games["retention_rate"] < median_retention)
            & (filtered_games["avg_playtime"] >= median_playtime),
            "Unten Rechts",
            "Unten Links",
        ),
    ),
)

print("\n=== Verteilung der Spiele auf die 4 Quadranten ===")
quadrant_summary = (
    filtered_games.groupby("quadrant")
    .size()
    .reindex(["Oben Rechts", "Oben Links", "Unten Rechts", "Unten Links"], fill_value=0)
)
print(quadrant_summary)
