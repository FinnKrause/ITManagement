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

