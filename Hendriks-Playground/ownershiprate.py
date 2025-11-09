import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSV laden
df = pd.read_csv("steam-200k.csv", header=None, names=["user_id", "game", "behavior", "hours", "zero"])

#zero spalte Eliminieren, durch ganzen Datensatz durch nur 0 Werte --> Redundant
df = df.drop(columns=["zero"])

#nur kaufwerte berücksichtigen
purchases = df[df["behavior"] == "purchase"]

#ownershiprate
ownership_counts = purchases["game"].value_counts().head(20)

#diagramm
plt.figure(figsize=(10, 6))
ownership_counts.sort_values().plot(kind="barh")
plt.title("Top 20 Spiele mit der höchsten Ownership-Rate")
plt.xlabel("Anzahl der Besitzer (Käufe)")
plt.ylabel("Spielname")
plt.ticklabel_format(style='plain', axis='x')
plt.tight_layout()
plt.show()