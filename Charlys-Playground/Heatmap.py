import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

csv_path = "../steam-200k.csv"
cols = ["user_id", "game", "action", "hours", "extra"]
df = pd.read_csv(csv_path, header=None, names=cols)

# 1. Nur numerische Spalten auswählen
numeric_df = df.select_dtypes(include=["number"])

# 2. Korrelation berechnen
corr = numeric_df.corr()

# 3. Heatmap zeichnen
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Korrelations-Heatmap der numerischen Merkmale")
plt.show()

# das ist hier alles Käse