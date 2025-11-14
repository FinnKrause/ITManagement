import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import numpy as np

# Daten laden
cols = ["user_id", "game", "action", "hours", "extra"]
df = pd.read_csv("../../steam-200k.csv", names=cols)

# Nur play-Daten
df_play = df[df["action"] == "play"]

# Aggregation
games = df_play.groupby("game").agg(
    num_players=("user_id", "nunique"),
    avg_hours=("hours", "mean"),
    total_hours=("hours", "sum")
).reset_index()

print(games)

# Skalieren mit Log und RobustScaler
# 1) Log reduziert Ausreißer-Effekte
games_log = np.log1p(games[["num_players", "avg_hours"]])

# 2) RobustScaler: weniger anfällig als StandardScaler auf Ausreißer, nutzt Median + IQR
scaler = RobustScaler()
X_scaled = scaler.fit_transform(games_log)

print(X_scaled)


#k-Means : In wie viele Gruppen soll es die Spiele einteilen? 
# -> Dafür prüfen wir k mit Elbow-Plot (= Für jedes k berechnen wir die Summe der Abstände der Punkte zu ihrem Clusterzentrum,
# ab einem bestimmten Punkt bringt ein zusätzliches Cluster kaum noch Verbesserung -> Ellenbogen)
# -> Gegenprüfung: Silhouette-Score (= misst wie gut die Punkte in ihrem Cluster zusammenpassen und wie stark sie von anderen Clustern 
# getrennt sind -> für jedes k berechnet man den Durchschnitts-Silhouette-Score)
# -> dann können wir beide k der beiden Berechnungen vergleichen

inertia_values = []
silhouette_values = []
K = range(2, 11)  # Testet k = 2 bis 10

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)
    silhouette_values.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot: Elbow
plt.plot(K, inertia_values, marker="o")
plt.xlabel("Anzahl der Cluster (k)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow-Methode")
plt.show()

# Plot: Silhouette
plt.plot(K, silhouette_values, marker="o")
plt.xlabel("Anzahl der Cluster (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette-Analyse")
plt.show()

# Ergebnis: k = 4 ist am Besten, da zwischen k=4 und k=5 ungefähr Knick in der Elbow-Methode ist (Silhouette Wert ist vogelwild) -> vllt einfach nicht überprüfen falls wir diese Version Arnold abgeben)

optimal_k = 4

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
games["cluster"] = kmeans_final.fit_predict(X_scaled)

# Kein PCA, weil wir eh nur 2 Variablen haben (ABER ist das dann das was Arnold wollte? Weil jetzt haben wir nur noch 2 variablen, aber an sich macht es mit der korrelation schon sinn)

# 2D-Plot 
plt.figure(figsize=(12, 8))

scatter = plt.scatter(
    games_log["num_players"], 
    games_log["avg_hours"], 
    c=games["cluster"], 
    cmap="viridis"
)

plt.xlabel("log(num_players)")
plt.ylabel("log(avg_hours)")
plt.title("Clustering der Spiele")

# Cluster-Legende
plt.legend(*scatter.legend_elements(), title="Cluster")

# Pro Cluster Top5 Spiele nach Spieleranzahl beschriften
for c in games["cluster"].unique():
    top_games = games[games["cluster"] == c].nlargest(5, "num_players")
    for _, row in top_games.iterrows():
        plt.text(
            np.log1p(row["num_players"]),
            np.log1p(row["avg_hours"]),
            row["game"],
            fontsize=6,
            fontweight="bold",
            alpha=0.9
        )

plt.show()

# Ergebnis: Lila (Wenig Spieler, wenig durchschnittliche Spielzeit), blau (viele Spieler, mittlere bis hohe durschnittliche Spielzeit), 
#            grün (wenige Spieler, hohe durschnittliche spielzeit), gelb (viele spieler, geringe durschnittliche spielzeit)
# FRAGE: darf man es jetzt so überhaupt noch interpretieren oder ist das schon zu viel interpretation, da es nicht mehr die reinen daten sind, sondern mit log und so...??
