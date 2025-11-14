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
games_log = np.log1p(games[["num_players", "avg_hours", "total_hours"]])

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

# Ergebnis: k = 4 ist am Besten, da zwischen k=4 und k=5 ungefähr Knick in der Elbow-Methode ist (Silhouette Wert ist überall schlecht (2 am besten, daher k=4) -> vllt einfach nicht überprüfen falls wir diese Version Arnold abgeben)

optimal_k = 4

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
games["cluster"] = kmeans_final.fit_predict(X_scaled)

# PCA: Wir haben 3 Dimensionen (num_players, avg_hours, total_hours) und PCA komprimiert diese 3 Dimensionen auf 2 Dimensionen
# Dies macht es so indem es eine neue Achse versucht zu finden, die die Richtung maximaler Variation zeigt, also dort, wo sich die 
# Daten am stärksten unterscheiden -> PCA1. Dann sucht PCA eine zweite Achse (senkrecht zu PCA1) die möglichst viel der restlichen Varianz erklärt -> PCA2

# PCA berechnen
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)

games["PCA1"] = pca_components[:, 0]
games["PCA2"] = pca_components[:, 1]

# Wichtige Spiele auswählen
important_games_list = []

for c in games ["cluster"].unique():
    top_c = games[games["cluster"] ==c].nlargest(5, "num_players")
    important_games_list.append(top_c)

important_games = pd.concat(important_games_list)


print(pca.components_)
# Ergebnis:
# [[ 0.4891868   0.5919779   0.6405142 ]
# [ 0.71274683 -0.69459669  0.09760836]]
# Bedeutung: PCA1 =  0.489 * num_player + 0.592 * avg_hours + 0.64 * total_hours -> alle ungefähr gleich gewichtet
#           PCA2 = 0.713 * num_players - 0.695 * avg_hours + 0.098 * total_hours -> PCA2 misst fast ausschließlich num_players

print(pca.explained_variance_ratio_)
# Ergebnis: [[0.80724289 0.19202489]
# Bedeutung: PCA1 erklärt 80.72% der Gesamt-Information (Varianz)
#           PCA2 erklärt 19.20%
# Unsere 2 PCA-Komponenten bewahren 99.92% der gesamten Information aus den 3 originalen Features. -> fast kein Informationsverlust


# 2D-Plot 
plt.figure(figsize=(12, 8))

scatter = plt.scatter(
    games["PCA1"], 
    games["PCA2"], 
    c=games["cluster"], 
    cmap="viridis"
)

plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("PCA der Spiele (2D reduziert)")

# Cluster-Legende
plt.legend(*scatter.legend_elements(), title="Cluster")

# Spielenamen reinschreiben
for _, row in important_games.iterrows():
    plt.text(row["PCA1"], row["PCA2"], row["game"], fontsize=6, alpha=0.7)

plt.show()

# Ergebnis: Man hat eine Linie in dem Plot. Liegt daran, dass Features extrem korrelieren (total_hours = num_players x avg_hours)             