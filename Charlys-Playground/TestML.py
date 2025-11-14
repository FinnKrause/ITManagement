
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# 1️⃣ Datensatz einlesen
csv_path = "../steam-200k.csv" 
cols = ["user_id", "game", "action", "hours", "extra"]
df = pd.read_csv(csv_path, header=None, names=cols)
 
# =======================
# 2️⃣ 'extra'-Spalte entfernen (nur Nullen)
# =======================
df = df.drop(columns=["extra"])

# =======================
# 3️⃣ Stunden bereinigen
# =======================
# 'action' = 'play' → echte Stunden aus 'hours'
# 'purchase' → 0 Stunden
df["hours_clean"] = df.apply(lambda row: row["hours"] if row["action"] == "play" else 0, axis=1)

# =======================
# 4️⃣ Spielnamen numerisch kodieren
# =======================
le_game = LabelEncoder()
df["game_encoded"] = le_game.fit_transform(df["game"])

# =======================
# 5️⃣ Spieler x Spiel Matrix erstellen
# =======================
# Zeilen = Spieler, Spalten = Spiele, Werte = Stunden
player_game_matrix = df.pivot_table(index="user_id",
                                    columns="game_encoded",
                                    values="hours_clean",
                                    fill_value=0)

# =======================
# 6️⃣ Optional: Stunden standardisieren
# =======================
scaler = StandardScaler()
player_game_matrix_scaled = pd.DataFrame(scaler.fit_transform(player_game_matrix),
                                         index=player_game_matrix.index,
                                         columns=player_game_matrix.columns)

# =======================
# 7️⃣ KNN auf Spieler anwenden
# =======================
knn = NearestNeighbors(n_neighbors=5, metric='cosine')  # Cosine similarity für ähnliches Spielverhalten
knn.fit(player_game_matrix_scaled)

# =======================
# 8️⃣ Funktion: Empfehlungen für einen Spieler
# =======================
def recommend_games(player_id, top_k_neighbors=5):
    distances, indices = knn.kneighbors([player_game_matrix_scaled.loc[player_id]])
    similar_players = player_game_matrix_scaled.index[indices.flatten()[1:]]  # erste ist Spieler selbst

    # Spiele der Nachbarn, die der Spieler noch nicht gespielt hat
    player_games = player_game_matrix.loc[player_id] > 0
    recommended_games = set()
    for neighbor in similar_players:
        neighbor_games = player_game_matrix.loc[neighbor] > 0
        recs = neighbor_games.index[neighbor_games & ~player_games]
        recommended_games.update(recs)

    # In Spielnamen zurückwandeln
    recommended_game_names = le_game.inverse_transform(list(recommended_games))
    return recommended_game_names

# =======================
# 9️⃣ Beispiel: Empfehlungen für Spieler 1
# =======================
player_id = player_game_matrix.index[0]  # z.B. erster Spieler
print("Empfohlene Spiele für Spieler", player_id, ":", recommend_games(player_id))

X = player_game_matrix_scaled  # skaliert

# =======================
# 2️⃣ K-Means Clustering
# =======================
kmeans = KMeans(n_clusters=3, random_state=42)  # 3 Gruppen, kann angepasst werden
clusters = kmeans.fit_predict(X)
player_game_matrix_scaled["Cluster"] = clusters

# =======================
# 3️⃣ Visualisierung: Spielstunden vs Spielname (erste 2 Dimensionen)
# =======================
plt.figure(figsize=(10,6))

# PCA kann helfen, Dimensionen auf 2D zu reduzieren
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.drop(columns="Cluster"))

plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap="viridis", alpha=0.7)
plt.xlabel("PCA-Komponente 1")
plt.ylabel("PCA-Komponente 2")
plt.title("Spielergruppen nach Spielverhalten (K-Means Clustering)")
plt.colorbar(label="Cluster")
plt.show()