import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Memuat embedding hasil GCN
embedding_path = "research/gcn/all_gcn_embeddings.npy"
label_path = "research/gcn/all_gcn_labels.npy"

embeddings = np.load(embedding_path)
labels = np.load(label_path)

print("GCN CLUSTERING ANALYSIS")
print()
print("Jumlah embedding :", embeddings.shape[0])
print("Dimensi embedding :", embeddings.shape[1])
print()

# Mencoba beberapa jumlah cluster
results = []

for k in range(2, 9):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(embeddings)

    score = silhouette_score(
        embeddings,
        cluster_labels
    )

    results.append((k, score))

    print(
        f"k = {k} | "
        f"Silhouette Score = {score:.4f}"
    )

# Mencari jumlah cluster dengan skor tertinggi
best_k, best_score = max(
    results,
    key=lambda x: x[1]
)

print()
print("HASIL TERBAIK")
print("Jumlah cluster :", best_k)
print("Silhouette Score :", f"{best_score:.4f}")

# Membuat clustering menggunakan k terbaik
kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

final_clusters = kmeans.fit_predict(embeddings)

print()
print("KOMPOSISI CLUSTER")
print()

unique_clusters = np.unique(final_clusters)

for cluster in unique_clusters:
    print(f"Cluster {cluster}:")

    cluster_indices = np.where(
        final_clusters == cluster
    )[0]

    cluster_videos = labels[cluster_indices]

    unique_videos, counts = np.unique(
        cluster_videos,
        return_counts=True
    )

    for video, count in zip(unique_videos, counts):
        print(f"  {video}: {count}")

    print()