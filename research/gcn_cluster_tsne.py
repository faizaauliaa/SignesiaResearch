import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# Memuat embedding GCN
embedding_path = "research/gcn/all_gcn_embeddings.npy"

embeddings = np.load(embedding_path)

print("GCN CLUSTER t-SNE")
print()
print("Jumlah embedding :", embeddings.shape[0])
print("Dimensi embedding :", embeddings.shape[1])

# K-Means dengan 3 cluster
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(embeddings)

# t-SNE
tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30
)

embedding_2d = tsne.fit_transform(embeddings)

# Visualisasi
plt.figure(figsize=(10, 7))

for cluster in range(3):
    mask = cluster_labels == cluster

    plt.scatter(
        embedding_2d[mask, 0],
        embedding_2d[mask, 1],
        label=f"Cluster {cluster}",
        alpha=0.7,
        s=30
    )

plt.title("t-SNE Visualization of GCN Embeddings by Cluster")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend()
plt.tight_layout()

output_path = "research/gcn/gcn_cluster_tsne.png"

plt.savefig(
    output_path,
    dpi=300
)

plt.show()

print()
print("Visualisasi selesai.")
print("Output :", output_path)