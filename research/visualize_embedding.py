import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


GCN_PATH = (
    "research/gcn/"
    "gcn_autoencoder_test_embeddings.npy"
)


def visualize_tsne(
    embeddings,
    title,
    save_name
):

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42
    )

    result = tsne.fit_transform(
        embeddings
    )

    plt.figure(
        figsize=(8,6)
    )

    plt.scatter(
        result[:,0],
        result[:,1],
        s=20
    )

    plt.title(
        title
    )

    plt.xlabel(
        "Dimension 1"
    )

    plt.ylabel(
        "Dimension 2"
    )

    plt.grid(
        True
    )

    plt.savefig(
        save_name,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()



def main():

    print(
        "VISUALISASI EMBEDDING GCN"
    )


    embeddings = np.load(
        GCN_PATH
    )


    print(
        "Embedding shape:",
        embeddings.shape
    )


    visualize_tsne(
        embeddings,
        "GCN Autoencoder Embedding (32D to 2D)",
        "research/gcn/gcn_tsne.png"
    )


    print(
        "Selesai"
    )



if __name__ == "__main__":

    main()