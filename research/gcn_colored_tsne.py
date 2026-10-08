import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


EMBEDDING_PATH = (
    "research/gcn/"
    "all_gcn_embeddings.npy"
)

LABEL_PATH = (
    "research/gcn/"
    "all_gcn_labels.npy"
)



def main():

    embeddings = np.load(
        EMBEDDING_PATH
    )

    labels = np.load(
        LABEL_PATH
    )


    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30
    )


    result = tsne.fit_transform(
        embeddings
    )


    plt.figure(
        figsize=(10,7)
    )


    unique_labels = np.unique(
        labels
    )


    for label in unique_labels:

        idx = labels == label

        plt.scatter(
            result[idx,0],
            result[idx,1],
            label=label,
            s=25
        )


    plt.title(
        "GCN Autoencoder Embedding Space"
    )


    plt.xlabel(
        "Dimension 1"
    )

    plt.ylabel(
        "Dimension 2"
    )


    plt.legend()

    plt.grid(
        True
    )


    plt.savefig(
        "research/gcn/gcn_colored_tsne.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.show()



if __name__ == "__main__":

    main()