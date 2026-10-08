import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


EMBEDDING_PATH = (
    "research/gcn/"
    "gcn_autoencoder_test_embeddings.npy"
)


def calculate_motion_change():

    # ambil ulang data adaptasi
    from gcn_autoencoder_holdout import (
        create_graph_dataset
    )

    graphs = create_graph_dataset(
        "adaptasi"
    )

    vectors = []

    for graph in graphs:

        vectors.append(
            graph.y.numpy()
        )

    vectors = np.array(
        vectors
    )

    changes = []

    for i in range(
        len(vectors)-1
    ):

        diff = np.linalg.norm(
            vectors[i+1]
            -
            vectors[i]
        )

        changes.append(
            diff
        )

    return np.array(
        changes
    )


def calculate_embedding_change():

    embeddings = np.load(
        EMBEDDING_PATH
    )

    changes = []

    for i in range(
        len(embeddings)-1
    ):

        diff = np.linalg.norm(
            embeddings[i+1]
            -
            embeddings[i]
        )

        changes.append(
            diff
        )

    return np.array(
        changes
    )


def main():

    print(
        "GCN AUTOENCODER MOTION EVALUATION"
    )
    print()

    motion_change = (
        calculate_motion_change()
    )

    embedding_change = (
        calculate_embedding_change()
    )

    length = min(
        len(motion_change),
        len(embedding_change)
    )

    motion_change = (
        motion_change[:length]
    )

    embedding_change = (
        embedding_change[:length]
    )


    pearson, pearson_p = pearsonr(
        motion_change,
        embedding_change
    )

    spearman, spearman_p = spearmanr(
        motion_change,
        embedding_change
    )


    print(
        f"Jumlah pasangan window : {length}"
    )

    print()

    print(
        f"Mean motion change     : "
        f"{motion_change.mean():.6f}"
    )

    print(
        f"Mean embedding change  : "
        f"{embedding_change.mean():.6f}"
    )

    print()

    print(
        f"Pearson correlation    : "
        f"{pearson:.6f}"
    )

    print(
        f"Pearson p-value        : "
        f"{pearson_p:.6f}"
    )

    print()

    print(
        f"Spearman correlation  : "
        f"{spearman:.6f}"
    )

    print(
        f"Spearman p-value       : "
        f"{spearman_p:.6f}"
    )

    print()

    print(
        "Evaluasi selesai."
    )


if __name__ == "__main__":

    main()