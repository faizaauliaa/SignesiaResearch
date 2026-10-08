import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr


DATASET_FILE = "research/gcn/temporal_graph_dataset.pt"
EMBEDDING_FILE = "research/gcn/trained_gcn_embeddings.pt"


print("EVALUASI TEMPORAL GCN")

dataset = torch.load(
    DATASET_FILE,
    weights_only=False
)

embedding_data = torch.load(
    EMBEDDING_FILE,
    weights_only=False
)

embeddings = embedding_data["embeddings"]
metadata = embedding_data["metadata"]

print(f"Jumlah graph    : {len(dataset)}")
print(f"Jumlah embedding: {len(embeddings)}")
print(f"Dimensi embedding: {embeddings.shape[1]}")


motion_values = []
embedding_values = []


for i in range(
    len(dataset) - 1
):

    current_graph = dataset[i]
    next_graph = dataset[i + 1]

    current_meta = metadata[i]
    next_meta = metadata[i + 1]

    same_video = (
        current_meta["video"]
        == next_meta["video"]
    )

    same_hand = (
        current_meta["hand"]
        == next_meta["hand"]
    )

    consecutive_window = (
        next_meta["start_frame"]
        == current_meta["start_frame"] + 1
    )

    if not (
        same_video
        and same_hand
        and consecutive_window
    ):
        continue

    current_x = current_graph.x.numpy()
    next_x = next_graph.x.numpy()

    current_embedding = embeddings[
        i
    ].numpy()

    next_embedding = embeddings[
        i + 1
    ].numpy()

    motion_change = np.mean(
        np.linalg.norm(
            next_x - current_x,
            axis=1
        )
    )

    embedding_change = np.linalg.norm(
        next_embedding
        - current_embedding
    )

    motion_values.append(
        motion_change
    )

    embedding_values.append(
        embedding_change
    )


motion_values = np.array(
    motion_values
)

embedding_values = np.array(
    embedding_values
)


print()
print("HASIL EVALUASI")

print(
    f"Jumlah pasangan window : "
    f"{len(motion_values)}"
)

if len(motion_values) > 1:

    pearson_r, pearson_p = pearsonr(
        motion_values,
        embedding_values
    )

    spearman_r, spearman_p = spearmanr(
        motion_values,
        embedding_values
    )

    print()
    print(
        f"Mean motion change     : "
        f"{motion_values.mean():.6f}"
    )

    print(
        f"Mean embedding change  : "
        f"{embedding_values.mean():.6f}"
    )

    print()
    print(
        f"Pearson correlation    : "
        f"{pearson_r:.6f}"
    )

    print(
        f"Pearson p-value        : "
        f"{pearson_p:.6f}"
    )

    print()
    print(
        f"Spearman correlation   : "
        f"{spearman_r:.6f}"
    )

    print(
        f"Spearman p-value       : "
        f"{spearman_p:.6f}"
    )

print()
print("Evaluasi temporal selesai.")