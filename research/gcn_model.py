import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader


DATASET_FILE = "research/gcn/temporal_graph_dataset.pt"
OUTPUT_FILE = "research/gcn/gcn_embeddings.pt"

BATCH_SIZE = 32
HIDDEN_CHANNELS = 64
EMBEDDING_SIZE = 32


class GCNEncoder(nn.Module):

    def __init__(
        self,
        input_channels,
        hidden_channels,
        embedding_size
    ):
        super().__init__()

        self.conv1 = GCNConv(
            input_channels,
            hidden_channels
        )

        self.conv2 = GCNConv(
            hidden_channels,
            hidden_channels
        )

        self.embedding = nn.Linear(
            hidden_channels,
            embedding_size
        )

    def forward(
        self,
        x,
        edge_index,
        batch
    ):
        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = F.relu(x)

        x = global_mean_pool(
            x,
            batch
        )

        x = self.embedding(x)

        return x


print("MEMBUAT GCN ENCODER")

dataset = torch.load(
    DATASET_FILE,
    weights_only=False
)

print(f"Jumlah graph : {len(dataset)}")

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = GCNEncoder(
    input_channels=4,
    hidden_channels=HIDDEN_CHANNELS,
    embedding_size=EMBEDDING_SIZE
)

model.eval()

embeddings = []
metadata = []

with torch.no_grad():

    for batch in loader:

        output = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        embeddings.append(
            output
        )

        for i in range(
            len(batch.video)
        ):
            metadata.append({
                "video": batch.video[i],
                "hand": batch.hand[i],
                "start_frame": int(
                    batch.start_frame[i]
                ),
                "end_frame": int(
                    batch.end_frame[i]
                )
            })


embeddings = torch.cat(
    embeddings,
    dim=0
)

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

torch.save(
    {
        "embeddings": embeddings,
        "metadata": metadata
    },
    OUTPUT_FILE
)

print()
print("GCN ENCODER SELESAI")
print(f"Embedding shape : {embeddings.shape}")
print(f"Output           : {OUTPUT_FILE}")