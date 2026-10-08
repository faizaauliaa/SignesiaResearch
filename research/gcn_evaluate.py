import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


DATASET_FILE = "research/gcn/temporal_graph_dataset.pt"
MODEL_FILE = "research/gcn/gcn_reconstruction_model.pt"
OUTPUT_FILE = "research/gcn/trained_gcn_embeddings.pt"

BATCH_SIZE = 32


class GCNReconstructionModel(nn.Module):

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

        self.decoder = nn.Linear(
            embedding_size,
            input_channels
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

        graph_embedding = global_mean_pool(
            x,
            batch
        )

        embedding = self.embedding(
            graph_embedding
        )

        reconstruction = self.decoder(
            embedding
        )

        return embedding, reconstruction


print("EVALUASI GCN TERLATIH")

dataset = torch.load(
    DATASET_FILE,
    weights_only=False
)

checkpoint = torch.load(
    MODEL_FILE,
    weights_only=False
)

model = GCNReconstructionModel(
    input_channels=checkpoint["input_channels"],
    hidden_channels=checkpoint["hidden_channels"],
    embedding_size=checkpoint["embedding_size"]
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

criterion = nn.MSELoss()

all_embeddings = []
all_metadata = []

total_loss = 0.0
total_graphs = 0

with torch.no_grad():

    for batch in loader:

        embedding, reconstruction = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        target = global_mean_pool(
            batch.x,
            batch.batch
        )

        loss = criterion(
            reconstruction,
            target
        )

        total_loss += (
            loss.item() * batch.num_graphs
        )

        total_graphs += batch.num_graphs

        all_embeddings.append(
            embedding
        )

        for i in range(
            len(batch.video)
        ):
            all_metadata.append({
                "video": batch.video[i],
                "hand": batch.hand[i],
                "start_frame": int(
                    batch.start_frame[i]
                ),
                "end_frame": int(
                    batch.end_frame[i]
                )
            })

average_loss = (
    total_loss / total_graphs
)

embeddings = torch.cat(
    all_embeddings,
    dim=0
)

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

torch.save(
    {
        "embeddings": embeddings,
        "metadata": all_metadata,
        "reconstruction_loss": average_loss
    },
    OUTPUT_FILE
)

print()
print("EVALUASI SELESAI")
print(f"Jumlah graph        : {total_graphs}")
print(f"Embedding shape     : {embeddings.shape}")
print(f"Reconstruction loss : {average_loss:.8f}")
print(f"Output              : {OUTPUT_FILE}")