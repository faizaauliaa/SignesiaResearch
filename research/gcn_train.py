import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


DATASET_FILE = "research/gcn/temporal_graph_dataset.pt"
MODEL_FILE = "research/gcn/gcn_reconstruction_model.pt"

BATCH_SIZE = 32
HIDDEN_CHANNELS = 64
EMBEDDING_SIZE = 32

EPOCHS = 50
LEARNING_RATE = 0.001


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


print("MEMULAI TRAINING GCN")

dataset = torch.load(
    DATASET_FILE,
    weights_only=False
)

print(f"Jumlah graph : {len(dataset)}")

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

model = GCNReconstructionModel(
    input_channels=4,
    hidden_channels=HIDDEN_CHANNELS,
    embedding_size=EMBEDDING_SIZE
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = nn.MSELoss()


for epoch in range(1, EPOCHS + 1):

    model.train()

    total_loss = 0.0
    total_graphs = 0

    for batch in loader:

        optimizer.zero_grad()

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

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() * batch.num_graphs
        )

        total_graphs += batch.num_graphs

    average_loss = (
        total_loss / total_graphs
    )

    print(
        f"Epoch {epoch:03d}/{EPOCHS} "
        f"| Loss: {average_loss:.6f}"
    )


os.makedirs(
    os.path.dirname(MODEL_FILE),
    exist_ok=True
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_channels": 4,
        "hidden_channels": HIDDEN_CHANNELS,
        "embedding_size": EMBEDDING_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE
    },
    MODEL_FILE
)

print()
print("TRAINING GCN SELESAI")
print(f"Model disimpan : {MODEL_FILE}")