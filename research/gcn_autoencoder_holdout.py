import os
import numpy as np
import pandas as pd
import torch

from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv


WINDOW_SIZE = 5
NUM_LANDMARKS = 21
NUM_FEATURES = 4
EMBEDDING_DIM = 32

HIDDEN_DIM = 32
NODE_EMBED_DIM = 16

EPOCHS = 50
LEARNING_RATE = 0.001

DATA_DIR = "research/optimized"

TRAIN_VIDEOS = [
    "a",
    "abjad",
    "acak",
    "adab"
]

TEST_VIDEO = "adaptasi"

FEATURE_TYPES = [
    "relative_x",
    "relative_y",
    "delta_x",
    "delta_y"
]

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def load_video_data(video_name):

    path = os.path.join(
        DATA_DIR,
        f"{video_name}_optimized.csv"
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"File tidak ditemukan: {path}"
        )

    df = pd.read_csv(path)

    temporal_columns = []

    for feature in [
        "delta_x",
        "delta_y"
    ]:

        temporal_columns.extend(
            [
                f"{feature}_{i}"
                for i in range(NUM_LANDMARKS)
            ]
        )

    df[temporal_columns] = (
        df[temporal_columns]
        .fillna(0.0)
    )

    numeric_values = (
        df.select_dtypes(
            include=[np.number]
        ).to_numpy()
    )

    if not np.isfinite(
        numeric_values
    ).all():

        raise ValueError(
            f"NaN atau Inf ditemukan "
            f"pada {video_name}"
        )

    return df


def create_frame_features(row):

    features = []

    for landmark in range(
        NUM_LANDMARKS
    ):

        node_features = []

        for feature_type in FEATURE_TYPES:

            value = row[
                f"{feature_type}_{landmark}"
            ]

            node_features.append(
                float(value)
            )

        features.append(
            node_features
        )

    return np.array(
        features,
        dtype=np.float32
    )


def create_spatial_edges():

    edges = []

    skeleton_edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20)
    ]

    for source, target in skeleton_edges:

        edges.append(
            [source, target]
        )

        edges.append(
            [target, source]
        )

    return edges


def create_temporal_edges():

    edges = []

    for frame_idx in range(
        WINDOW_SIZE - 1
    ):

        current_start = (
            frame_idx * NUM_LANDMARKS
        )

        next_start = (
            (frame_idx + 1)
            * NUM_LANDMARKS
        )

        for landmark in range(
            NUM_LANDMARKS
        ):

            source = (
                current_start
                + landmark
            )

            target = (
                next_start
                + landmark
            )

            edges.append(
                [source, target]
            )

            edges.append(
                [target, source]
            )

    return edges


def create_graph_from_window(
    window,
    video_name,
    hand,
    start_frame
):

    frame_features = []

    for _, row in window.iterrows():

        features = create_frame_features(
            row
        )

        frame_features.append(
            features
        )

    x = np.concatenate(
        frame_features,
        axis=0
    )

    spatial_edges = []

    base_spatial_edges = (
        create_spatial_edges()
    )

    for frame_idx in range(
        WINDOW_SIZE
    ):

        offset = (
            frame_idx
            * NUM_LANDMARKS
        )

        for source, target in base_spatial_edges:

            spatial_edges.append(
                [
                    source + offset,
                    target + offset
                ]
            )

    temporal_edges = (
        create_temporal_edges()
    )

    all_edges = (
        spatial_edges
        +
        temporal_edges
    )

    edge_index = torch.tensor(
        all_edges,
        dtype=torch.long
    ).t().contiguous()

    x_tensor = torch.tensor(
        x,
        dtype=torch.float32
    )

    y_tensor = x_tensor.reshape(
        -1
    )

    graph = Data(
        x=x_tensor,
        edge_index=edge_index,
        y=y_tensor
    )

    graph.video = video_name
    graph.hand_name = str(hand)
    graph.start_frame = int(
        start_frame
    )

    return graph


def create_graph_dataset(
    video_name
):

    df = load_video_data(
        video_name
    )

    graphs = []

    for hand, hand_df in df.groupby("hand"):

        hand_df = (
            hand_df
            .sort_values("frame")
            .reset_index(drop=True)
        )

        for start in range(
            len(hand_df)
            - WINDOW_SIZE
            + 1
        ):

            window = hand_df.iloc[
                start:start + WINDOW_SIZE
            ]

            frames = (
                window["frame"]
                .to_numpy()
            )

            expected_frames = np.arange(
                frames[0],
                frames[0] + WINDOW_SIZE
            )

            if not np.array_equal(
                frames,
                expected_frames
            ):
                continue

            graph = create_graph_from_window(
                window,
                video_name,
                hand,
                frames[0]
            )

            graphs.append(
                graph
            )

    return graphs


class GCNEncoderDecoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = GCNConv(
            NUM_FEATURES,
            HIDDEN_DIM
        )

        self.conv2 = GCNConv(
            HIDDEN_DIM,
            NODE_EMBED_DIM
        )

        self.encoder = nn.Sequential(
            nn.Linear(
                105 * NODE_EMBED_DIM,
                128
            ),
            nn.ReLU(),
            nn.Linear(
                128,
                EMBEDDING_DIM
            )
        )

        self.decoder = nn.Sequential(
            nn.Linear(
                EMBEDDING_DIM,
                128
            ),
            nn.ReLU(),
            nn.Linear(
                128,
                420
            )
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

        x = torch.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = torch.relu(x)

        # Setiap graph memiliki 105 node.
        # Flatten mempertahankan urutan node
        # sehingga informasi landmark dan waktu
        # tetap tersedia sebelum bottleneck 32D.
        x = x.reshape(
            -1,
            105 * NODE_EMBED_DIM
        )

        embedding = self.encoder(
            x
        )

        reconstruction = self.decoder(
            embedding
        )

        return (
            embedding,
            reconstruction
        )


def evaluate_model(
    model,
    loader
):

    model.eval()

    all_true = []
    all_pred = []

    embeddings = []

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(DEVICE)

            embedding, prediction = model(
                batch.x,
                batch.edge_index,
                batch.batch
            )

            batch_size = prediction.shape[0]

            true_values = batch.y.reshape(
                batch_size,
                -1
            )

            all_true.append(
                true_values.cpu()
            )

            all_pred.append(
                prediction.cpu()
            )

            embeddings.append(
                embedding.cpu()
            )

    true_values = torch.cat(
        all_true,
        dim=0
    ).numpy()

    predicted_values = torch.cat(
        all_pred,
        dim=0
    ).numpy()

    embedding_values = torch.cat(
        embeddings,
        dim=0
    ).numpy()

    mae = np.mean(
        np.abs(
            true_values
            - predicted_values
        )
    )

    rmse = np.sqrt(
        np.mean(
            (
                true_values
                - predicted_values
            ) ** 2
        )
    )

    return (
        mae,
        rmse,
        embedding_values
    )


def main():

    print("GCN AUTOENCODER HOLD-OUT")
    print(
        f"Device : {DEVICE}"
    )
    print()

    train_graphs = []
    test_graphs = []

    print("MEMBUAT DATASET GRAPH")
    print()

    for video_name in TRAIN_VIDEOS:

        print(
            f"Memproses training: "
            f"{video_name}.mp4"
        )

        graphs = create_graph_dataset(
            video_name
        )

        print(
            f"  Jumlah graph: "
            f"{len(graphs)}"
        )

        train_graphs.extend(
            graphs
        )

    print()

    print(
        f"Memproses testing: "
        f"{TEST_VIDEO}.mp4"
    )

    test_graphs = create_graph_dataset(
        TEST_VIDEO
    )

    print(
        f"  Jumlah graph: "
        f"{len(test_graphs)}"
    )

    print()
    print("PEMBAGIAN DATASET")

    print(
        f"Training graph : "
        f"{len(train_graphs)}"
    )

    print(
        f"Testing graph  : "
        f"{len(test_graphs)}"
    )

    train_loader = DataLoader(
        train_graphs,
        batch_size=16,
        shuffle=True
    )

    test_loader = DataLoader(
        test_graphs,
        batch_size=16,
        shuffle=False
    )

    model = GCNEncoderDecoder().to(
        DEVICE
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.MSELoss()

    print()
    print("MEMULAI TRAINING")
    print()

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        total_loss = 0.0

        for batch in train_loader:

            batch = batch.to(
                DEVICE
            )

            optimizer.zero_grad()

            _, reconstruction = model(
                batch.x,
                batch.edge_index,
                batch.batch
            )

            loss = criterion(
                reconstruction,
                batch.y.reshape(
                    reconstruction.shape
                )
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * batch.num_graphs
            )

        epoch_loss = (
            total_loss
            /
            len(train_graphs)
        )

        if (
            epoch == 1
            or epoch % 10 == 0
        ):

            print(
                f"Epoch {epoch:03d}/{EPOCHS} "
                f"| Loss: {epoch_loss:.8f}"
            )

    print()
    print(
        "TRAINING SELESAI"
    )

    model_path = (
        "research/gcn/"
        "gcn_autoencoder_holdout.pt"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    print(
        f"Model disimpan : "
        f"{model_path}"
    )

    print()
    print(
        "MENGUJI PADA VIDEO "
        "YANG TIDAK DILATIH"
    )

    mae, rmse, embeddings = (
        evaluate_model(
            model,
            test_loader
        )
    )

    print()
    print("HASIL RECONSTRUCTION")
    print(
        f"Video testing : "
        f"{TEST_VIDEO}.mp4"
    )

    print(
        f"Jumlah graph  : "
        f"{len(test_graphs)}"
    )

    print(
        f"Embedding shape : "
        f"{embeddings.shape}"
    )

    print(
        f"MAE  : "
        f"{mae:.8f}"
    )

    print(
        f"RMSE : "
        f"{rmse:.8f}"
    )

    embedding_path = (
        "research/gcn/"
        "gcn_autoencoder_test_embeddings.npy"
    )

    np.save(
        embedding_path,
        embeddings
    )

    print()
    print(
        f"Embedding disimpan : "
        f"{embedding_path}"
    )

    print()
    print(
        "GCN AUTOENCODER HOLD-OUT SELESAI"
    )


if __name__ == "__main__":

    main()