import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from scipy.stats import pearsonr, spearmanr


WINDOW_SIZE = 5

TEST_VIDEO = "adaptasi"

EPOCHS = 50
LEARNING_RATE = 0.001
HIDDEN_DIM = 64
EMBEDDING_DIM = 32

RANDOM_SEED = 42

OPTIMIZED_DIR = "research/optimized"

MODEL_OUTPUT = "research/gcn/holdout_gcn_model.pt"
EMBEDDING_OUTPUT = "research/gcn/holdout_test_embeddings.pt"


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


SPATIAL_EDGES = [
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


def load_video_csv(video_name):

    path = os.path.join(
        OPTIMIZED_DIR,
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
                for i in range(21)
            ]
        )

        df[temporal_columns] = (
        df[temporal_columns]
        .fillna(0.0)
    )

    numeric_values = df.select_dtypes(
        include=[np.number]
    ).to_numpy()

    if not np.isfinite(
        numeric_values
    ).all():

        raise ValueError(
            f"Masih terdapat NaN atau Inf "
            f"pada data {video_name}"
        )

    required_columns = [
        "frame",
        "hand"
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Kolom tidak ditemukan pada {path}: {missing}"
        )

    return df

def get_feature_columns(df):

    feature_columns = []

    for feature in [
        "relative_x",
        "relative_y",
        "delta_x",
        "delta_y"
    ]:

        columns = [
            f"{feature}_{i}"
            for i in range(21)
        ]

        missing = [
            col
            for col in columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Fitur {feature} tidak lengkap. "
                f"Kolom yang hilang: {missing}"
            )

        feature_columns.append(columns)

    return feature_columns


def create_temporal_graphs(df, video_name):

    graphs = []

    feature_columns = get_feature_columns(df)

    relative_x_cols = feature_columns[0]
    relative_y_cols = feature_columns[1]
    delta_x_cols = feature_columns[2]
    delta_y_cols = feature_columns[3]

    df = df.sort_values(
        ["hand", "frame"]
    ).reset_index(drop=True)

    for hand in df["hand"].dropna().unique():

        hand_df = df[
            df["hand"] == hand
        ].copy()

        frames = sorted(
            hand_df["frame"].unique()
        )

        for i in range(
            len(frames) - WINDOW_SIZE + 1
        ):

            window_frames = frames[
                i:i + WINDOW_SIZE
            ]

            expected_frames = list(
                range(
                    window_frames[0],
                    window_frames[0] + WINDOW_SIZE
                )
            )

            if window_frames != expected_frames:
                continue

            window_df = hand_df[
                hand_df["frame"].isin(window_frames)
            ].copy()

            if len(window_df) != WINDOW_SIZE:
                continue

            nodes = []
            node_lookup = {}

            node_index = 0

            for frame in window_frames:

                frame_rows = window_df[
                    window_df["frame"] == frame
                ]

                if len(frame_rows) != 1:
                    continue

                row = frame_rows.iloc[0]

                for landmark in range(21):

                    feature = [
                        float(
                            row[
                                relative_x_cols[
                                    landmark
                                ]
                            ]
                        ),

                        float(
                            row[
                                relative_y_cols[
                                    landmark
                                ]
                            ]
                        ),

                        float(
                            row[
                                delta_x_cols[
                                    landmark
                                ]
                            ]
                        ),

                        float(
                            row[
                                delta_y_cols[
                                    landmark
                                ]
                            ]
                        )
                    ]

                    nodes.append(feature)

                    node_lookup[
                        (frame, landmark)
                    ] = node_index

                    node_index += 1

            if len(nodes) != WINDOW_SIZE * 21:
                continue

            edges = []

            for frame in window_frames:

                for landmark_a, landmark_b in SPATIAL_EDGES:

                    node_a = node_lookup[
                        (frame, landmark_a)
                    ]

                    node_b = node_lookup[
                        (frame, landmark_b)
                    ]

                    edges.append(
                        [node_a, node_b]
                    )

                    edges.append(
                        [node_b, node_a]
                    )

            for frame_index in range(
                WINDOW_SIZE - 1
            ):

                frame_a = window_frames[
                    frame_index
                ]

                frame_b = window_frames[
                    frame_index + 1
                ]

                for landmark in range(21):

                    node_a = node_lookup[
                        (frame_a, landmark)
                    ]

                    node_b = node_lookup[
                        (frame_b, landmark)
                    ]

                    edges.append(
                        [node_a, node_b]
                    )

                    edges.append(
                        [node_b, node_a]
                    )

            x = torch.tensor(
                nodes,
                dtype=torch.float
            )

            edge_index = torch.tensor(
                edges,
                dtype=torch.long
            ).t().contiguous()

            target = x.mean(
                dim=0,
                keepdim=True
            )

            graph = Data(
                x=x,
                edge_index=edge_index,
                y=target
            )

            graph.video = video_name
            graph.hand = str(hand)
            graph.start_frame = int(
                window_frames[0]
            )

            graphs.append(graph)

    return graphs


def build_dataset():

    videos = [
        "a",
        "abjad",
        "acak",
        "adab",
        "adaptasi"
    ]

    all_graphs = []

    for video in videos:

        print(
            f"Memproses video: {video}.mp4"
        )

        df = load_video_csv(video)

        graphs = create_temporal_graphs(
            df,
            video
        )

        print(
            f"  Jumlah graph: {len(graphs)}"
        )

        all_graphs.extend(graphs)

    return all_graphs


class GCNReconstruction(nn.Module):

    def __init__(
        self,
        input_dim=4,
        hidden_dim=64,
        embedding_dim=32
    ):

        super().__init__()

        self.conv1 = GCNConv(
            input_dim,
            hidden_dim
        )

        self.conv2 = GCNConv(
            hidden_dim,
            hidden_dim
        )

        self.embedding = nn.Linear(
            hidden_dim,
            embedding_dim
        )

        self.decoder = nn.Linear(
            embedding_dim,
            input_dim
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

        x = global_mean_pool(
            x,
            batch
        )

        embedding = self.embedding(x)

        reconstruction = self.decoder(
            embedding
        )

        return embedding, reconstruction


def train_model(
    train_graphs,
    device
):

    model = GCNReconstruction(
        input_dim=4,
        hidden_dim=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM
    ).to(device)

    loader = DataLoader(
        train_graphs,
        batch_size=32,
        shuffle=True
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.MSELoss()

    print()
    print("MEMULAI TRAINING")

    print(
        f"Jumlah graph training : "
        f"{len(train_graphs)}"
    )

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        total_loss = 0.0

        for batch in loader:

            batch = batch.to(device)

            optimizer.zero_grad()

            embedding, reconstruction = model(
                batch.x,
                batch.edge_index,
                batch.batch
            )

            target = batch.y

            loss = criterion(
                reconstruction,
                target
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                *
                batch.num_graphs
            )

        average_loss = (
            total_loss
            /
            len(train_graphs)
        )

        if (
            epoch <= 5
            or epoch % 10 == 0
            or epoch == EPOCHS
        ):

            print(
                f"Epoch {epoch:03d}/{EPOCHS} "
                f"| Loss: {average_loss:.8f}"
            )

    return model


def evaluate_test(
    model,
    test_graphs,
    device
):

    model.eval()

    embeddings = []
    reconstruction_losses = []

    criterion = nn.MSELoss()

    with torch.no_grad():

        for graph in test_graphs:

            graph_device = graph.to(device)

            batch = torch.zeros(
                graph_device.x.size(0),
                dtype=torch.long,
                device=device
            )

            embedding, reconstruction = model(
                graph_device.x,
                graph_device.edge_index,
                batch
            )

            loss = criterion(
                reconstruction,
                graph_device.y
            )

            embeddings.append(
                embedding.cpu().squeeze(0)
            )

            reconstruction_losses.append(
                loss.item()
            )

    embeddings = torch.stack(
        embeddings
    )

    mean_loss = np.mean(
        reconstruction_losses
    )

    return embeddings, mean_loss


def evaluate_motion_relationship(
    test_graphs,
    embeddings
):

    samples = []

    for index, graph in enumerate(
        test_graphs
    ):

        samples.append({
            "video": graph.video,
            "hand": graph.hand,
            "start_frame": graph.start_frame,
            "embedding_index": index
        })

    samples = pd.DataFrame(
        samples
    )

    motion_changes = []
    embedding_changes = []

    for i in range(
        len(samples)
    ):

        current = samples.iloc[i]

        next_candidates = samples[
            (samples["video"] == current["video"])
            &
            (samples["hand"] == current["hand"])
            &
            (
                samples["start_frame"]
                ==
                current["start_frame"] + 1
            )
        ]

        if len(next_candidates) == 0:
            continue

        next_row = next_candidates.iloc[0]

        current_index = int(
            current["embedding_index"]
        )

        next_index = int(
            next_row["embedding_index"]
        )

        graph_current = test_graphs[
            current_index
        ]

        graph_next = test_graphs[
            next_index
        ]

        motion_change = torch.norm(
            graph_next.x
            -
            graph_current.x,
            dim=1
        ).mean().item()

        embedding_change = torch.norm(
            embeddings[next_index]
            -
            embeddings[current_index]
        ).item()

        motion_changes.append(
            motion_change
        )

        embedding_changes.append(
            embedding_change
        )

    print()
    print("HASIL EVALUASI MOTION vs EMBEDDING")

    print(
        f"Jumlah pasangan window : "
        f"{len(motion_changes)}"
    )

    if len(motion_changes) < 2:

        print(
            "Pasangan data tidak cukup "
            "untuk menghitung korelasi."
        )

        return

    motion_changes = np.array(
        motion_changes
    )

    embedding_changes = np.array(
        embedding_changes
    )

    pearson_corr, pearson_p = pearsonr(
        motion_changes,
        embedding_changes
    )

    spearman_corr, spearman_p = spearmanr(
        motion_changes,
        embedding_changes
    )

    print()
    print(
        f"Mean motion change     : "
        f"{motion_changes.mean():.6f}"
    )

    print(
        f"Mean embedding change  : "
        f"{embedding_changes.mean():.6f}"
    )

    print()
    print(
        f"Pearson correlation    : "
        f"{pearson_corr:.6f}"
    )

    print(
        f"Pearson p-value        : "
        f"{pearson_p:.6f}"
    )

    print()
    print(
        f"Spearman correlation   : "
        f"{spearman_corr:.6f}"
    )

    print(
        f"Spearman p-value       : "
        f"{spearman_p:.6f}"
    )

    return {
        "motion_changes": motion_changes,
        "embedding_changes": embedding_changes,
        "pearson": pearson_corr,
        "pearson_p": pearson_p,
        "spearman": spearman_corr,
        "spearman_p": spearman_p
    }


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("GCN HOLD-OUT VIDEO EVALUATION")

    print(
        f"Device : {device}"
    )

    print()
    print(
        "Membuat dataset temporal graph..."
    )

    all_graphs = build_dataset()

    train_graphs = [
        graph
        for graph in all_graphs
        if graph.video != TEST_VIDEO
    ]

    test_graphs = [
        graph
        for graph in all_graphs
        if graph.video == TEST_VIDEO
    ]

    print()
    print("PEMBAGIAN DATASET")

    print(
        "Video training : "
        "a, abjad, acak, adab"
    )

    print(
        f"Video testing  : "
        f"{TEST_VIDEO}"
    )

    print(
        f"Jumlah graph training : "
        f"{len(train_graphs)}"
    )

    print(
        f"Jumlah graph testing  : "
        f"{len(test_graphs)}"
    )

    model = train_model(
        train_graphs,
        device
    )

    os.makedirs(
        os.path.dirname(MODEL_OUTPUT),
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        MODEL_OUTPUT
    )

    print()
    print(
        f"Model disimpan : "
        f"{MODEL_OUTPUT}"
    )

    print()
    print(
        "MENGUJI MODEL PADA VIDEO "
        "YANG TIDAK DILATIH"
    )

    test_embeddings, test_loss = evaluate_test(
        model,
        test_graphs,
        device
    )

    print()
    print("HASIL TESTING")

    print(
        f"Video testing        : "
        f"{TEST_VIDEO}.mp4"
    )

    print(
        f"Jumlah graph testing : "
        f"{len(test_graphs)}"
    )

    print(
        f"Embedding shape      : "
        f"{test_embeddings.shape}"
    )

    print(
        f"Reconstruction loss  : "
        f"{test_loss:.8f}"
    )

    torch.save(
        test_embeddings,
        EMBEDDING_OUTPUT
    )

    print(
        f"Embedding disimpan   : "
        f"{EMBEDDING_OUTPUT}"
    )

    evaluate_motion_relationship(
        test_graphs,
        test_embeddings
    )

    print()
    print(
        "EVALUASI HOLD-OUT SELESAI"
    )


if __name__ == "__main__":
    main()