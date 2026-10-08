import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gcn_autoencoder_holdout import GCNEncoderDecoder


WINDOW_SIZE = 5
NUM_LANDMARKS = 21
NUM_FEATURES = 4

CSV_PATH = "research/optimized/adaptasi_optimized.csv"
MODEL_PATH = "research/gcn/gcn_autoencoder_holdout.pt"


SPATIAL_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]


def build_edge_index():
    edges = []

    for frame in range(WINDOW_SIZE):
        offset = frame * NUM_LANDMARKS

        for a, b in SPATIAL_EDGES:
            edges.append((offset + a, offset + b))
            edges.append((offset + b, offset + a))

    for frame in range(WINDOW_SIZE - 1):
        current_offset = frame * NUM_LANDMARKS
        next_offset = (frame + 1) * NUM_LANDMARKS

        for landmark in range(NUM_LANDMARKS):
            a = current_offset + landmark
            b = next_offset + landmark

            edges.append((a, b))
            edges.append((b, a))

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def create_windows(df):
    windows = []

    df = df.sort_values(["hand", "frame"])

    for hand in df["hand"].unique():
        hand_df = df[df["hand"] == hand].copy()
        hand_df = hand_df.sort_values("frame")

        frames = hand_df["frame"].unique()

        for i in range(len(frames) - WINDOW_SIZE + 1):
            selected_frames = frames[i:i + WINDOW_SIZE]

            expected_frames = np.arange(
                selected_frames[0],
                selected_frames[0] + WINDOW_SIZE
            )

            if not np.array_equal(selected_frames, expected_frames):
                continue

            rows = hand_df[
                hand_df["frame"].isin(selected_frames)
            ].copy()

            if len(rows) != WINDOW_SIZE:
                continue

            rows = rows.sort_values("frame")

            feature_columns = []

            for prefix in ["relative_x", "relative_y", "delta_x", "delta_y"]:
                feature_columns.extend(
                    [f"{prefix}_{i}" for i in range(NUM_LANDMARKS)]
                )

            values = rows[feature_columns].fillna(0.0).values

            x = []

            for frame_values in values:
                for landmark in range(NUM_LANDMARKS):
                    x.append([
                        frame_values[landmark],
                        frame_values[NUM_LANDMARKS + landmark],
                        frame_values[2 * NUM_LANDMARKS + landmark],
                        frame_values[3 * NUM_LANDMARKS + landmark]
                    ])

            x = np.array(x, dtype=np.float32)

            windows.append({
                "x": x,
                "frames": selected_frames,
                "hand": hand
            })

    return windows


print("GCN RECONSTRUCTION VISUALIZATION")
print()

df = pd.read_csv(CSV_PATH)

windows = create_windows(df)

print("Jumlah temporal window :", len(windows))

device = torch.device("cpu")

model = GCNEncoderDecoder().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

edge_index = build_edge_index().to(device)

# Menggunakan temporal window pertama
sample = windows[0]

x = torch.tensor(
    sample["x"],
    dtype=torch.float32
).to(device)

batch = torch.zeros(
    x.shape[0],
    dtype=torch.long
).to(device)

with torch.no_grad():
    embedding, reconstruction = model(
        x,
        edge_index,
        batch
    )

reconstruction = reconstruction.cpu().numpy().reshape(
    WINDOW_SIZE,
    NUM_LANDMARKS,
    NUM_FEATURES
)

original = sample["x"].reshape(
    WINDOW_SIZE,
    NUM_LANDMARKS,
    NUM_FEATURES
)

print("Embedding shape :", embedding.shape)
print("Reconstruction shape :", reconstruction.shape)

# Membandingkan frame tengah
frame_index = 2

original_xy = original[
    frame_index, :, :2
]

reconstructed_xy = reconstruction[
    frame_index, :, :2
]

plt.figure(figsize=(8, 7))

plt.scatter(
    original_xy[:, 0],
    original_xy[:, 1],
    label="Landmark asli",
    s=45
)

plt.scatter(
    reconstructed_xy[:, 0],
    reconstructed_xy[:, 1],
    marker="x",
    label="Rekonstruksi GCN",
    s=60
)

for i in range(NUM_LANDMARKS):
    plt.plot(
        [
            original_xy[i, 0],
            reconstructed_xy[i, 0]
        ],
        [
            original_xy[i, 1],
            reconstructed_xy[i, 1]
        ],
        alpha=0.3
    )

plt.title("Landmark Asli vs Rekonstruksi GCN")
plt.xlabel("Relative X")
plt.ylabel("Relative Y")
plt.legend()
plt.axis("equal")
plt.grid(alpha=0.2)
plt.tight_layout()

output_path = "research/gcn/gcn_reconstruction_comparison.png"

plt.savefig(
    output_path,
    dpi=300
)

plt.show()

print()
print("Visualisasi selesai.")
print("Output :", output_path)