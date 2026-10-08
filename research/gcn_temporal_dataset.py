import os
import pandas as pd
import torch
from torch_geometric.data import Data


NODE_FILE = "research/spatiotemporal/graphs/all_graph_nodes.csv"
OUTPUT_DIR = "research/gcn"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "temporal_graph_dataset.pt"
)

WINDOW_SIZE = 5

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


print("MEMBUAT TEMPORAL GRAPH DATASET")

nodes = pd.read_csv(NODE_FILE)

print(f"Total nodes : {len(nodes)}")

graphs = []

grouped = nodes.groupby(
    ["video", "hand"],
    sort=False
)

for (video, hand), group in grouped:

    group = group.sort_values(
        ["frame", "node"]
    )

    frames = sorted(
        group["frame"].unique()
    )

    for start in range(
        len(frames) - WINDOW_SIZE + 1
    ):

        window_frames = frames[
            start:start + WINDOW_SIZE
        ]

        expected_frames = list(
            range(
                window_frames[0],
                window_frames[-1] + 1
            )
        )

        if window_frames != expected_frames:
            continue

        window = group[
            group["frame"].isin(window_frames)
        ]

        if len(window) != WINDOW_SIZE * 21:
            continue

        window = window.sort_values(
            ["frame", "node"]
        )

        features = window[
            [
                "x",
                "y",
                "dx",
                "dy"
            ]
        ].values

        x = torch.tensor(
            features,
            dtype=torch.float
        )

        edge_pairs = []

        frame_to_offset = {}

        for frame_index, frame in enumerate(
            window_frames
        ):
            frame_to_offset[frame] = (
                frame_index * 21
            )

        for frame in window_frames:

            offset = frame_to_offset[frame]

            for source, target in SPATIAL_EDGES:

                edge_pairs.append([
                    offset + source,
                    offset + target
                ])

                edge_pairs.append([
                    offset + target,
                    offset + source
                ])

        for i in range(
            WINDOW_SIZE - 1
        ):

            current_offset = i * 21
            next_offset = (i + 1) * 21

            for landmark in range(21):

                edge_pairs.append([
                    current_offset + landmark,
                    next_offset + landmark
                ])

                edge_pairs.append([
                    next_offset + landmark,
                    current_offset + landmark
                ])

        edge_index = torch.tensor(
            edge_pairs,
            dtype=torch.long
        ).t().contiguous()

        data = Data(
            x=x,
            edge_index=edge_index
        )

        data.video = video
        data.hand = hand
        data.start_frame = int(
            window_frames[0]
        )
        data.end_frame = int(
            window_frames[-1]
        )

        graphs.append(data)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

torch.save(
    graphs,
    OUTPUT_FILE
)

print()
print("TEMPORAL GRAPH DATASET SELESAI")
print(f"Jumlah graph : {len(graphs)}")
print(f"Node per graph : {WINDOW_SIZE * 21}")
print(f"Output : {OUTPUT_FILE}")