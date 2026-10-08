import pandas as pd
import os
import glob

# Configuration
INPUT_DIR = "research/optimized"
OUTPUT_DIR = "research/spatiotemporal/graphs"

# 21 landmark tangan
LANDMARKS = list(range(21))

# Hubungan antar-landmark dalam satu tangan
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
    (19, 20),
]

# Cari semua file hasil optimasi
input_files = sorted(
    [
        file
        for file in glob.glob(
            os.path.join(
                INPUT_DIR,
                "*_optimized.csv"
            )
        )
        if not os.path.basename(file).startswith("all_videos_")
    ]
)

print("BATCH SPATIO-TEMPORAL GRAPH REPRESENTATION")
print("=" * 70)
print(f"Input directory : {INPUT_DIR}")
print(f"Jumlah file     : {len(input_files)}")

all_nodes = []
all_edges = []

# Proses setiap video
for input_file in input_files:

    filename = os.path.basename(input_file)

    video_name = filename.replace(
        "_optimized.csv",
        ""
    )

    print("\n" + "-" * 70)
    print(f"Memproses : {filename}")

    # Membaca data
    df = pd.read_csv(input_file)

    print(f"Raw rows : {len(df)}")

    # Membuat node representation
    graph_rows = []

    for _, row in df.iterrows():

        frame = int(row["frame"])
        timestamp = int(row["timestamp_ms"])
        hand = row["hand"]

        for landmark in LANDMARKS:

            x = row[f"relative_x_{landmark}"]
            y = row[f"relative_y_{landmark}"]

            dx = row[f"delta_x_{landmark}"]
            dy = row[f"delta_y_{landmark}"]

            # Frame pertama tidak memiliki delta
            if pd.isna(dx):
                dx = 0.0

            if pd.isna(dy):
                dy = 0.0

            graph_rows.append({
                "video": video_name,
                "frame": frame,
                "timestamp_ms": timestamp,
                "hand": hand,
                "node": landmark,
                "x": x,
                "y": y,
                "dx": dx,
                "dy": dy
            })

    nodes_df = pd.DataFrame(graph_rows)

    # Membuat spatial edges
    edge_rows = []

    for _, row in nodes_df.iterrows():

        frame = row["frame"]
        timestamp = row["timestamp_ms"]
        hand = row["hand"]
        node = int(row["node"])

        for source, target in SPATIAL_EDGES:

            if node == source:

                edge_rows.append({
                    "video": video_name,
                    "frame": frame,
                    "timestamp_ms": timestamp,
                    "hand": hand,
                    "source": source,
                    "target": target,
                    "edge_type": "spatial"
                })

    # Membuat temporal edges
    for hand in nodes_df["hand"].unique():

        hand_data = nodes_df[
            nodes_df["hand"] == hand
        ]

        frames = sorted(
            hand_data["frame"].unique()
        )

        for i in range(len(frames) - 1):

            current_frame = frames[i]
            next_frame = frames[i + 1]

            # Hanya menghubungkan frame yang berurutan
            if next_frame != current_frame + 1:
                continue

            current_data = hand_data[
                hand_data["frame"] == current_frame
            ]

            next_data = hand_data[
                hand_data["frame"] == next_frame
            ]

            for landmark in LANDMARKS:

                current_node = current_data[
                    current_data["node"] == landmark
                ]

                next_node = next_data[
                    next_data["node"] == landmark
                ]

                if current_node.empty or next_node.empty:
                    continue

                edge_rows.append({
                    "video": video_name,
                    "frame": current_frame,
                    "next_frame": next_frame,
                    "timestamp_ms": int(
                        next_node.iloc[0]["timestamp_ms"]
                    ),
                    "hand": hand,
                    "source": landmark,
                    "target": landmark,
                    "edge_type": "temporal"
                })

    edges_df = pd.DataFrame(edge_rows)

    # Ringkasan setiap video
    spatial_count = len(
        edges_df[
            edges_df["edge_type"] == "spatial"
        ]
    )

    temporal_count = len(
        edges_df[
            edges_df["edge_type"] == "temporal"
        ]
    )

    print(f"Nodes          : {len(nodes_df)}")
    print(f"Spatial edges  : {spatial_count}")
    print(f"Temporal edges : {temporal_count}")
    print(f"Total edges    : {len(edges_df)}")

    all_nodes.append(nodes_df)
    all_edges.append(edges_df)

# Menggabungkan semua video
combined_nodes = pd.concat(
    all_nodes,
    ignore_index=True
)

combined_edges = pd.concat(
    all_edges,
    ignore_index=True
)

# Membuat folder output
os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

nodes_output = os.path.join(
    OUTPUT_DIR,
    "all_graph_nodes.csv"
)

edges_output = os.path.join(
    OUTPUT_DIR,
    "all_graph_edges.csv"
)

# Menyimpan hasil
combined_nodes.to_csv(
    nodes_output,
    index=False
)

combined_edges.to_csv(
    edges_output,
    index=False
)

# Rekapitulasi
spatial_total = len(
    combined_edges[
        combined_edges["edge_type"] == "spatial"
    ]
)

temporal_total = len(
    combined_edges[
        combined_edges["edge_type"] == "temporal"
    ]
)

print("\n" + "=" * 70)
print("REKAPITULASI")
print("=" * 70)

print(f"Jumlah video  : {len(input_files)}")
print(f"Total nodes   : {len(combined_nodes)}")
print(f"Spatial edges : {spatial_total}")
print(f"Temporal edges: {temporal_total}")
print(f"Total edges   : {len(combined_edges)}")

print("\nOUTPUT")
print("-" * 70)
print(f"Nodes : {nodes_output}")
print(f"Edges : {edges_output}")

print("\nSELESAI")