import os
import pandas as pd
import torch
from torch_geometric.data import Data


NODE_FILE = "research/spatiotemporal/graphs/all_graph_nodes.csv"
EDGE_FILE = "research/spatiotemporal/graphs/all_graph_edges.csv"

OUTPUT_DIR = "research/gcn"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "graph_dataset.pt"
)


print("MEMBUAT DATASET GCN")

nodes = pd.read_csv(NODE_FILE)
edges = pd.read_csv(EDGE_FILE)

print(f"Total nodes : {len(nodes)}")
print(f"Total edges : {len(edges)}")

graphs = []

group_columns = [
    "video",
    "frame",
    "hand"
]

grouped = nodes.groupby(
    group_columns,
    sort=False
)

for (video, frame, hand), group in grouped:

    group = group.sort_values("node")

    if len(group) != 21:
        continue

    x = torch.tensor(
        group[
            [
                "x",
                "y",
                "dx",
                "dy"
            ]
        ].values,
        dtype=torch.float
    )

    edge_group = edges[
        (edges["video"] == video) &
        (edges["frame"] == frame) &
        (edges["hand"] == hand)
    ]

    edge_pairs = []

    for _, edge in edge_group.iterrows():

        source = int(edge["source"])
        target = int(edge["target"])

        edge_pairs.append(
            [source, target]
        )

    if not edge_pairs:
        continue

    edge_index = torch.tensor(
        edge_pairs,
        dtype=torch.long
    ).t().contiguous()

    data = Data(
        x=x,
        edge_index=edge_index
    )

    data.video = video
    data.frame = int(frame)
    data.hand = hand

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
print("DATASET GCN SELESAI")
print(f"Jumlah graph : {len(graphs)}")
print(f"Output       : {OUTPUT_FILE}")