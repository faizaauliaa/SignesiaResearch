import pandas as pd
import numpy as np

INPUT_PATH = "research/batch_test/a.csv"

print("\nANALISIS SPATIAL LANDMARK")


df = pd.read_csv(INPUT_PATH)

print(f"Jumlah baris : {len(df)}")
print(f"Jumlah frame : {df['frame'].nunique()}")
print(f"Tangan      : {df['hand'].unique().tolist()}")


def distance(point_a, point_b):
    dx = point_a["x"] - point_b["x"]
    dy = point_a["y"] - point_b["y"]

    return np.sqrt(dx**2 + dy**2)


# PASANGAN LANDMARK
#
# 0  = wrist
# 4  = thumb tip
# 8  = index finger tip
# 12 = middle finger tip
# 16 = ring finger tip
# 20 = pinky tip
#
# Kita gunakan beberapa hubungan penting
# antara wrist dan ujung jari.
#

pairs = [
    (0, 4),
    (0, 8),
    (0, 12),
    (0, 16),
    (0, 20),
    (4, 8),
    (8, 12),
    (12, 16),
    (16, 20)
]


results = []

for (frame, hand), group in df.groupby(
    ["frame", "hand"]
):

    points = {
        int(row["landmark"]): row
        for _, row in group.iterrows()
    }

    for a, b in pairs:

        if a in points and b in points:

            d = distance(
                points[a],
                points[b]
            )

            results.append({
                "frame": frame,
                "hand": hand,
                "landmark_a": a,
                "landmark_b": b,
                "distance_xy": d
            })


spatial = pd.DataFrame(results)



print("\nStatistik jarak antar-landmark:")

print(
    spatial["distance_xy"]
    .describe()
    .to_string()
)


pair_stats = (
    spatial
    .groupby(
        ["landmark_a", "landmark_b"]
    )["distance_xy"]
    .agg(
        ["mean", "std", "min", "max"]
    )
    .sort_values(
        "mean",
        ascending=False
    )
)

print("\nRata-rata jarak setiap pasangan:")

print(
    pair_stats.to_string(
        float_format=lambda x: f"{x:.6f}"
    )
)


print("\nPasangan dengan variasi jarak terbesar:")

variation = (
    spatial
    .groupby(
        ["landmark_a", "landmark_b"]
    )["distance_xy"]
    .std()
    .sort_values(
        ascending=False
    )
)

print(
    variation.head(10).to_string(
        float_format=lambda x: f"{x:.6f}"
    )
)


output_path = "research/spatial_a.csv"

spatial.to_csv(
    output_path,
    index=False
)

print("\nHasil disimpan:")
print(output_path)

print("\nANALISIS SPATIAL SELESAI")