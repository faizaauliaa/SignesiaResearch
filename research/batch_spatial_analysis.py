import pandas as pd
import numpy as np
import os

INPUT_FOLDER = "research/batch_test"

FILES = [
    "a.csv",
    "abjad.csv",
    "acak.csv",
    "adab.csv",
    "adaptasi.csv"
]



def distance(a, b):
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]

    return np.sqrt(dx**2 + dy**2)


def calculate_angle(a, b, c):

    ba = np.array([
        a["x"] - b["x"],
        a["y"] - b["y"]
    ])

    bc = np.array([
        c["x"] - b["x"],
        c["y"] - b["y"]
    ])

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return np.nan

    cosine = np.dot(ba, bc) / (
        norm_ba * norm_bc
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return np.degrees(
        np.arccos(cosine)
    )


distance_pairs = [
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


angle_triplets = {
    "thumb": (1, 2, 4),
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20)
}

results = []


print("\nBATCH SPATIAL REPRESENTATION ANALYSIS")


for filename in FILES:

    path = os.path.join(
        INPUT_FOLDER,
        filename
    )

    if not os.path.exists(path):

        print(f"\nFile tidak ditemukan: {path}")
        continue

    df = pd.read_csv(path)

    distances = []
    angles = []

    for (frame, hand), group in df.groupby(
        ["frame", "hand"]
    ):

        points = {
            int(row["landmark"]): row
            for _, row in group.iterrows()
        }

        for a, b in distance_pairs:

            if a in points and b in points:

                d = distance(
                    points[a],
                    points[b]
                )

                distances.append(d)

        for name, (a, b, c) in angle_triplets.items():

            if (
                a in points and
                b in points and
                c in points
            ):

                angle = calculate_angle(
                    points[a],
                    points[b],
                    points[c]
                )

                if not np.isnan(angle):
                    angles.append(angle)


    distances = np.array(distances)
    angles = np.array(angles)

    results.append({

        "video": filename.replace(".csv", ""),

        "distance_count": len(distances),

        "distance_mean":
            np.mean(distances),

        "distance_std":
            np.std(distances),

        "distance_min":
            np.min(distances),

        "distance_max":
            np.max(distances),

        "angle_count":
            len(angles),

        "angle_mean":
            np.mean(angles),

        "angle_std":
            np.std(angles),

        "angle_min":
            np.min(angles),

        "angle_max":
            np.max(angles)
    })


results = pd.DataFrame(results)

print("\nHASIL SPATIAL PER VIDEO")

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nRATA-RATA 5 VIDEO")

numeric_columns = [
    "distance_count",
    "distance_mean",
    "distance_std",
    "distance_min",
    "distance_max",
    "angle_count",
    "angle_mean",
    "angle_std",
    "angle_min",
    "angle_max"
]

print(
    results[numeric_columns]
    .mean()
    .to_string(
        float_format=lambda x: f"{x:.6f}"
    )
)


output_path = (
    "research/batch_spatial_results.csv"
)

results.to_csv(
    output_path,
    index=False
)

print("\nHasil disimpan ke:")
print(output_path)

print("\nBATCH SPATIAL ANALYSIS SELESAI")