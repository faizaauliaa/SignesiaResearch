import pandas as pd
import numpy as np

INPUT_PATH = "research/batch_test/a.csv"

print("\nANALISIS ANGLE LANDMARK")

df = pd.read_csv(INPUT_PATH)

print(f"Jumlah baris : {len(df)}")
print(f"Jumlah frame : {df['frame'].nunique()}")
print(f"Tangan      : {df['hand'].unique().tolist()}")

def calculate_angle(a, b, c):
    """
    Menghitung sudut ABC berdasarkan koordinat X-Y.
    B = titik pusat sudut.
    """

    vector_ba = np.array([
        a["x"] - b["x"],
        a["y"] - b["y"]
    ])

    vector_bc = np.array([
        c["x"] - b["x"],
        c["y"] - b["y"]
    ])

    norm_ba = np.linalg.norm(vector_ba)
    norm_bc = np.linalg.norm(vector_bc)

    if norm_ba == 0 or norm_bc == 0:
        return np.nan

    cosine = np.dot(vector_ba, vector_bc) / (
        norm_ba * norm_bc
    )

    # Mencegah error floating point
    cosine = np.clip(cosine, -1.0, 1.0)

    angle = np.degrees(
        np.arccos(cosine)
    )

    return angle


# DEFINISI SUDUT
#
# Setiap jari memiliki tiga titik:
#
# Thumb  : 1 - 2 - 4
# Index  : 5 - 6 - 8
# Middle : 9 - 10 - 12
# Ring   : 13 - 14 - 16
# Pinky  : 17 - 18 - 20
#
# Titik tengah menjadi vertex.
#

angle_triplets = {
    "thumb": (1, 2, 4),
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20)
}

results = []

for (frame, hand), group in df.groupby(
    ["frame", "hand"]
):

    points = {
        int(row["landmark"]): row
        for _, row in group.iterrows()
    }

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

            results.append({
                "frame": frame,
                "hand": hand,
                "joint": name,
                "angle": angle
            })


angles = pd.DataFrame(results)


print("\nStatistik sudut:")

print(
    angles["angle"]
    .describe()
    .to_string()
)


joint_stats = (
    angles
    .groupby("joint")["angle"]
    .agg(
        ["mean", "std", "min", "max"]
    )
    .sort_values(
        "std",
        ascending=False
    )
)

print("\nStatistik sudut setiap jari:")

print(
    joint_stats.to_string(
        float_format=lambda x: f"{x:.6f}"
    )
)


print("\nJari dengan variasi sudut terbesar:")

variation = (
    angles
    .groupby("joint")["angle"]
    .std()
    .sort_values(
        ascending=False
    )
)

print(
    variation.to_string(
        float_format=lambda x: f"{x:.6f}"
    )
)


output_path = "research/angles_a.csv"

angles.to_csv(
    output_path,
    index=False
)

print("\nHasil disimpan:")
print(output_path)

print("\nANALISIS ANGLE SELESAI")