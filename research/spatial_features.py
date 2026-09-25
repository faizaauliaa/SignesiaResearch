import pandas as pd
import numpy as np
import os

# KONFIGURASI

INPUT_PATH = "research/landmarks_a.csv"
OUTPUT_PATH = "research/spatial_features_a.csv"

# Triplet landmark untuk menghitung sudut tiap jari
ANGLE_TRIPLETS = {
    "thumb":  (1, 2, 4),
    "index":  (5, 6, 8),
    "middle": (9, 10, 12),
    "ring":   (13, 14, 16),
    "pinky":  (17, 18, 20)
}

# FUNGSI HITUNG SUDUT

def calculate_angle(p1, p2, p3):
    """
    Menghitung sudut pada titik p2
    menggunakan koordinat X-Y.
    """

    v1 = np.array([
        p1[0] - p2[0],
        p1[1] - p2[1]
    ])

    v2 = np.array([
        p3[0] - p2[0],
        p3[1] - p2[1]
    ])

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return np.nan

    cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)

    # Menghindari error floating point
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(
        np.arccos(cosine_angle)
    )

    return angle

# LOAD DATA

if not os.path.exists(INPUT_PATH):
    print(f"File tidak ditemukan: {INPUT_PATH}")
    exit()

df = pd.read_csv(INPUT_PATH)

print("\nSPATIAL FEATURE EXTRACTION")
print(f"Input file : {INPUT_PATH}")
print(f"Total rows : {len(df)}")
print()

# HITUNG SUDUT SETIAP FRAME DAN TANGAN

angle_records = []

grouped = df.groupby(["frame", "timestamp_ms", "hand"])

for (frame, timestamp_ms, hand), group in grouped:

    landmarks = {}

    for _, row in group.iterrows():
        landmarks[int(row["landmark"])] = (
            row["x"],
            row["y"]
        )

    record = {
        "frame": frame,
        "timestamp_ms": timestamp_ms,
        "hand": hand
    }

    for finger, (a, b, c) in ANGLE_TRIPLETS.items():

        if (
            a in landmarks
            and b in landmarks
            and c in landmarks
        ):
            angle = calculate_angle(
                landmarks[a],
                landmarks[b],
                landmarks[c]
            )
        else:
            angle = np.nan

        record[f"{finger}_angle"] = angle

    angle_records.append(record)


angles_df = pd.DataFrame(angle_records)

# GABUNGKAN DENGAN SPATIAL REPRESENTATION

spatial_df = pd.read_csv(
    "research/spatial_representation_a.csv"
)

# Ambil satu baris per frame + hand
base_df = spatial_df[
    [
        "frame",
        "timestamp_ms",
        "hand",
        "landmark",
        "relative_x",
        "relative_y",
        "distance_from_wrist"
    ]
].copy()

# Gabungkan sudut
result = base_df.merge(
    angles_df,
    on=["frame", "timestamp_ms", "hand"],
    how="left"
)


result.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\nHASIL SPATIAL FEATURES")
print(f"Total rows       : {len(result)}")
print(f"Unique frames    : {result['frame'].nunique()}")
print(f"Unique landmarks : {result['landmark'].nunique()}")
print(f"Hands            : {result['hand'].unique().tolist()}")

print()
print("Kolom hasil:")
print(result.columns.tolist())

print()
print("Statistik sudut:")

angle_columns = [
    "thumb_angle",
    "index_angle",
    "middle_angle",
    "ring_angle",
    "pinky_angle"
]

print(
    result[angle_columns].describe()
)

print()
print("\nSELESAI")
print(f"Output : {OUTPUT_PATH}")