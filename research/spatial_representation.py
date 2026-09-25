import pandas as pd
import numpy as np
import os


INPUT_PATH = "research/landmarks_a.csv"

OUTPUT_PATH = "research/spatial_representation_a.csv"

# CHECK FILE

if not os.path.exists(INPUT_PATH):
    print(f"File tidak ditemukan: {INPUT_PATH}")
    exit()


df = pd.read_csv(INPUT_PATH)

print("SPATIAL REPRESENTATION")
print(f"Input file : {INPUT_PATH}")
print(f"Total rows : {len(df)}")
print()


# LANDMARK REFERENCE
#
# Landmark 0 digunakan sebagai reference point
# (wrist).
#
# Setiap landmark akan direpresentasikan secara relatif
# terhadap landmark 0.
#
# relative_x = x_landmark - x_wrist
# relative_y = y_landmark - y_wrist
#
# Fokus penelitian tahap ini menggunakan X dan Y.

REFERENCE_LANDMARK = 0

# PROCESS EACH HAND AND FRAME

results = []

grouped = df.groupby(["frame", "hand"])

for (frame, hand), group in grouped:

    group = group.sort_values("landmark")

    # Pastikan landmark reference tersedia
    reference = group[group["landmark"] == REFERENCE_LANDMARK]

    if reference.empty:
        continue

    wrist_x = reference.iloc[0]["x"]
    wrist_y = reference.iloc[0]["y"]

    for _, row in group.iterrows():

        landmark_id = int(row["landmark"])

        x = row["x"]
        y = row["y"]


        # Relative position terhadap wrist

        relative_x = x - wrist_x
        relative_y = y - wrist_y

        # Euclidean distance dari wrist

        distance_from_wrist = np.sqrt(
            relative_x**2 +
            relative_y**2
        )

        results.append([
            frame,
            row["timestamp_ms"],
            hand,
            landmark_id,
            x,
            y,
            relative_x,
            relative_y,
            distance_from_wrist
        ])

# CREATE DATAFRAME

spatial_df = pd.DataFrame(
    results,
    columns=[
        "frame",
        "timestamp_ms",
        "hand",
        "landmark",
        "x",
        "y",
        "relative_x",
        "relative_y",
        "distance_from_wrist"
    ]
)

# SAVE RESULT

spatial_df.to_csv(
    OUTPUT_PATH,
    index=False
)

# SUMMARY

print("HASIL SPATIAL REPRESENTATION")

print(f"Total rows           : {len(spatial_df)}")
print(f"Unique frames        : {spatial_df['frame'].nunique()}")
print(f"Unique landmarks     : {spatial_df['landmark'].nunique()}")
print(f"Hands                : {spatial_df['hand'].unique().tolist()}")

print()
print("Kolom hasil:")
print(spatial_df.columns.tolist())

print()
print("Contoh data:")
print(spatial_df.head(10))

print()
print("Statistik relative_x:")
print(spatial_df["relative_x"].describe())

print()
print("Statistik relative_y:")
print(spatial_df["relative_y"].describe())

print()
print("Statistik distance_from_wrist:")
print(spatial_df["distance_from_wrist"].describe())

print()
print("SELESAI")
print(f"Output : {OUTPUT_PATH}")