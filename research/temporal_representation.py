import pandas as pd
import numpy as np
import os

INPUT_PATH = "research/spatial_representation_a.csv"
OUTPUT_PATH = "research/temporal_representation_a.csv"


# LOAD DATA

if not os.path.exists(INPUT_PATH):
    print(f"File tidak ditemukan: {INPUT_PATH}")
    exit()

df = pd.read_csv(INPUT_PATH)

print("\nTEMPORAL REPRESENTATION")

print(f"Input file : {INPUT_PATH}")
print(f"Total rows : {len(df)}")
print()


# URUTKAN DATA

df = df.sort_values(
    ["hand", "landmark", "frame"]
).reset_index(drop=True)

# HITUNG PERUBAHAN POSISI
df["frame_diff"] = (
    df.groupby(["hand", "landmark"])["frame"]
    .diff()
)


df["delta_x"] = (
    df.groupby(["hand", "landmark"])["relative_x"]
    .diff()
)


df["delta_y"] = (
    df.groupby(["hand", "landmark"])["relative_y"]
    .diff()
)


# HANYA FRAME BERURUTAN

consecutive = df[
    df["frame_diff"] == 1
].copy()

# DISPLACEMENT

consecutive["displacement"] = np.sqrt(
    consecutive["delta_x"] ** 2
    +
    consecutive["delta_y"] ** 2
)

# ARAH GERAKAN

consecutive["direction"] = np.degrees(
    np.arctan2(
        consecutive["delta_y"],
        consecutive["delta_x"]
    )
)

# OUTPUT

columns = [
    "frame",
    "timestamp_ms",
    "hand",
    "landmark",
    "relative_x",
    "relative_y",
    "delta_x",
    "delta_y",
    "displacement",
    "direction"
]

result = consecutive[columns].copy()

result.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\nHASIL TEMPORAL REPRESENTATION")

print(f"Total data temporal : {len(result)}")
print(f"Unique frames       : {result['frame'].nunique()}")
print(f"Unique landmarks    : {result['landmark'].nunique()}")
print(f"Hands               : {result['hand'].unique().tolist()}")

print()

print("Kolom hasil:")
print(result.columns.tolist())

print()

print("Statistik delta_x:")
print(result["delta_x"].describe())

print()

print("Statistik delta_y:")
print(result["delta_y"].describe())

print()

print("Statistik displacement:")
print(result["displacement"].describe())

print()

print("Statistik direction:")
print(result["direction"].describe())

print("\nSELESAI")

print(f"Output : {OUTPUT_PATH}")