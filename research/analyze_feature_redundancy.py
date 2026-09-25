import pandas as pd
import numpy as np
import os

INPUT_DIR = "research/spatiotemporal"

FEATURES = [
    "relative_x",
    "relative_y",
    "distance_from_wrist",
    "thumb_angle",
    "index_angle",
    "middle_angle",
    "ring_angle",
    "pinky_angle"
]

TEMPORAL_FEATURES = [
    "delta_x",
    "delta_y",
    "displacement",
    "direction"
]

print("\nANALISIS REDUNDANSI FITUR")

print(f"Input directory : {INPUT_DIR}")

files = [
    "a_spatial.csv",
    "abjad_spatial.csv",
    "acak_spatial.csv",
    "adab_spatial.csv",
    "adaptasi_spatial.csv"
]

all_spatial = []

for filename in files:

    path = os.path.join(
        INPUT_DIR,
        filename
    )

    if not os.path.exists(path):
        print(f"\nFile tidak ditemukan : {filename}")
        continue

    df = pd.read_csv(path)

    all_spatial.append(
        df[FEATURES]
    )

    print(
        f"\n{filename} : {len(df)} rows"
    )

spatial_df = pd.concat(
    all_spatial,
    ignore_index=True
)

print("\nKORELASI FITUR SPATIAL")

spatial_corr = spatial_df.corr()

print(
    spatial_corr.round(3)
)

print("\nKORELASI FITUR TEMPORAL")

all_temporal = []

temporal_files = [
    "a_temporal.csv",
    "abjad_temporal.csv",
    "acak_temporal.csv",
    "adab_temporal.csv",
    "adaptasi_temporal.csv"
]

for filename in temporal_files:

    path = os.path.join(
        INPUT_DIR,
        filename
    )

    if not os.path.exists(path):
        print(f"\nFile tidak ditemukan : {filename}")
        continue

    df = pd.read_csv(path)

    all_temporal.append(
        df[TEMPORAL_FEATURES]
    )

    print(
        f"\n{filename} : {len(df)} rows"
    )

temporal_df = pd.concat(
    all_temporal,
    ignore_index=True
)


temporal_corr = temporal_df.corr()

print(
    temporal_corr.round(3)
)

print("\nSTATISTIK FITUR SPATIAL")

print(
    spatial_df.describe().round(4)
)

print("\nSTATISTIK FITUR TEMPORAL")

print(
    temporal_df.describe().round(4)
)

print("\nANALISIS SELESAI")