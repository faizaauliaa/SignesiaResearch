import pandas as pd
import numpy as np
import os

INPUT_DIR = "research/frame_hand"
OUTPUT_DIR = "research/optimized"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nOPTIMASI REPRESENTASI SPATIAL-TEMPORAL")
print(f"Input directory  : {INPUT_DIR}")
print(f"Output directory : {OUTPUT_DIR}")


FILES = [
    "a_frame_hand.csv",
    "abjad_frame_hand.csv",
    "acak_frame_hand.csv",
    "adab_frame_hand.csv",
    "adaptasi_frame_hand.csv"
]


def optimize_features(df):

    # METADATA

    selected_columns = [
        "frame",
        "timestamp_ms",
        "hand"
    ]

    # ========================================================
    # SPATIAL REPRESENTATION
    #
    # Dipertahankan:
    # - relative_x
    # - relative_y
    #
    # Dihilangkan:
    # - distance_from_wrist
    #
    # Karena:
    # distance = sqrt(relative_x^2 + relative_y^2)
    # ========================================================

    for landmark in range(21):

        selected_columns.append(
            f"relative_x_{landmark}"
        )

        selected_columns.append(
            f"relative_y_{landmark}"
        )

    # ========================================================
    # FINGER CONFIGURATION
    #
    # Semua sudut jari tetap dipertahankan karena masing-masing
    # merepresentasikan konfigurasi jari yang berbeda.
    # ========================================================

    angle_features = [
        "thumb_angle",
        "index_angle",
        "middle_angle",
        "ring_angle",
        "pinky_angle"
    ]

    selected_columns.extend(angle_features)

    # ========================================================
    # TEMPORAL REPRESENTATION
    #
    # Dipertahankan:
    # - delta_x
    # - delta_y
    #
    # Dihilangkan:
    # - displacement
    # - direction
    #
    # Karena keduanya dapat dihitung kembali dari delta_x
    # dan delta_y.
    # ========================================================

    for landmark in range(21):

        selected_columns.append(
            f"delta_x_{landmark}"
        )

        selected_columns.append(
            f"delta_y_{landmark}"
        )

    # HANYA AMBIL KOLOM YANG TERSEDIA

    selected_columns = [
        column
        for column in selected_columns
        if column in df.columns
    ]

    optimized = df[selected_columns].copy()

    return optimized


all_data = []

# PROSES SEMUA VIDEO

for filename in FILES:

    path = os.path.join(INPUT_DIR, filename)

    if not os.path.exists(path):
        print(f"\nFile tidak ditemukan : {filename}")
        continue

    print(f"\nMemproses : {filename}")

    df = pd.read_csv(path)

    print(f"Jumlah baris awal : {len(df)}")
    print(f"Jumlah kolom awal : {len(df.columns)}")

    optimized = optimize_features(df)

    output_name = filename.replace(
        "_frame_hand.csv",
        "_optimized.csv"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    optimized.to_csv(
        output_path,
        index=False
    )

    all_data.append(optimized)

    print(f"Jumlah kolom hasil : {len(optimized.columns)}")
    print(f"Kolom dikurangi    : {len(df.columns) - len(optimized.columns)}")
    print(f"Output             : {output_path}")


# GABUNGKAN SEMUA DATA

if len(all_data) > 0:

    combined = pd.concat(
        all_data,
        ignore_index=True
    )

    combined_path = os.path.join(
        OUTPUT_DIR,
        "all_videos_optimized.csv"
    )

    combined.to_csv(
        combined_path,
        index=False
    )

    print("\nREKAP OPTIMASI")
    print(f"Jumlah data gabungan : {len(combined)}")
    print(f"Jumlah fitur awal    : 155")
    print(f"Jumlah fitur hasil   : {len(combined.columns)}")
    print(f"Fitur dikurangi      : {155 - len(combined.columns)}")
    print(f"Output gabungan      : {combined_path}")


print("\nOPTIMASI REPRESENTASI SELESAI")