import pandas as pd
import os

INPUT_DIR = "research/spatiotemporal"
OUTPUT_DIR = "research/frame_hand"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nMEMBENTUK REPRESENTASI FRAME-HAND")
print(f"Input directory  : {INPUT_DIR}")
print(f"Output directory : {OUTPUT_DIR}")

spatial_files = [
    "a_spatial.csv",
    "abjad_spatial.csv",
    "acak_spatial.csv",
    "adab_spatial.csv",
    "adaptasi_spatial.csv"
]

temporal_files = [
    "a_temporal.csv",
    "abjad_temporal.csv",
    "acak_temporal.csv",
    "adab_temporal.csv",
    "adaptasi_temporal.csv"
]

for spatial_file, temporal_file in zip(spatial_files, temporal_files):

    video_name = spatial_file.replace("_spatial.csv", "")

    spatial_path = os.path.join(INPUT_DIR, spatial_file)
    temporal_path = os.path.join(INPUT_DIR, temporal_file)

    print(f"\nMemproses : {video_name}")

    if not os.path.exists(spatial_path):
        print(f"File spatial tidak ditemukan : {spatial_file}")
        continue

    if not os.path.exists(temporal_path):
        print(f"File temporal tidak ditemukan : {temporal_file}")
        continue

    spatial = pd.read_csv(spatial_path)
    temporal = pd.read_csv(temporal_path)

    # 1. Ambil satu nilai fitur spatial untuk setiap frame + hand

    spatial_features = [
        "frame",
        "timestamp_ms",
        "hand",
        "thumb_angle",
        "index_angle",
        "middle_angle",
        "ring_angle",
        "pinky_angle"
    ]

    spatial_frame_hand = (
        spatial[spatial_features]
        .drop_duplicates(subset=["frame", "hand"])
        .copy()
    )

    # 2. Ambil fitur posisi relatif dari landmark
    #    Setiap landmark menjadi kolom tersendiri

    position = spatial[
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

    position["landmark"] = position["landmark"].astype(int)

    position_wide = position.pivot_table(
        index=["frame", "timestamp_ms", "hand"],
        columns="landmark",
        values=[
            "relative_x",
            "relative_y",
            "distance_from_wrist"
        ],
        aggfunc="first"
    )

    position_wide.columns = [
        f"{feature}_{landmark}"
        for feature, landmark in position_wide.columns
    ]

    position_wide = position_wide.reset_index()

    # 3. Gabungkan fitur spatial
    spatial_frame_hand = spatial_frame_hand.merge(
        position_wide,
        on=["frame", "timestamp_ms", "hand"],
        how="left"
    )

    # 4. Gabungkan fitur temporal

    temporal_features = [
        "frame",
        "timestamp_ms",
        "hand",
        "delta_x",
        "delta_y",
        "displacement",
        "direction"
    ]

    temporal_selected = temporal[temporal_features].copy()

    # Temporal juga masih per-landmark.
    # Untuk tahap awal ini kita simpan displacement dan direction
    # pada setiap landmark sebagai fitur tersendiri.

    temporal_wide = temporal_selected.pivot_table(
        index=["frame", "timestamp_ms", "hand"],
        columns=None
    )

    # 5. Buat fitur temporal per landmark

    temporal_detail = temporal[
        [
            "frame",
            "timestamp_ms",
            "hand",
            "landmark",
            "delta_x",
            "delta_y",
            "displacement",
            "direction"
        ]
    ].copy()

    temporal_detail["landmark"] = temporal_detail["landmark"].astype(int)

    temporal_wide = temporal_detail.pivot_table(
        index=["frame", "timestamp_ms", "hand"],
        columns="landmark",
        values=[
            "delta_x",
            "delta_y",
            "displacement",
            "direction"
        ],
        aggfunc="first"
    )

    temporal_wide.columns = [
        f"{feature}_{landmark}"
        for feature, landmark in temporal_wide.columns
    ]

    temporal_wide = temporal_wide.reset_index()

    # 6. Gabungkan spatial + temporal

    frame_hand = spatial_frame_hand.merge(
        temporal_wide,
        on=["frame", "timestamp_ms", "hand"],
        how="left"
    )

    # 7. Urutkan data

    frame_hand = frame_hand.sort_values(
        by=["frame", "hand"]
    ).reset_index(drop=True)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{video_name}_frame_hand.csv"
    )

    frame_hand.to_csv(output_path, index=False)

    print(f"Spatial rows awal : {len(spatial)}")
    print(f"Temporal rows awal: {len(temporal)}")
    print(f"Frame-hand rows   : {len(frame_hand)}")
    print(f"Jumlah kolom      : {len(frame_hand.columns)}")
    print(f"Output            : {output_path}")

print("\nPROSES SELESAI")
print(f"Semua output tersimpan di : {OUTPUT_DIR}")