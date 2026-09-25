import pandas as pd
import numpy as np
import os
import glob

INPUT_DIR = "research/batch_test"
OUTPUT_DIR = "research/spatiotemporal"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# KONFIGURASI LANDMARK UNTUK SUDUT JARI

ANGLE_TRIPLETS = {
    "thumb": (1, 2, 4),
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20)
}

# FUNGSI MENGHITUNG SUDUT

def calculate_angle(p1, p2, p3):

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

    cosine_angle = np.dot(v1, v2) / (
        norm_v1 * norm_v2
    )

    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0
    )

    angle = np.degrees(
        np.arccos(cosine_angle)
    )

    return angle

# MENCARI FILE CSV

csv_files = sorted(
    glob.glob(
        os.path.join(
            INPUT_DIR,
            "*.csv"
        )
    )
)


if not csv_files:

    print("Tidak ditemukan file CSV.")

    print(
        f"Folder yang diperiksa: {INPUT_DIR}"
    )

    exit()

# HEADER

print("\nBATCH SPATIO-TEMPORAL REPRESENTATION")

print(
    f"Input directory  : {INPUT_DIR}"
)

print(
    f"Jumlah CSV       : {len(csv_files)}"
)

print()


summary = []

# PROSES SEMUA VIDEO

for csv_path in csv_files:

    filename = os.path.basename(
        csv_path
    )

    video_name = os.path.splitext(
        filename
    )[0]


    print("-" * 70)
    print(
        f"Memproses : {filename}"
    )

    # LOAD DATA

    df = pd.read_csv(
        csv_path
    )


    raw_rows = len(df)


    print(
        f"Raw rows : {raw_rows}"
    )

    # URUTKAN DATA

    df = df.sort_values(
        [
            "frame",
            "hand",
            "landmark"
        ]
    ).reset_index(
        drop=True
    )

    # SPATIAL REPRESENTATION

    # Ambil wrist / landmark 0
    wrist = (
        df[
            df["landmark"] == 0
        ]
        [
            [
                "frame",
                "hand",
                "x",
                "y"
            ]
        ]
        .drop_duplicates(
            subset=[
                "frame",
                "hand"
            ]
        )
        .rename(
            columns={
                "x": "wrist_x",
                "y": "wrist_y"
            }
        )
    )


    # Gabungkan wrist ke semua landmark
    spatial = df.merge(
        wrist,
        on=[
            "frame",
            "hand"
        ],
        how="left"
    )


    # POSISI RELATIF
    spatial["relative_x"] = (
        spatial["x"]
        -
        spatial["wrist_x"]
    )


    spatial["relative_y"] = (
        spatial["y"]
        -
        spatial["wrist_y"]
    )


    # JARAK DARI WRIST

    spatial[
        "distance_from_wrist"
    ] = np.sqrt(
        spatial["relative_x"] ** 2
        +
        spatial["relative_y"] ** 2
    )

    # JOINT ANGLE

    angle_records = []


    grouped = spatial.groupby(
        [
            "frame",
            "timestamp_ms",
            "hand"
        ]
    )


    for (
        frame,
        timestamp_ms,
        hand
    ), group in grouped:

        landmarks = {}


        for _, row in group.iterrows():

            landmarks[
                int(row["landmark"])
            ] = (
                row["x"],
                row["y"]
            )


        record = {
            "frame": frame,
            "timestamp_ms": timestamp_ms,
            "hand": hand
        }


        for finger, (
            a,
            b,
            c
        ) in ANGLE_TRIPLETS.items():

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


            record[
                f"{finger}_angle"
            ] = angle


        angle_records.append(
            record
        )


    angles = pd.DataFrame(
        angle_records
    )

    # GABUNGKAN ANGLE

    spatial = spatial.merge(
        angles,
        on=[
            "frame",
            "timestamp_ms",
            "hand"
        ],
        how="left"
    )


    # KOLOM SPATIAL

    spatial_columns = [
        "frame",
        "timestamp_ms",
        "hand",
        "landmark",
        "relative_x",
        "relative_y",
        "distance_from_wrist",
        "thumb_angle",
        "index_angle",
        "middle_angle",
        "ring_angle",
        "pinky_angle"
    ]


    spatial_output = spatial[
        spatial_columns
    ].copy()

    # TEMPORAL REPRESENTATION

    temporal = spatial.sort_values(
        [
            "hand",
            "landmark",
            "frame"
        ]
    ).copy()


    # Perbedaan frame
    temporal["frame_diff"] = (
        temporal
        .groupby(
            [
                "hand",
                "landmark"
            ]
        )["frame"]
        .diff()
    )


    # Perubahan posisi X
    temporal["delta_x"] = (
        temporal
        .groupby(
            [
                "hand",
                "landmark"
            ]
        )["relative_x"]
        .diff()
    )


    # Perubahan posisi Y
    temporal["delta_y"] = (
        temporal
        .groupby(
            [
                "hand",
                "landmark"
            ]
        )["relative_y"]
        .diff()
    )

    # HANYA FRAME BERURUTAN

    temporal = temporal[
        temporal["frame_diff"] == 1
    ].copy()

    # DISPLACEMENT

    temporal[
        "displacement"
    ] = np.sqrt(
        temporal["delta_x"] ** 2
        +
        temporal["delta_y"] ** 2
    )

    # ARAH GERAKAN

    temporal[
        "direction"
    ] = np.degrees(
        np.arctan2(
            temporal["delta_y"],
            temporal["delta_x"]
        )
    )

    # KOLOM TEMPORAL

    temporal_columns = [
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


    temporal_output = temporal[
        temporal_columns
    ].copy()

    # VALIDASI JUMLAH BARIS

    if len(spatial_output) != raw_rows:

        print(
            "PERINGATAN:"
        )

        print(
            f"Raw rows    : {raw_rows}"
        )

        print(
            f"Spatial rows: {len(spatial_output)}"
        )

        print(
            "Jumlah baris spatial tidak sama!"
        )

    else:

        print(
            f"Spatial  : {len(spatial_output)} rows"
        )


    print(
        f"Temporal : {len(temporal_output)} rows"
    )

    spatial_path = os.path.join(
        OUTPUT_DIR,
        f"{video_name}_spatial.csv"
    )


    spatial_output.to_csv(
        spatial_path,
        index=False
    )


    temporal_path = os.path.join(
        OUTPUT_DIR,
        f"{video_name}_temporal.csv"
    )


    temporal_output.to_csv(
        temporal_path,
        index=False
    )


    summary.append({

        "video": video_name,

        "raw_rows": raw_rows,

        "spatial_rows":
            len(spatial_output),

        "temporal_rows":
            len(temporal_output),

        "frames":
            df["frame"].nunique(),

        "hands":
            df["hand"].nunique(),

        "landmarks":
            df["landmark"].nunique(),

        "mean_displacement":
            temporal_output[
                "displacement"
            ].mean(),

        "median_displacement":
            temporal_output[
                "displacement"
            ].median(),

        "std_displacement":
            temporal_output[
                "displacement"
            ].std(),

        "max_displacement":
            temporal_output[
                "displacement"
            ].max(),

        "mean_direction":
            temporal_output[
                "direction"
            ].mean()
    })



summary_df = pd.DataFrame(
    summary
)


summary_path = os.path.join(
    OUTPUT_DIR,
    "summary_spatiotemporal.csv"
)


summary_df.to_csv(
    summary_path,
    index=False
)


print("\nREKAP SPATIO-TEMPORAL")
print(
    summary_df.to_string(
        index=False
    )
)

print("\nSELESAI")

print(
    f"Output folder : {OUTPUT_DIR}"
)

print(
    f"Summary       : {summary_path}"
)