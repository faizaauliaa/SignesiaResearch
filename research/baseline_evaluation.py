import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

WINDOW_SIZE = 5

DATA_DIR = "research/optimized"

TEST_VIDEO = "adaptasi"

FEATURE_TYPES = [
    "relative_x",
    "relative_y",
    "delta_x",
    "delta_y"
]


def load_video_data(video_name):

    path = os.path.join(
        DATA_DIR,
        f"{video_name}_optimized.csv"
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"File tidak ditemukan: {path}"
        )

    df = pd.read_csv(path)

    temporal_columns = []

    for feature in [
        "delta_x",
        "delta_y"
    ]:

        temporal_columns.extend(
            [
                f"{feature}_{i}"
                for i in range(21)
            ]
        )

    df[temporal_columns] = (
        df[temporal_columns]
        .fillna(0.0)
    )

    numeric_values = (
        df.select_dtypes(
            include=[np.number]
        ).to_numpy()
    )

    if not np.isfinite(
        numeric_values
    ).all():

        raise ValueError(
            f"Masih terdapat NaN atau Inf "
            f"pada {video_name}"
        )

    return df


def create_feature_vector(row):

    features = []

    for feature_type in FEATURE_TYPES:

        for landmark in range(21):

            column = (
                f"{feature_type}_{landmark}"
            )

            features.append(
                row[column]
            )

    return np.array(
        features,
        dtype=np.float32
    )


def create_windows(df):

    windows = []

    grouped = df.groupby("hand")

    for hand, hand_df in grouped:

        hand_df = hand_df.sort_values(
            "frame"
        ).reset_index(drop=True)

        frames = hand_df["frame"].to_numpy()

        for start in range(
            len(hand_df) - WINDOW_SIZE + 1
        ):

            window = hand_df.iloc[
                start:start + WINDOW_SIZE
            ]

            window_frames = (
                window["frame"].to_numpy()
            )

            expected_frames = np.arange(
                window_frames[0],
                window_frames[0] + WINDOW_SIZE
            )

            if not np.array_equal(
                window_frames,
                expected_frames
            ):
                continue

            feature_matrix = np.array(
                [
                    create_feature_vector(row)
                    for _, row in window.iterrows()
                ]
            )

            windows.append(
                {
                    "hand": hand,
                    "start_frame": int(
                        window_frames[0]
                    ),
                    "features": feature_matrix
                }
            )

    return windows


def calculate_motion_change(
    current_window,
    next_window
):

    difference = (
        next_window["features"]
        -
        current_window["features"]
    )

    distances = np.linalg.norm(
        difference,
        axis=1
    )

    return float(
        np.mean(distances)
    )


def main():

    print("BASELINE EVALUATION")
    print("Tanpa GCN")
    print()

    print(
        f"Video testing : {TEST_VIDEO}.mp4"
    )

    df = load_video_data(
        TEST_VIDEO
    )

    windows = create_windows(df)

    print(
        f"Jumlah window : {len(windows)}"
    )

    motion_changes = []

    for i in range(
        len(windows) - 1
    ):

        current = windows[i]
        next_window = windows[i + 1]

        if (
            current["hand"]
            !=
            next_window["hand"]
        ):
            continue

        if (
            next_window["start_frame"]
            !=
            current["start_frame"] + 1
        ):
            continue

        motion_change = calculate_motion_change(
            current,
            next_window
        )

        motion_changes.append(
            motion_change
        )

    motion_changes = np.array(
        motion_changes
    )

    print()
    print("HASIL BASELINE")
    print(
        f"Jumlah pasangan window : "
        f"{len(motion_changes)}"
    )

    print(
        f"Mean motion change     : "
        f"{motion_changes.mean():.6f}"
    )

    print(
        f"Std motion change      : "
        f"{motion_changes.std():.6f}"
    )

    print(
        f"Min motion change      : "
        f"{motion_changes.min():.6f}"
    )

    print(
        f"Max motion change      : "
        f"{motion_changes.max():.6f}"
    )

    print()
    print(
        "Baseline selesai."
    )


if __name__ == "__main__":

    main()