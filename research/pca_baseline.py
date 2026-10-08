import os
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr


WINDOW_SIZE = 5
EMBEDDING_DIM = 32

DATA_DIR = "research/optimized"

TRAIN_VIDEOS = [
    "a",
    "abjad",
    "acak",
    "adab"
]

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

    for hand, hand_df in df.groupby("hand"):

        hand_df = (
            hand_df
            .sort_values("frame")
            .reset_index(drop=True)
        )

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
                ],
                dtype=np.float32
            )

            windows.append(
                {
                    "video": TEST_VIDEO,
                    "hand": hand,
                    "start_frame": int(
                        window_frames[0]
                    ),
                    "features": feature_matrix
                }
            )

    return windows


def prepare_training_vectors():

    all_vectors = []

    print("Mempersiapkan data PCA...")

    for video_name in TRAIN_VIDEOS:

        print(
            f"Memproses video: "
            f"{video_name}.mp4"
        )

        df = load_video_data(
            video_name
        )

        windows = create_windows(df)

        for window in windows:

            flattened = (
                window["features"]
                .reshape(-1)
            )

            all_vectors.append(
                flattened
            )

    all_vectors = np.array(
        all_vectors,
        dtype=np.float32
    )

    print(
        f"Jumlah training window : "
        f"{len(all_vectors)}"
    )

    print(
        f"Dimensi input           : "
        f"{all_vectors.shape[1]}"
    )

    return all_vectors


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

    print("PCA BASELINE")
    print()

    # Training PCA hanya menggunakan video training
    training_vectors = (
        prepare_training_vectors()
    )

    print()
    print("MEMBUAT PCA")

    pca = PCA(
        n_components=EMBEDDING_DIM
    )

    pca.fit(
        training_vectors
    )

    explained_variance = (
        pca.explained_variance_ratio_.sum()
    )

    print(
        f"Dimensi PCA              : "
        f"{EMBEDDING_DIM}"
    )

    print(
        f"Explained variance       : "
        f"{explained_variance:.6f}"
    )

    # Data testing
    print()
    print(
        f"Memproses testing: "
        f"{TEST_VIDEO}.mp4"
    )

    test_df = load_video_data(
        TEST_VIDEO
    )

    windows = create_windows(
        test_df
    )

    print(
        f"Jumlah testing window    : "
        f"{len(windows)}"
    )

    # Membuat embedding PCA
    embeddings = []

    for window in windows:

        flattened = (
            window["features"]
            .reshape(-1)
        )

        embedding = pca.transform(
            flattened.reshape(1, -1)
        )[0]

        embeddings.append(
            embedding
        )

    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )

    motion_changes = []
    embedding_changes = []

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

        motion_change = (
            calculate_motion_change(
                current,
                next_window
            )
        )

        embedding_change = np.linalg.norm(
            embeddings[i + 1]
            -
            embeddings[i]
        )

        motion_changes.append(
            motion_change
        )

        embedding_changes.append(
            embedding_change
        )

    motion_changes = np.array(
        motion_changes
    )

    embedding_changes = np.array(
        embedding_changes
    )

    pearson_r, pearson_p = pearsonr(
        motion_changes,
        embedding_changes
    )

    spearman_r, spearman_p = spearmanr(
        motion_changes,
        embedding_changes
    )

    print()
    print("HASIL EVALUASI PCA")
    print(
        f"Jumlah pasangan window : "
        f"{len(motion_changes)}"
    )

    print(
        f"Mean motion change     : "
        f"{motion_changes.mean():.6f}"
    )

    print(
        f"Mean embedding change  : "
        f"{embedding_changes.mean():.6f}"
    )

    print()
    print(
        f"Pearson correlation    : "
        f"{pearson_r:.6f}"
    )

    print(
        f"Pearson p-value        : "
        f"{pearson_p:.6f}"
    )

    print(
        f"Spearman correlation   : "
        f"{spearman_r:.6f}"
    )

    print(
        f"Spearman p-value       : "
        f"{spearman_p:.6f}"
    )

    print()
    print("PCA baseline selesai.")


if __name__ == "__main__":

    main()