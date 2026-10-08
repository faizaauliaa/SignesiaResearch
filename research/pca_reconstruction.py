import os
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error


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
            f"NaN atau Inf ditemukan "
            f"pada {video_name}"
        )

    return df


def create_feature_vector(row):

    features = []

    for feature_type in FEATURE_TYPES:

        for landmark in range(21):

            features.append(
                row[
                    f"{feature_type}_{landmark}"
                ]
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

            frames = (
                window["frame"].to_numpy()
            )

            expected = np.arange(
                frames[0],
                frames[0] + WINDOW_SIZE
            )

            if not np.array_equal(
                frames,
                expected
            ):
                continue

            vector = np.array(
                [
                    create_feature_vector(row)
                    for _, row in window.iterrows()
                ],
                dtype=np.float32
            ).reshape(-1)

            windows.append(vector)

    return windows


def main():

    print("PCA LANDMARK RECONSTRUCTION")
    print()

    training_vectors = []

    for video_name in TRAIN_VIDEOS:

        print(
            f"Memproses training: "
            f"{video_name}.mp4"
        )

        df = load_video_data(
            video_name
        )

        windows = create_windows(df)

        training_vectors.extend(
            windows
        )

    training_vectors = np.array(
        training_vectors,
        dtype=np.float32
    )

    print()
    print(
        f"Training windows : "
        f"{len(training_vectors)}"
    )

    print(
        f"Input dimension   : "
        f"{training_vectors.shape[1]}"
    )

    print()
    print("Training PCA...")

    pca = PCA(
        n_components=EMBEDDING_DIM
    )

    pca.fit(
        training_vectors
    )

    print(
        f"Embedding dimension : "
        f"{EMBEDDING_DIM}"
    )

    print(
        f"Explained variance  : "
        f"{pca.explained_variance_ratio_.sum():.6f}"
    )

    print()
    print(
        f"Testing: "
        f"{TEST_VIDEO}.mp4"
    )

    test_df = load_video_data(
        TEST_VIDEO
    )

    test_windows = create_windows(
        test_df
    )

    test_vectors = np.array(
        test_windows,
        dtype=np.float32
    )

    print(
        f"Testing windows : "
        f"{len(test_vectors)}"
    )

    # Encode ke 32 dimensi
    embeddings = pca.transform(
        test_vectors
    )

    # Decode kembali ke 420 dimensi
    reconstructed = pca.inverse_transform(
        embeddings
    )

    mae = mean_absolute_error(
        test_vectors,
        reconstructed
    )

    rmse = np.sqrt(
        mean_squared_error(
            test_vectors,
            reconstructed
        )
    )

    absolute_errors = np.abs(
        test_vectors
        -
        reconstructed
    )

    print()
    print("HASIL RECONSTRUCTION")
    print(
        f"MAE  : {mae:.8f}"
    )

    print(
        f"RMSE : {rmse:.8f}"
    )

    print()
    print(
        "PCA reconstruction selesai."
    )


if __name__ == "__main__":

    main()