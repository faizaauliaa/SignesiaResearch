import pandas as pd
import numpy as np
import os

INPUT_FOLDER = "research/batch_test"

FILES = [
    "a.csv",
    "abjad.csv",
    "acak.csv",
    "adab.csv",
    "adaptasi.csv"
]

print("BATCH TEMPORAL MOTION ANALYSIS")

all_results = []

for filename in FILES:

    path = os.path.join(INPUT_FOLDER, filename)

    if not os.path.exists(path):
        print(f"\nFile tidak ditemukan: {path}")
        continue

    df = pd.read_csv(path)

    # Urutkan landmark berdasarkan tangan, landmark, dan frame
    df = df.sort_values(
        ["hand", "landmark", "frame"]
    ).reset_index(drop=True)


    df["frame_diff"] = df.groupby(
        ["hand", "landmark"]
    )["frame"].diff()


    df["dx"] = df.groupby(
        ["hand", "landmark"]
    )["x"].diff()

    df["dy"] = df.groupby(
        ["hand", "landmark"]
    )["y"].diff()

    df["displacement_xy"] = np.sqrt(
        df["dx"] ** 2 +
        df["dy"] ** 2
    )

    consecutive = df[
        df["frame_diff"] == 1
    ].copy()

    motion = consecutive["displacement_xy"].dropna()


    if len(motion) > 0:

        mean_motion = motion.mean()
        median_motion = motion.median()
        std_motion = motion.std()
        max_motion = motion.max()

        q25 = motion.quantile(0.25)
        q75 = motion.quantile(0.75)

    else:

        mean_motion = 0
        median_motion = 0
        std_motion = 0
        max_motion = 0
        q25 = 0
        q75 = 0


    total_transition = len(
        df.dropna(subset=["frame_diff"])
    )

    consecutive_transition = len(consecutive)

    gap_transition = (
        total_transition -
        consecutive_transition
    )

    all_results.append({
        "video": filename.replace(".csv", ""),
        "frames": df["frame"].nunique(),
        "landmarks": df["landmark"].nunique(),
        "motion_count": len(motion),
        "frame_gap": gap_transition,
        "mean": mean_motion,
        "median": median_motion,
        "std": std_motion,
        "q25": q25,
        "q75": q75,
        "max": max_motion
    })


results = pd.DataFrame(all_results)

print("\nHASIL PER VIDEO")

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


print("\nRATA-RATA 5 VIDEO")

numeric_columns = [
    "frames",
    "motion_count",
    "frame_gap",
    "mean",
    "median",
    "std",
    "q25",
    "q75",
    "max"
]

print(
    results[numeric_columns]
    .mean()
    .to_string(
        float_format=lambda x: f"{x:.6f}"
    )
)


output_path = "research/batch_motion_results.csv"

results.to_csv(
    output_path,
    index=False
)

print("\nHasil disimpan ke:")
print(output_path)

print("\nANALISIS SELESAI")