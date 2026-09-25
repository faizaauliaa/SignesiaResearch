import pandas as pd

INPUT_PATH = "research/landmarks_a.csv"

df = pd.read_csv(INPUT_PATH)

print("=" * 50)
print("ANALISIS LANDMARK")
print("=" * 50)


print(f"Jumlah baris       : {len(df)}")
print(f"Frame unik         : {df['frame'].nunique()}")
print(f"Landmark unik      : {df['landmark'].nunique()}")
print(f"Tangan             : {df['hand'].unique().tolist()}")

frame_hand_count = (
    df.groupby(["frame", "hand"])
      .size()
      .reset_index(name="jumlah_landmark")
)

print("\nJumlah landmark per frame dan tangan:")
print(frame_hand_count.to_string(index=False))

hands_per_frame = (
    df.groupby("frame")["hand"]
      .nunique()
)

print("\nJumlah tangan per frame:")
print(hands_per_frame.value_counts().sort_index())

all_frames = set(range(int(df["frame"].max()) + 1))
detected_frames = set(df["frame"].unique())

missing_frames = sorted(all_frames - detected_frames)

print("\nFrame tanpa deteksi tangan:")
print(missing_frames)

print("\nStatistik koordinat:")
print(
    df[["x", "y", "z"]]
    .describe()
    .to_string()
)

print("\nANALISIS SELESAI")