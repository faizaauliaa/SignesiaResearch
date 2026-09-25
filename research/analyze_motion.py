import pandas as pd
import numpy as np

INPUT_PATH = "research/landmarks_a.csv"

df = pd.read_csv(INPUT_PATH)

print("VALIDASI TEMPORAL MOTION")

print(f"Jumlah baris : {len(df)}")
print(f"Jumlah frame : {df['frame'].nunique()}")
print(f"Tangan      : {df['hand'].unique().tolist()}")


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


print("\nDistribusi frame gap:")

print(
    df["frame_diff"]
    .value_counts()
    .sort_index()
    .head(10)
    .to_string()
)


consecutive = df[
    df["frame_diff"] == 1
].copy()

print("\nData temporal:")
print(f"Total perpindahan       : {len(df.dropna(subset=['displacement_xy']))}")
print(f"Frame benar-benar urut  : {len(consecutive)}")

motion = consecutive["displacement_xy"]

print("\nStatistik displacement X-Y")
print("(hanya frame berurutan):")

print(
    motion.describe().to_string()
)


print("\n10 displacement terbesar:")

largest = consecutive[
    ["frame", "hand", "landmark", "frame_diff",
     "dx", "dy", "displacement_xy"]
].sort_values(
    "displacement_xy",
    ascending=False
).head(10)

print(largest.to_string(index=False))


print("\nVALIDASI TEMPORAL SELESAI")