import os
import pandas as pd
import numpy as np


OPTIMIZED_DIR = "research/optimized"

videos = [
    "a",
    "abjad",
    "acak",
    "adab",
    "adaptasi"
]


for video in videos:

    path = os.path.join(
        OPTIMIZED_DIR,
        f"{video}_optimized.csv"
    )

    print()
    print(f"CHECK: {video}.csv")

    df = pd.read_csv(path)

    numeric_df = df.select_dtypes(
        include=[np.number]
    )

    nan_count = numeric_df.isna().sum().sum()
    inf_count = np.isinf(
        numeric_df.to_numpy()
    ).sum()

    print(
        f"Jumlah baris     : {len(df)}"
    )

    print(
        f"Jumlah kolom     : {len(df.columns)}"
    )

    print(
        f"Jumlah NaN       : {nan_count}"
    )

    print(
        f"Jumlah Inf       : {inf_count}"
    )

    if nan_count > 0:

        print()
        print("Kolom yang memiliki NaN:")

        nan_columns = (
            numeric_df.isna()
            .sum()
        )

        nan_columns = nan_columns[
            nan_columns > 0
        ]

        print(nan_columns)

    if inf_count > 0:

        print()
        print("Kolom yang memiliki Inf:")

        inf_columns = pd.Series(
            np.isinf(
                numeric_df.to_numpy()
            ).sum(axis=0),
            index=numeric_df.columns
        )

        print(
            inf_columns[
                inf_columns > 0
            ]
        )


print()
print("PEMERIKSAAN SELESAI")