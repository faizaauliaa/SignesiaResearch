import pandas as pd
import numpy as np
import os

ORIGINAL_DIR = "research/frame_hand"
OPTIMIZED_DIR = "research/optimized"

FILES = [
    "a_frame_hand.csv",
    "abjad_frame_hand.csv",
    "acak_frame_hand.csv",
    "adab_frame_hand.csv",
    "adaptasi_frame_hand.csv"
]

print("\nVALIDASI OPTIMASI REPRESENTASI")
print(f"Original directory : {ORIGINAL_DIR}")
print(f"Optimized directory: {OPTIMIZED_DIR}")


all_results = []


for filename in FILES:

    video_name = filename.replace("_frame_hand.csv", "")

    original_path = os.path.join(
        ORIGINAL_DIR,
        filename
    )

    optimized_path = os.path.join(
        OPTIMIZED_DIR,
        f"{video_name}_optimized.csv"
    )

    print(f"\nMemproses : {video_name}")

    if not os.path.exists(original_path):
        print(f"File original tidak ditemukan : {filename}")
        continue

    if not os.path.exists(optimized_path):
        print(
            f"File optimized tidak ditemukan : "
            f"{video_name}_optimized.csv"
        )
        continue

    original = pd.read_csv(original_path)
    optimized = pd.read_csv(optimized_path)

    distance_errors = []
    displacement_errors = []
    direction_errors = []

    # VALIDASI DISTANCE
    # distance = sqrt(relative_x^2 + relative_y^2)

    for landmark in range(21):

        x_col = f"relative_x_{landmark}"
        y_col = f"relative_y_{landmark}"
        distance_col = f"distance_from_wrist_{landmark}"

        if (
            x_col not in optimized.columns
            or y_col not in optimized.columns
            or distance_col not in original.columns
        ):
            continue

        calculated_distance = np.sqrt(
            optimized[x_col] ** 2 +
            optimized[y_col] ** 2
        )

        original_distance = original[distance_col]

        error = np.abs(
            calculated_distance -
            original_distance
        )

        distance_errors.extend(
            error.dropna().tolist()
        )

    # VALIDASI DISPLACEMENT
    # displacement = sqrt(delta_x^2 + delta_y^2)

    for landmark in range(21):

        dx_col = f"delta_x_{landmark}"
        dy_col = f"delta_y_{landmark}"
        displacement_col = f"displacement_{landmark}"

        if (
            dx_col not in optimized.columns
            or dy_col not in optimized.columns
            or displacement_col not in original.columns
        ):
            continue

        calculated_displacement = np.sqrt(
            optimized[dx_col] ** 2 +
            optimized[dy_col] ** 2
        )

        original_displacement = original[displacement_col]

        error = np.abs(
            calculated_displacement -
            original_displacement
        )

        displacement_errors.extend(
            error.dropna().tolist()
        )

    # VALIDASI DIRECTION
    # direction = atan2(delta_y, delta_x)

    for landmark in range(21):

        dx_col = f"delta_x_{landmark}"
        dy_col = f"delta_y_{landmark}"
        direction_col = f"direction_{landmark}"

        if (
            dx_col not in optimized.columns
            or dy_col not in optimized.columns
            or direction_col not in original.columns
        ):
            continue

        calculated_direction = np.degrees(
            np.arctan2(
                optimized[dy_col],
                optimized[dx_col]
            )
        )

        original_direction = original[direction_col]

        # Perbedaan sudut memperhatikan circular angle
        angle_error = np.abs(
            (
                calculated_direction -
                original_direction +
                180
            ) % 360 - 180
        )

        valid = angle_error.dropna()

        direction_errors.extend(
            valid.tolist()
        )

    # HASIL

    distance_mae = (
        np.mean(distance_errors)
        if len(distance_errors) > 0
        else np.nan
    )

    displacement_mae = (
        np.mean(displacement_errors)
        if len(displacement_errors) > 0
        else np.nan
    )

    direction_mae = (
        np.mean(direction_errors)
        if len(direction_errors) > 0
        else np.nan
    )

    result = {
        "video": video_name,
        "distance_mae": distance_mae,
        "displacement_mae": displacement_mae,
        "direction_mae": direction_mae,
        "distance_samples": len(distance_errors),
        "displacement_samples": len(displacement_errors),
        "direction_samples": len(direction_errors)
    }

    all_results.append(result)

    print(
        f"Distance MAE      : "
        f"{distance_mae:.10f}"
    )

    print(
        f"Displacement MAE  : "
        f"{displacement_mae:.10f}"
    )

    print(
        f"Direction MAE     : "
        f"{direction_mae:.10f} degrees"
    )

# REKAP
if len(all_results) > 0:

    results_df = pd.DataFrame(all_results)

    print("\nREKAP VALIDASI")

    print(
        results_df[
            [
                "video",
                "distance_mae",
                "displacement_mae",
                "direction_mae"
            ]
        ].round(10).to_string(index=False)
    )

    print("\nRATA-RATA ERROR")

    print(
        f"Distance MAE     : "
        f"{results_df['distance_mae'].mean():.10f}"
    )

    print(
        f"Displacement MAE : "
        f"{results_df['displacement_mae'].mean():.10f}"
    )

    print(
        f"Direction MAE    : "
        f"{results_df['direction_mae'].mean():.10f} degrees"
    )

    output_path = os.path.join(
        OPTIMIZED_DIR,
        "optimization_validation.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(f"\nHasil validasi : {output_path}")


print("\nVALIDASI SELESAI")