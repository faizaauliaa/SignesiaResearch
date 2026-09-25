import cv2
import csv
import os
import mediapipe as mp

DATASET_DIR = "dataset/data/dataset"
MODEL_PATH = "models/hand_landmarker.task"
OUTPUT_DIR = "research/batch_test"

os.makedirs(OUTPUT_DIR, exist_ok=True)

videos = [
    "a.mp4",
    "abjad.mp4",
    "acak.mp4",
    "adab.mp4",
    "adaptasi.mp4"
]

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def create_landmarker():

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    return HandLandmarker.create_from_options(options)


for video_name in videos:

    video_path = os.path.join(
        DATASET_DIR,
        video_name
    )

    if not os.path.exists(video_path):
        print(f"[SKIP] {video_name} tidak ditemukan")
        continue

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[GAGAL] {video_name}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    output_name = (
        os.path.splitext(video_name)[0] + ".csv"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    frame_number = 0
    detected_frames = 0
    total_landmarks = 0

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow([
            "frame",
            "timestamp_ms",
            "hand",
            "landmark",
            "x",
            "y",
            "z"
        ])

        # Landmarker BARU untuk setiap video
        with create_landmarker() as landmarker:

            while True:

                success, frame = cap.read()

                if not success:
                    break

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                timestamp_ms = int(
                    (frame_number / fps) * 1000
                ) if fps > 0 else frame_number

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                if result.hand_landmarks:

                    detected_frames += 1

                    for hand_index, hand_landmarks in enumerate(
                        result.hand_landmarks
                    ):

                        hand_label = "Unknown"

                        if (
                            result.handedness
                            and hand_index < len(result.handedness)
                            and result.handedness[hand_index]
                        ):
                            hand_label = (
                                result.handedness[
                                    hand_index
                                ][0].category_name
                            )

                        for landmark_index, landmark in enumerate(
                            hand_landmarks
                        ):

                            writer.writerow([
                                frame_number,
                                timestamp_ms,
                                hand_label,
                                landmark_index,
                                landmark.x,
                                landmark.y,
                                landmark.z
                            ])

                            total_landmarks += 1

                frame_number += 1

    cap.release()

    print(
        f"{video_name:15} | "
        f"Frame: {frame_number:4} | "
        f"Terdeteksi: {detected_frames:4} | "
        f"Landmark: {total_landmarks:5}"
    )


print("\nBatch test selesai.")