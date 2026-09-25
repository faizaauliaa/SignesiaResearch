import cv2
import csv
import os
import mediapipe as mp


VIDEO_PATH = "dataset/data/dataset/a.mp4"
MODEL_PATH = "models/hand_landmarker.task"
OUTPUT_PATH = "research/landmarks_a.csv"


if not os.path.exists(VIDEO_PATH):
    print(f"Video tidak ditemukan: {VIDEO_PATH}")
    exit()

if not os.path.exists(MODEL_PATH):
    print(f"Model tidak ditemukan: {MODEL_PATH}")
    exit()


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

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


cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Video gagal dibuka.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("VIDEO BERHASIL DIBUKA")
print(f"Video       : {VIDEO_PATH}")
print(f"FPS         : {fps}")
print(f"Total frame : {frame_count}")
print()

os.makedirs("research", exist_ok=True)

with open(
    OUTPUT_PATH,
    mode="w",
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

    with HandLandmarker.create_from_options(options) as landmarker:

        frame_number = 0
        detected_frames = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            # BGR → RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Buat MediaPipe Image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Timestamp video
            timestamp_ms = int(
                (frame_number / fps) * 1000
            ) if fps > 0 else frame_number

            # Deteksi landmark
            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            # Jika ada tangan
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

                    # 21 landmark setiap tangan
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

            frame_number += 1

            # Progress
            if frame_number % 30 == 0:
                print(
                    f"Memproses frame "
                    f"{frame_number}/{frame_count}"
                )


cap.release()

print()
print("EKSTRAKSI SELESAI")
print(f"Total frame      : {frame_number}")
print(f"Frame ada tangan : {detected_frames}")
print(f"CSV              : {OUTPUT_PATH}")