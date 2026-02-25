import numpy as np
import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_DIR = os.path.join(BASE_DIR, "datasets", "ucsd", "vidf")
OUTPUT_DIR = os.path.join(BASE_DIR, "extracted_frames")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# UCSD video properties
WIDTH = 238
HEIGHT = 158
FRAME_SIZE = WIDTH * HEIGHT * 3 // 2  # YUV420

print("📂 Looking for YUV videos in:", VIDEO_DIR)

for video_file in os.listdir(VIDEO_DIR):
    if video_file.endswith(".y") or video_file.endswith(".yuv"):
        video_path = os.path.join(VIDEO_DIR, video_file)
        video_name = os.path.splitext(video_file)[0]

        frame_output_dir = os.path.join(OUTPUT_DIR, video_name)
        os.makedirs(frame_output_dir, exist_ok=True)

        with open(video_path, "rb") as f:
            frame_count = 0
            while True:
                raw = f.read(FRAME_SIZE)
                if len(raw) < FRAME_SIZE:
                    break

                yuv = np.frombuffer(raw, dtype=np.uint8)
                yuv = yuv.reshape((HEIGHT * 3 // 2, WIDTH))

                bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

                frame_path = os.path.join(
                    frame_output_dir, f"frame_{frame_count}.jpg"
                )
                cv2.imwrite(frame_path, bgr)
                frame_count += 1

        print(f"✅ Extracted {frame_count} frames from {video_file}")

print("🎉 YUV frame extraction completed successfully.")
