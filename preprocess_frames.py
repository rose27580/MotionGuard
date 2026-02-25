import os
import cv2
import numpy as np

# Paths
FRAMES_DIR = "frames"
PROCESSED_DIR = "processed_frames"

# Create processed_frames folder if not exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Parameters
IMG_SIZE = 128      # Resize to 128x128
FRAME_INTERVAL = 5  # Take every 5th frame

frame_files = sorted(os.listdir(FRAMES_DIR))
processed_count = 0

for i, frame_name in enumerate(frame_files):
    # Take every Nth frame
    if i % FRAME_INTERVAL != 0:
        continue

    frame_path = os.path.join(FRAMES_DIR, frame_name)
    frame = cv2.imread(frame_path)

    if frame is None:
        continue

    # Resize
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    # Normalize (0–255 → 0–1)
    frame = frame / 255.0

    # Save as numpy array
    save_path = os.path.join(PROCESSED_DIR, f"frame_{processed_count}.npy")
    np.save(save_path, frame)

    processed_count += 1

print(f"Preprocessing complete. {processed_count} frames saved.")
