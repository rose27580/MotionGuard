import os
import numpy as np

# Path to processed frames
PROCESSED_DIR = "processed_frames"

# Sequence length (number of frames per sequence)
SEQUENCE_LENGTH = 10

# Load all frame files
frame_files = sorted(os.listdir(PROCESSED_DIR))

sequences = []

for i in range(len(frame_files) - SEQUENCE_LENGTH + 1):
    sequence = []
    for j in range(SEQUENCE_LENGTH):
        frame_path = os.path.join(PROCESSED_DIR, frame_files[i + j])
        frame = np.load(frame_path)
        sequence.append(frame)

    sequences.append(sequence)

# Convert to numpy array
X = np.array(sequences)

# Save sequences
np.save("frame_sequences.npy", X)

print("Sequence creation complete.")
print("Total sequences:", X.shape[0])
print("Sequence shape:", X.shape)
