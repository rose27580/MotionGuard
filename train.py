import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, TimeDistributed
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 64        # Reduced from 128
SEQ_LEN = 10
MAX_SEQUENCES_PER_VIDEO = 50  # Limit sequences
DATA_PATH = "training_data"
MODEL_SAVE_PATH = "model/motionguard_model.h5"

os.makedirs("model", exist_ok=True)

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % 10 == 0:  # sample every 10 frames
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame.astype("float32") / 255.0
            frames.append(frame)

        count += 1

    cap.release()
    return frames

X = []
y = []

for label, category in enumerate(["normal", "motion"]):
    folder = os.path.join(DATA_PATH, category)

    for file in os.listdir(folder):
        video_path = os.path.join(folder, file)
        frames = extract_frames(video_path)

        if len(frames) < SEQ_LEN:
            continue

        sequences_added = 0

        for i in range(len(frames) - SEQ_LEN + 1):
            sequence = frames[i:i+SEQ_LEN]
            X.append(sequence)
            y.append(label)

            sequences_added += 1
            if sequences_added >= MAX_SEQUENCES_PER_VIDEO:
                break

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("Total sequences:", len(X))

model = Sequential()
model.add(TimeDistributed(Conv2D(16, (3,3), activation='relu'),
                          input_shape=(SEQ_LEN, IMG_SIZE, IMG_SIZE, 3)))
model.add(TimeDistributed(MaxPooling2D((2,2))))
model.add(TimeDistributed(Flatten()))

model.add(LSTM(32))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=5, batch_size=8)

model.save(MODEL_SAVE_PATH)

print("Model saved successfully!")
