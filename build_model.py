import numpy as np
import tensorflow as tf

# Load sequences
X = np.load("frame_sequences.npy")

print("Input shape:", X.shape)

# Temporary labels (for pipeline testing)
y = np.ones((X.shape[0], 1))

# Model
model = tf.keras.models.Sequential()

# CNN (spatial feature extraction)
model.add(tf.keras.layers.TimeDistributed(
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    input_shape=(X.shape[1], 128, 128, 3)
))
model.add(tf.keras.layers.TimeDistributed(
    tf.keras.layers.MaxPooling2D((2, 2))
))
model.add(tf.keras.layers.TimeDistributed(
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
))
model.add(tf.keras.layers.TimeDistributed(
    tf.keras.layers.MaxPooling2D((2, 2))
))
model.add(tf.keras.layers.TimeDistributed(
    tf.keras.layers.Flatten()
))

# BiLSTM (temporal feature learning)
model.add(tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(64)
))

# Classification layers
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Summary
model.summary()

# Train (small epochs – laptop safe)
model.fit(
    X, y,
    epochs=3,
    batch_size=2
)

# Save model
model.save("model/motionguard_model.h5")

print("✅ Model training complete and saved.")
