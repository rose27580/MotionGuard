import numpy as np
import matplotlib.pyplot as plt

# Load one processed frame
frame = np.load("processed_frames/frame_0.npy")

print("Frame shape:", frame.shape)

# Display the frame
plt.imshow(frame)
plt.axis("off")
plt.title("Processed Frame")
plt.show()
