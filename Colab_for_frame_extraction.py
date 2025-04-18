# 1. Colab for Frame Extraction

!pip install opencv-python

import cv2
import os

# Upload your video to Colab files first
from google.colab import files
uploaded = files.upload()

video_path = list(uploaded.keys())[0]
output_dir = "Extracted_Frames"
os.makedirs(output_dir, exist_ok=True)

# Extract frames
cap = cv2.VideoCapture(video_path)
count = 0
frame_interval = 10  # extract every 10th frame

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if count % frame_interval == 0:
        filename = f"{output_dir}/frame{count:04d}.jpg"
        cv2.imwrite(filename, frame)
    count += 1

cap.release()
print(f"Frames saved in {output_dir}")

# Download the extracted frames as a ZIP
!zip -r extracted_frames.zip Extracted_Frames
files.download("extracted_frames.zip")

# --- End of Frame Extraction Colab ---
