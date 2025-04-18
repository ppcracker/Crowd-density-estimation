# 📂 Step 1: Frame Extraction from Uploaded Videos

!pip install opencv-python --quiet

import cv2
import os
from google.colab import files

# Upload Video Files
uploaded = files.upload()

# Create output directory
output_dir = "Extracted_Frames"
os.makedirs(output_dir, exist_ok=True)

# Extract frames
frame_interval = 10  # Save every 10th frame

for video_file in uploaded.keys():
    cap = cv2.VideoCapture(video_file)
    count = 0
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            filename = f"{output_dir}/{video_file.split('.')[0]}_frame{frame_num:04d}.jpg"
            cv2.imwrite(filename, frame)
            frame_num += 1
        count += 1
    cap.release()

print(f"Frames saved in {output_dir}")

# Zip extracted frames
!zip -r extracted_frames.zip Extracted_Frames
files.download("extracted_frames.zip")
