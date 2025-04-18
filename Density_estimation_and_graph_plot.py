# 📂 Step 3: Test Detection + Crowd Graph Plotting

import torch
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from google.colab import files

# Upload best.pt model and frames
uploaded = files.upload()

model_path = [f for f in uploaded.keys() if f.endswith('.pt')][0]

# Load model
from yolov5.models.common import DetectMultiBackend
model = DetectMultiBackend(model_path)

# Upload frames (test images)
uploaded = files.upload()
frame_files = list(uploaded.keys())

people_counts = []

# Inference
for frame in frame_files:
    img = cv2.imread(frame)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    img_resized = img_resized.transpose((2, 0, 1))
    img_resized = torch.from_numpy(img_resized).float().unsqueeze(0) / 255.0

    pred = model(img_resized, augment=False, visualize=False)[0]
    num_people = pred.shape[1] if pred is not None else 0
    people_counts.append(num_people)

# Plotting People Count Over Time
plt.figure(figsize=(10,6))
plt.plot(range(len(people_counts)), people_counts, marker='o')
plt.title('People Count over Frames')
plt.xlabel('Frame Number')
plt.ylabel('People Count')
plt.grid(True)
plt.savefig('time_vs_people.png')
plt.show()

# Density Distribution
plt.figure(figsize=(6,6))
plt.hist(people_counts, bins=[0,10,20,30,40,50], edgecolor='black')
plt.title('Density Distribution')
plt.xlabel('People Count Range')
plt.ylabel('Frequency')
plt.savefig('density_distribution.png')
plt.show()

# Download Graphs
files.download('time_vs_people.png')
files.download('density_distribution.png')
