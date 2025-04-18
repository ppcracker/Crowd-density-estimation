# 📓 People Detection & Density Estimation Project

# ---- SETUP ----
# Install required libraries
!pip install ultralytics roboflow opencv-python matplotlib --quiet

# Import libraries
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

# ---- STEP 1: Frame Extraction Example ----
# Simulate frame extraction from video

# Dummy download (public domain video)
!gdown --id 1W8uEfr9waKkIok-7KZJmZdj8Ap_GPxJd -O sample_video.mp4

# Create directory
os.makedirs('frames', exist_ok=True)

# Extract frames
video = cv2.VideoCapture('sample_video.mp4')
frame_count = 0
success, image = video.read()

while success:
    if frame_count % 30 == 0:  # Save 1 frame per second if video is 30 fps
        cv2.imwrite(f'frames/frame_{frame_count}.jpg', image)
    success, image = video.read()
    frame_count += 1

print(f"Extracted {len(os.listdir('frames'))} frames.")

# Show sample frames
sample_frames = os.listdir('frames')[:5]
for frame in sample_frames:
    img = cv2.imread(f'frames/{frame}')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# ---- STEP 2: Annotation ----
# Manual step: Upload 'frames/' to Roboflow or LabelImg
# Annotate people -> Export as YOLO format dataset
# Download annotated dataset as ZIP and unzip

# (Demo) We will use a free Roboflow public dataset for now
from roboflow import Roboflow
rf = Roboflow(api_key="your_roboflow_api_key")  # Optional if you want real Roboflow datasets
project = rf.workspace("your-workspace").project("your-project")
dataset = project.version(1).download("yolov8")

# ---- STEP 3: Model Training ----

# Load model
model = YOLO('yolov8n.pt')  # YOLOv8 nano model (small & fast)

# Train
model.train(data="/content/dataset/data.yaml", epochs=20, imgsz=640)

# ---- STEP 4: Testing and Counting People ----

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Test frames
test_images = os.listdir('frames')[:10]
people_counts = []

for frame in test_images:
    results = model.predict(source=f'frames/{frame}', conf=0.5)
    people = len(results[0].boxes)
    people_counts.append(people)

# ---- STEP 5: Classify Density ----

density_labels = []
for count in people_counts:
    if count <= 10:
        density_labels.append("Low")
    elif 11 <= count <= 25:
        density_labels.append("Medium")
    else:
        density_labels.append("High")

# Create DataFrame
df = pd.DataFrame({
    'Frame': test_images,
    'People Count': people_counts,
    'Density Category': density_labels
})

print(df)

# ---- STEP 6: Plotting Graphs ----

# People Count over frames
plt.figure(figsize=(10,6))
plt.plot(df['Frame'], df['People Count'], marker='o')
plt.title('People Count over Frames')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Density Distribution
density_dist = df['Density Category'].value_counts()

plt.figure(figsize=(6,6))
plt.bar(density_dist.index, density_dist.values, color=['green', 'orange', 'red'])
plt.title('Density Category Distribution')
plt.xlabel('Density Category')
plt.ylabel('Number of Frames')
plt.show()

# ---- STEP 7: Final Submission ----
# 1. Upload extracted frames + annotated labels to Google Drive
# 2. Upload code and screenshots to GitHub
# 3. Share public links to both
