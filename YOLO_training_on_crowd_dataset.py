# 📂 Step 2: Train YOLOv5 on your Annotated Dataset

# Clone YOLOv5 repo
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!pip install -r requirements.txt --quiet

# Upload Dataset
from google.colab import files
uploaded = files.upload()  # Upload your dataset ZIP (exported in YOLO format)

# Unzip Dataset
import zipfile
import os
dataset_zip = list(uploaded.keys())[0]
with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
    zip_ref.extractall('dataset')

# Create custom.yaml file
custom_yaml = '''
train: dataset/train/images
val: dataset/valid/images

nc: 1
names: ['person']
'''
with open("custom.yaml", "w") as f:
    f.write(custom_yaml)

# Train YOLOv5
!python train.py --img 640 --batch 16 --epochs 30 --data custom.yaml --weights yolov5s.pt --name crowd_detect

# Best weights will be inside runs/train/crowd_detect/weights/best.pt
