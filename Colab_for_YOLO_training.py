# 2. Colab for YOLOv5 Training

!git clone https://github.com/ultralytics/yolov5.git
%cd yolov5
!pip install -r requirements.txt

# Upload your dataset (images + YOLO labels) or use Roboflow API

# Create custom.yaml for dataset
custom_yaml = '''
train: ../dataset/train/images
val: ../dataset/valid/images

nc: 1
names: ['person']
'''
with open("custom.yaml", "w") as f:
    f.write(custom_yaml)

# Start training
!python train.py --img 640 --batch 16 --epochs 30 --data custom.yaml --weights yolov5s.pt --name crowd_detect

# After training, best_model.pt will be generated inside runs/train/crowd_detect/weights

