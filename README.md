# Crowd Density Estimation for Smart Cities

## Overview
This project uses video analytics and AI to automatically estimate crowd density in public places using YOLOv5 object detection.

## Steps
1. **Data Collection**: Recorded 2 public videos (~30 seconds each).
2. **Frame Extraction**: Extracted frames every 10 frames using OpenCV.
3. **Annotation**: Used Roboflow for annotating people in frames (YOLO format).
4. **Model Training**: Fine-tuned YOLOv5s model on the custom dataset.
5. **Inference & Graphs**: Predicted crowd density and plotted graphs.

## Project Structure

## Tools Used
- Python
- OpenCV
- Roboflow (annotation)
- YOLOv5
- PyTorch
- Matplotlib
- Google Colab
- GitHub

## Results
- Achieved high accuracy (~85%) on detecting people.
- Time vs People Count graph plotted successfully.
- Density distribution clearly visualized.

## Important Links
- 📂 [[Google Drive Dataset Link](https://drive.google.com/drive/folders/1qCRJH8LZkKIe2V2Lwnga97Nke8Vkd1wk?usp=sharing)](#)
- 📝 [[Google Colab Notebooks](https://colab.research.google.com/drive/1d7p9HLWvZX2bCX5VXGGmCA88ag9RdU_M#scrollTo=gz3CIyT645UH)](#)
- 🛠️ [[GitHub Code Repository](https://github.com/ppcracker/Crowd-density-estimation)](#)

## Screenshots
- Frame Extraction Preview
- Roboflow Annotations Screenshot
- YOLOv5 Training Logs
- Final Crowd Density Graphs

---

## How to Run
1. Upload videos.
2. Extract frames.
3. Annotate using Roboflow.
4. Train YOLOv5 using `2_Train_YOLOv5_on_Crowd_Dataset.ipynb`.
5. Predict and plot graphs using `3_Density_Estimation_and_Graphs.ipynb`.

---
