# Crowd Density Estimation - Smart City Monitoring

## Project Overview
This project simulates an IoT-based surveillance system that estimates crowd density using video analytics and AI models.

## Steps
1. **Data Collection**: Captured 2 real-world public videos (~30s each).
2. **Frame Extraction**: Extracted 500+ frames using OpenCV.
3. **Annotation**: Annotated all frames using Roboflow in YOLO format.
4. **Model Training**: Fine-tuned YOLOv5 on the annotated dataset.
5. **Density Estimation**: Performed inference to classify frames into Low, Medium, High crowd density.
6. **Reporting**: Generated graphs for Time vs People Count and Density Distribution.

## Tools Used
- Python, OpenCV, Roboflow, YOLOv5, PyTorch
- Google Colab for training
- GitHub for version control
- Google Drive for dataset hosting

## Submission
- [Google Drive Link](#) (dataset, model, graphs)
- [GitHub Repository](#) (code and instructions)

## Results
- Achieved ~85% accuracy on custom data
- Successful classification into crowd density categories
