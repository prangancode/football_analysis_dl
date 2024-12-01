# Football Analysis System Using YOLO, OpenCV, and Python

This repository contains a project demonstrating the development of a football analysis system using YOLO, OpenCV, and Python. The goal is to utilize state-of-the-art object detection and computer vision techniques to analyze football matches, detecting and tracking players, referees, and the football across frames. The project also includes advanced features such as team assignment based on jersey colors, movement estimation, and speed/distance calculations.

## Overview

This project combines multiple machine learning and computer vision concepts to:
- Detect players, referees, and football using YOLO (You Only Look Once).
- Track detected objects across video frames.
- Assign players to teams using jersey color segmentation via K-Means clustering.
- Estimate camera movement using optical flow.
- Perform perspective transformation to calculate real-world distances and speeds of players.

## Table of Contents

1. [Datasets](#datasets)
2. [Features](#features)
3. [Setup](#setup)
4. [Project Workflow](#project-workflow)
5. [Results](#results)
6. [References](#references)

---

## Datasets

### Primary Dataset
- **Name**: Roboflow Football Player Detection Dataset
- **Link**: [Roboflow Dataset](https://universe.roboflow.com/roboflow/football-player-detection)

### Additional Resources
- **Kaggle Dataset**: [Football Video Dataset](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/data)
- **Video Link**: [Google Drive Video](https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view)

These datasets are used to train the YOLO model and validate player detection, tracking, and team classification functionalities.

---

## Features

1. **Object Detection with YOLO**
   - Detects players, referees, and football in each video frame.
   - Trained on custom datasets to improve accuracy.

2. **Object Tracking**
   - Tracks players, referees, and the ball across consecutive video frames.
   - Utilizes advanced tracking algorithms to maintain object consistency.

3. **Team Assignment Using K-Means**
   - Identifies team affiliation of players based on jersey colors.
   - Clusters pixel data to segment colors and assign players to teams.

4. **Camera Movement Estimation**
   - Optical flow techniques to measure and adjust for camera movements.
   - Ensures accurate tracking by accounting for perspective shifts.

5. **Real-World Speed and Distance Calculation**
   - Perspective transformation to map pixel-based positions to real-world coordinates.
   - Calculates speeds and distances covered by players in meters.

---

## Setup

### Prerequisites
1. Python (>= 3.7)
2. Libraries:
   - OpenCV
   - Numpy
   - Pandas
   - Matplotlib
   - Scikit-learn
   - PyTorch
   - Ultralytics YOLO

### Steps to Run the Project in Google Colab

1. **Open the Notebook in Google Colab**:
   - Click on the following link to open the notebook directly in Google Colab:
     [football_analysis_system_using_yolo_opencv_python.ipynb](https://github.com/prangancode/football_analysis_dl/blob/main/football_analysis_system_using_yolo_opencv_python.ipynb)

2. **Set Up the Environment**:
   - Ensure you have access to the required datasets. Upload them to your Google Drive or use the dataset links provided in the notebook.

3. **Mount Google Drive**:
   - Add the following code cell in the notebook to mount your Google Drive:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```

4. **Install Required Libraries**:
   - Add and execute the following code cell to install necessary dependencies:
     ```python
     !pip install -r requirements.txt
     ```

5. **Upload or Link Datasets**:
   - If your datasets are stored in Google Drive, ensure the notebook points to the correct file paths in your Drive.
   - If using an external link (e.g., Roboflow or Kaggle), follow the dataset download instructions provided in the notebook.

6. **Run the Notebook**:
   - Once everything is set up, run the code cells sequentially to execute the football detection and analysis system.

7. **Save Outputs**:
   - Save any outputs, such as processed videos or data, to your Google Drive for easy access:
     ```python
     from shutil import copyfile
     copyfile('output_file_path', '/content/drive/My Drive/desired_folder/output_file_name')
     ``` 

This setup allows seamless execution of the project in a cloud-based environment.

---

## Project Workflow

### Step 1: Object Detection with YOLO
- **Model Initialization**: Load pre-trained YOLO weights for player, referee, and football detection.
- **Training**: Fine-tune the YOLO model on the custom Roboflow football dataset.
- **Inference**: Perform real-time object detection, generating bounding boxes for detected objects.

### Step 2: Object Tracking
- **Tracking Algorithms**: Implement object tracking using OpenCV and Python.
- **Filtering Detections**: Apply confidence thresholds to eliminate false positives.

### Step 3: Team Assignment
- **Color Clustering**: Use K-Means clustering to segment jersey colors.
- **Team Identification**: Assign players to teams based on dominant color clusters.

### Step 4: Camera Movement Estimation
- **Optical Flow**: Measure camera movement using frame-by-frame feature tracking.
- **Position Adjustment**: Adjust player positions to compensate for camera motion.

### Step 5: Perspective Transformation
- Map field dimensions from video frames to real-world coordinates.
- Implement a transform function to calculate accurate player positions.

### Step 6: Speed and Distance Calculation
- Calculate the distance covered by players over time.
- Estimate real-world speeds by combining position data and timestamps.

---

## Results

The project outputs the following:
1. Annotated videos with bounding boxes for detected players, referees, and the ball.
2. Team affiliations based on jersey colors.
3. Heatmaps and trajectories showing player movements.
4. Real-time speed and distance metrics for individual players.

---

## References

- [Roboflow Football Dataset](https://universe.roboflow.com/roboflow/football-player-detection)
- [Kaggle Bundesliga Dataset](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/data)
- [YOLOv8 Documentation](https://docs.ultralytics.com)
- [OpenCV Documentation](https://docs.opencv.org/)

For more information, refer to the complete code in the [notebook](https://github.com/prangancode/football_analysis_dl/blob/main/football_analysis_system_using_yolo_opencv_python.ipynb).
