# Visual Odometry with OpenCV and KITTI Dataset

## 🚗 Project Overview
This project implements a basic Visual Odometry (VO) system using OpenCV and the KITTI dataset. The script tracks camera movement and trajectory by analyzing consecutive image frames.

## 📦 Prerequisites

### Software Requirements
- Python 3.7+
- OpenCV
- NumPy

### Installation
```bash
# Clone the repository
git clone https://github.com/mehmetsolves/MotionEstimation
cd MotionEstimation

# Install required packages
pip install opencv-python numpy
pip install opencv-contrib-python  # For additional feature detectors
```

## 🔧 Dataset Preparation

### KITTI Dataset
1. Download the KITTI Visual Odometry dataset from the [official KITTI website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
2. Extract the dataset to a known location on your computer
3. Update the file paths in the script to match your dataset location

## 🚀 Usage

### Running the Script
```bash
python motiontrack.py
```

### Configuration
- Modify `motiontrack.py` to change:
  - Dataset path
  - Feature detection method
  - Camera parameters

## 🖥️ Output
The script generates two windows:
1. **Road Facing Camera**: Current camera frame
2. **Trajectory**: Visual representation of camera movement

## 📊 Features
- ORB Feature Detection
- Optical Flow Tracking
- Essential Matrix Estimation
- Camera Pose Reconstruction

## 🛠️ Troubleshooting
- Ensure OpenCV is correctly installed
- Verify dataset file paths
- Check Python and library versions

## 🔬 Limitations
- Designed for specific KITTI dataset format
- Requires calibrated camera parameters
- Performance may vary with different datasets

## 📝 References
- [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/)
- [OpenCV Documentation](https://docs.opencv.org/)

