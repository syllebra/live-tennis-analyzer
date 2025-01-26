# Tennis Analytics System

A computer vision system for real-time tennis player and ball tracking using a transformer (RT-DETRv2) for object detection and keypoint detection.

![Sample Video Analysis](videos/output_1.gif)
## Overview
This system analyzes tennis video streams in real-time to:
- Detect and track tennis players and ball
- Project detections onto a 2D tennis court visualization (minimap)
- Track player positions and movements
- Track ball trajectory with interpolation between detections

It uses separate threads for decoding frames from the stream and for processing the frames. The frame-decoding thread dynamically skips frames based on the speed of the processing thread. This enables real-time analysis even if the processing is slower than real-time (obviously, this only makes sense for applications where it's not critical that every frame be processed). With an NVIDIA GeForce GTX 1650 Ti, [RT-DETRv2-M*](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch) and a frame resolution of 1920x1080, my frame-rate was about 15 FPS, 20 FPS for 1280x720 and 25 FPS for 640x480.

## Setup

### Prerequisites
- Python 3.8+
- CUDA and TensorRT installed
- Required model weights:
  - RT-DETR object detection model as TensorRT engine (.trt)
  - Keypoint detection model as TensorRT engine (.trt)

### Installation
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Model Preparation Object Detection

1. **Obtain the RT-DETR model weights:**
    Download the pre-trained pytorch RT-DETR model weights from the [RT-DETR repository](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch).

2. **Convert the model weights to ONNX format:**
    Use the ONNX export script provided in the RT-DETR repository to convert the model weights to ONNX format:
    ```bash
    python tools/export_onnx.py -c [path/to/model/config.yml] -r [path/to/weights.pth] -o [output/path/model.onnx] --check
    ```

3. **Convert the ONNX models to TensorRT engines:**
    Use `trtexec` to convert the ONNX models to TensorRT engines:
    ```bash
    trtexec --onnx=[path/model.onnx] --fp16 --saveEngine=[output/path/model.trt]
    ```

### Model Training for Keypoint Detection Model
1. Download training data from [here](https://github.com/yastrebksv/TennisCourtDetector?tab=readme-ov-file)
2. Use training/keypoint_training.py to train
3. Convert to .onnx using training/keypoint_to_onnx.py
4. Convert to .trt using trtexec util 

## Usage

### Training

### Running the System
1. Start a video stream, e.g. with:
```bash
python streaming/start_mjpeg_stream.py [path/to/video.mp4]
```

2. Start the main tracking system:
```bash
python main.py
```

## Components

### Perception
- Object detection using [RT-DETRv2](https://github.com/lyuwenyu/RT-DETR/tree/main)
- Keypoint detection for player pose estimation
- Custom TensorRT inference (for TensorRT version 10.7.0) to speed up inference

### Minimap
- 2D tennis court visualization
- Projection of player and ball positions using homography
- Ball tracking with simple kinematic prediction model


## To make it better
![Difficult Conditions](videos/output_difficult.gif)

- Fine-tune object detection model to detect only players and balls
- Improve prediction of ball position when ball is not detected (e.g. Kalman filter)
- Estimate height of ball for more accurate projection on minimap
- Train a better keypoint detection model for mor accurate minimap
- Calculate player and game statistics based on detections prespective transformation
- use c++ instead of python to make it faster, because pre- and post-processing is costly and python overhead is rather slow

